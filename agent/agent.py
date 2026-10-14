"""An LLM agent that plays Luanti (VoxeLibre) until it holds a diamond.

    python -m agent.agent [--max-steps 80] [--keep 8]

The loop is the one from the lecture: advertise tools, let the model pick one,
execute it, feed the result back as the next observation, stop at the goal.

Design decisions (the three questions):

* What can it do? Coarse tools. `craft` places, opens and picks back up a
  crafting table by itself; `smelt` does the same with a furnace; `mine_ore`
  tunnels to the nearest visible ore and refuses ore its pickaxe cannot
  harvest. The model plans the tech tree; the code handles the fiddly bits.
* What can it see? Every tool result ends with a one-line STATE (position,
  HP, inventory, nearby hostiles), so the model never has to spend a call on
  `status` just to stay oriented. `status` adds the visible ore summary.
* What does it remember? The last `--keep` steps verbatim. Older steps are
  rewritten as one-line summaries, and the model can pin facts with `note`.
  The context therefore stays roughly constant in size however long it runs.

Safety is not left to the model. A guard inside the network pump interrupts
any tool when the bot dies or is being hurt at low hp, and a watchdog applies
the same reflex (dig down, seal the hole) during the 10-30 s the model spends
thinking, because the world does not pause for it.

The bot lives on luantibot.mcp_server's worker thread, which keeps stepping it
while the model is thinking, so the connection never idles out.
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import threading
import time

from .llm import LLM, ROOT, load_env

load_env()

from luantibot import Bot, BotError, recipes          # noqa: E402
from luantibot.client import INTERACT_ACTIVATE       # noqa: E402
from luantibot.mcp_server import SESSION             # noqa: E402

DIAMOND = "mcl_core:diamond"
TABLE = "mcl_crafting_table:crafting_table"
FURNACE = "mcl_furnaces:furnace"
COBBLE = "mcl_core:cobble"

PICK_TIER = {"mcl_tools:pick_wood": 1, "mcl_tools:pick_stone": 2,
             "mcl_tools:pick_iron": 3, "mcl_tools:pick_diamond": 4,
             "mcl_tools:pick_netherite": 5}

ORES = {
    "diamond": ["mcl_core:stone_with_diamond",
                "mcl_deepslate:deepslate_with_diamond"],
    "iron": ["mcl_core:stone_with_iron", "mcl_deepslate:deepslate_with_iron"],
    "coal": ["mcl_core:stone_with_coal", "mcl_deepslate:deepslate_with_coal"],
}
# pickaxe tier an ore needs before it drops anything
ORE_TIER = {"diamond": 3, "iron": 2, "coal": 1}

# each log species crafts into its own planks
PLANKS = {"mcl_core:tree": "mcl_core:wood",
          "mcl_core:birchtree": "mcl_core:birchwood",
          "mcl_core:sprucetree": "mcl_core:sprucewood",
          "mcl_core:acaciatree": "mcl_core:acaciawood",
          "mcl_core:darktree": "mcl_core:darkwood",
          "mcl_core:jungletree": "mcl_core:junglewood"}

FUELS = ["mcl_core:coal_lump", "mcl_core:charcoal_lump", *PLANKS.values(),
         *PLANKS.keys()]

# best first; raw chicken, rotten flesh and spider eyes are left out because
# they can poison
FOODS = ["mcl_mobitems:cooked_beef", "mcl_mobitems:cooked_porkchop",
         "mcl_mobitems:cooked_mutton", "mcl_mobitems:cooked_chicken",
         "mcl_mobitems:cooked_rabbit", "mcl_farming:potato_item_baked",
         "mcl_farming:bread", "mcl_core:apple", "mcl_mobitems:beef",
         "mcl_mobitems:porkchop", "mcl_mobitems:mutton", "mcl_mobitems:rabbit",
         "mcl_farming:carrot_item", "mcl_farming:sweet_berry",
         "mcl_farming:melon_item"]
PREY = ("cow", "pig", "sheep", "rabbit", "chicken")


class ToolFailed(Exception):
    """A tool could not do what was asked; the message goes to the model."""


# --------------------------------------------------------------------------
# helpers (all run on the bot thread)
# --------------------------------------------------------------------------

def short(name):
    return name.split(":", 1)[-1]


def pick_tier(bot):
    return max((PICK_TIER.get(n, 0) for n in bot.item_counts()), default=0)


def ore_kind(name):
    for kind, names in ORES.items():
        if name in names:
            return kind
    return None


def nearest(bot, names, radius=6):
    hits = bot.find_blocks(names, radius=radius)
    return tuple(hits[0]) if hits else None


def plank_counts(bot):
    counts = bot.item_counts()
    return {p: counts[p] for p in PLANKS.values() if counts.get(p)}


def resolve_recipe(bot, item):
    """Recipe rows for `item`, adapted to the wood species actually carried."""
    counts = bot.item_counts()
    if item in ("planks", *PLANKS.values()):
        logs = [l for l in PLANKS if counts.get(l)]
        if item in PLANKS.values():
            logs = [l for l in logs if PLANKS[l] == item] or logs
        if not logs:
            raise ToolFailed("no logs to make planks from; use gather_wood")
        return [[logs[0]]], False
    entry = recipes.get(item)
    if entry is None:
        raise ToolFailed(f"no recipe known for {item}. Known: planks, "
                         + ", ".join(recipes.RECIPES))
    rows, needs_table = entry
    if any(recipes.WOOD in row for row in rows):
        need = sum(row.count(recipes.WOOD) for row in rows)
        planks = sorted(plank_counts(bot).items(), key=lambda kv: -kv[1])
        if not planks or planks[0][1] < need:
            raise ToolFailed(f"{item} needs {need} planks of one kind; have "
                             f"{dict(planks) or 'none'} -- craft planks first")
        rows = [[planks[0][0] if c == recipes.WOOD else c for c in row]
                for row in rows]
    return rows, needs_table


def place_station(bot, item):
    """Put a crafting table or furnace down next to the bot; return its pos."""
    if bot.inventory.count(item) == 0:
        return None
    spot = bot.find_placement_spot()
    if spot is None:
        raise ToolFailed(f"nowhere to place {short(item)} here")
    if not bot.place_block(spot, item=item):
        raise ToolFailed(f"failed to place {short(item)} at {spot}")
    return spot


def collect_drops(bot, radius=5):
    """Walk over item drops nearby; they are only collected at close range."""
    for _ in range(4):
        drops = [e for e in bot.entities.values()
                 if e.name == "__builtin:item"
                 and math.dist(e.position, bot.position) <= radius]
        if not drops:
            return
        drop = min(drops, key=lambda e: math.dist(e.position, bot.position))
        bot.goto(drop.node_pos, tolerance=0, timeout=8, replan=False)
        bot.wait(0.7)


def pick_up(bot, pos):
    """Dig a station back into the inventory so it travels with the bot.

    Returns True only once the item is actually back in the inventory: in a
    tight tunnel the drop can land out of pickup range, so walk over to it.
    """
    item = bot.block_at(pos)
    before = bot.inventory.count(item)
    try:
        if not bot.dig(pos, timeout=20):
            return False
    except BotError:
        return False
    bot.wait(1.0)
    if bot.inventory.count(item) <= before:
        collect_drops(bot)
    return bot.inventory.count(item) > before


def tunnel_to(bot, target, max_steps=120):
    """Dig a two-high tunnel until `target` is within reach."""
    for _ in range(max_steps):
        if bot.in_reach(target):
            return True
        x, y, z = here = bot.node_pos
        tx, ty, tz = target
        if abs(tx - x) >= abs(tz - z) and tx != x:
            dx, dz = (1 if tx > x else -1), 0
        elif tz != z:
            dx, dz = 0, (1 if tz > z else -1)
        else:
            dx = dz = 0
        dy = -1 if ty < y - 1 else (1 if ty > y + 1 else 0)
        if dx == dz == 0:
            if dy >= 0:
                return bot.in_reach(target)
            dx = 1                         # stairs down rather than a pit
        nxt = (x + dx, y + dy, z + dz)
        cells = [nxt, (nxt[0], nxt[1] + 1, nxt[2])]
        if dy == 1:
            cells.append((x, y + 2, z))    # headroom to step up
        for c in cells:
            if bot.world.get_def(c) is None:
                raise ToolFailed(f"map not loaded at {c}")
            if not bot._opening_is_safe(c):
                raise ToolFailed(f"lava or water next to {c}; choose another target")
            if bot.world.is_walkable(c) and not bot.dig(c, timeout=20, approach=False):
                raise ToolFailed(f"could not dig {bot.block_at(c)} at {c}")
        bot.goto(nxt, tolerance=0, timeout=10, replan=False)
        bot.settle(timeout=5)
        if bot.node_pos == here:
            raise ToolFailed(f"stuck tunnelling at {here}")
    return bot.in_reach(target)


def delta(before, after):
    return {k: after[k] - before.get(k, 0) for k in after
            if after[k] != before.get(k, 0)}


def is_night(bot):
    t = bot.client.time_of_day           # 0 = midnight, 12000 = noon
    return t < 5000 or t > 19000


def state_line(bot):
    x, y, z = bot.node_pos
    inv = ", ".join(f"{short(k)} x{v}" for k, v in sorted(bot.item_counts().items()))
    mobs = collections.Counter(short(e.name) for e in bot.hostiles(12))
    mobs = ", ".join(f"{k} x{v}" for k, v in mobs.items()) or "none"
    return (f"STATE pos=({x},{y},{z}) hp={bot.hp}/20 pick_tier={pick_tier(bot)} "
            f"{'night' if is_night(bot) else 'day'} deaths={GUARD.deaths} "
            f"sky={'open' if sky_open(bot) else 'BLOCKED (enclosed)'} "
            f"hostiles<12m: {mobs} | inventory: {inv or 'empty'}")


# --------------------------------------------------------------------------
# the guard: interrupt any tool when the bot dies or is about to
# --------------------------------------------------------------------------

class Interrupted(Exception):
    """Raised from inside a running tool; deliberately not a BotError, so
    nothing in the library catches it on the way out."""


class Guard:
    min_hp = 7
    armed = False          # only while a tool runs
    watch_hp = True        # off while the bot is deliberately fighting
    reason = None
    deaths = 0
    last_hp = 20
    last_hurt = 0.0        # monotonic time hp last went down


GUARD = Guard()


def in_danger(bot):
    """Low hp, *still being hurt*, with hostiles near.

    Without the "still being hurt" part a bot that is already sheltering at
    low hp would be interrupted on every tool and could never go and eat.
    """
    return (bot.hp <= GUARD.min_hp
            and time.monotonic() - GUARD.last_hurt < 3.0
            and bool(bot.hostiles(10)))


def install_guard(bot):
    """Every long action pumps the network, so check for danger there."""
    pump = bot.client.pump

    def guarded_pump(timeout=0.05):
        alive = pump(timeout)
        g = GUARD
        if bot.hp < g.last_hp:
            g.last_hurt = time.monotonic()
        g.last_hp = bot.hp
        if g.armed and g.reason is None:
            if bot.dead:
                g.reason = "died"
            elif g.watch_hp and in_danger(bot):
                g.reason = "low_hp"
            if g.reason:
                raise Interrupted(g.reason)
        return alive

    bot.client.pump = guarded_pump

    def died():
        GUARD.deaths += 1
        print("   *** the bot died ***")
    bot.on("death", died)


SEALS = ["mcl_core:dirt", COBBLE, "mcl_core:stone", *PLANKS.values()]


def take_cover(bot, depth=3):
    """Dig straight down a few nodes and plug the hole overhead."""
    start_y = bot.node_pos[1]
    for _ in range(depth):
        if not bot._dig_down_one(bot.node_pos):
            break
    x, y, z = bot.node_pos
    sealed = False
    if y < start_y:
        block = next((b for b in SEALS if bot.inventory.count(b)), None)
        if block:
            try:
                sealed = bot.place_block((x, y + 2, z), item=block)
            except BotError:
                pass
    return {"dug_down": start_y - y, "sealed_above": sealed}


class Watchdog:
    """Protects the bot while the model is thinking.

    A model call takes 10-30 s, and the world does not pause for it. This
    occupies the bot thread for exactly that window, stepping the bot and
    applying the same reflexes the guard applies inside tools. What happened
    is reported to the model with the next tool result.
    """

    def __init__(self):
        self.stop = threading.Event()
        self.events = []
        self.holder = None

    def _job(self):
        bot = SESSION.require()
        covers = 0
        while not self.stop.is_set():
            bot.step()
            if bot.dead:
                self.events.append("you DIED; respawned at spawn, items "
                                   "dropped where you died")
                bot.wait_for(lambda: not bot.dead, timeout=30)
                bot.wait(2)
            elif in_danger(bot) and covers < 2:
                covers += 1
                hp = bot.hp
                self.events.append(f"attacked (hp {hp}); reflex took cover: "
                                   f"{take_cover(bot)}")
            time.sleep(0.05)

    def start(self):
        self.stop.clear()
        self.holder = {"done": threading.Event()}
        SESSION.jobs.put((self._job, self.holder))

    def finish(self):
        self.stop.set()
        self.holder["done"].wait(120)
        if "error" in self.holder:
            self.events.append(f"watchdog error: {self.holder['error']}")
        events, self.events = self.events, []
        return events


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

def t_status(bot):
    x, y, z = bot.node_pos
    visible = {}
    for kind, names in ORES.items():
        hits = bot.scan(names, limit=3)
        if hits:
            visible[kind] = [{"pos": list(p),
                              "distance": round(bot.distance_to(p), 1)}
                             for p in hits]
    return {"standing_on": bot.block_at((x, y - 1, z)),
            "lowest_y_loaded": bot.world.lowest_loaded_y(),
            "craft_grid": len(bot._inv_list("craft")),
            "visible_ore": visible or "none in the loaded map"}


def t_gather_wood(bot, count=4):
    ensure_outside(bot, "trees")
    have = lambda: sum(bot.inventory.count(l) for l in PLANKS)
    start, tried = have(), set()
    deadline = time.monotonic() + 240
    while have() - start < count and time.monotonic() < deadline:
        logs = [tuple(p) for p in bot.scan(list(PLANKS), limit=60)]
        logs = [p for p in logs if p not in tried]
        if not logs:
            break
        target = min(logs, key=bot.distance_to)
        tried.add(target)
        try:
            if bot.dig(target, timeout=30):
                bot.wait(1.5)
        except BotError:
            continue
    got = have() - start
    if got == 0:
        raise ToolFailed("found no reachable trees in the loaded map")
    return {"logs_gained": got}


def t_craft(bot, item, times=1):
    times = max(1, int(times))
    rows, needs_table = resolve_recipe(bot, item)
    before = bot.item_counts()
    table, placed = None, False
    if needs_table and len(bot._inv_list("craft")) < 9:
        table = nearest(bot, TABLE)
        if table is None:
            table = place_station(bot, TABLE)
            placed = table is not None
        if table is None:
            raise ToolFailed(f"{item} needs a crafting table; craft "
                             f"{TABLE} (4 planks) first")
        if bot.open_node(table) < 9:
            raise ToolFailed("crafting table did not open")
    kept = None
    try:
        stack = bot.craft(rows, times=times)
    except BotError as e:
        raise ToolFailed(str(e))
    finally:
        if placed:
            kept = pick_up(bot, table)
    if stack is None:
        raise ToolFailed("the server produced nothing from that recipe")
    out = {"crafted": delta(before, bot.item_counts())}
    if placed:
        out["table"] = "picked back up" if kept else f"left behind at {list(table)}"
    return out


def t_mine_to_depth(bot, target_y):
    if int(target_y) >= bot.node_pos[1]:
        # the library would report "reached" without moving, which once had
        # the model believe it had dug upward
        raise ToolFailed(f"you are already at y={bot.node_pos[1]}; this tool "
                         "only digs DOWN. Use climb_out to go up.")
    before = bot.item_counts()
    reached = bot.mine_to_depth(int(target_y), timeout=600)
    out = {"reached": reached, "gained": delta(before, bot.item_counts())}
    if not reached:
        out["hint"] = ("stopped early (blocked, liquid, low hp or timeout); "
                       "try again, or goto a few nodes sideways first")
    return out


def t_find_ore(bot, ore):
    names = ORES.get(ore, [ore])
    hits = bot.scan(names, limit=8)
    return {"ore": ore, "found": [
        {"pos": list(p), "node": short(bot.block_at(p)),
         "distance": round(bot.distance_to(p), 1)} for p in hits]}


def t_mine_ore(bot, ore, count=1):
    kind = ore if ore in ORES else ore_kind(ore)
    names = ORES.get(kind, [ore])
    need = ORE_TIER.get(kind, 1)
    if pick_tier(bot) < need:
        raise ToolFailed(f"{ore} needs a pickaxe of tier {need}+ to drop "
                         f"anything; you have tier {pick_tier(bot)}")
    before, mined, skipped = bot.item_counts(), 0, set()
    deadline = time.monotonic() + 300
    while mined < int(count) and time.monotonic() < deadline:
        hits = [tuple(p) for p in bot.scan(names, limit=10)]
        hits = [p for p in hits if p not in skipped]
        if not hits:
            break
        target = min(hits, key=bot.distance_to)
        try:
            if not bot.in_reach(target) and not bot.approach(target, timeout=30):
                tunnel_to(bot, target)
            if bot.dig(target, timeout=30):
                mined += 1
                bot.wait(1.5)
            else:
                skipped.add(target)
        except (BotError, ToolFailed) as e:
            skipped.add(target)
            if len(skipped) >= 4:
                raise ToolFailed(f"gave up after 4 unreachable {ore}: {e}")
    if mined == 0:
        raise ToolFailed(f"no {ore} visible or reachable; explore or descend")
    return {"mined": mined, "gained": delta(before, bot.item_counts())}


def t_smelt(bot, item, count=1, fuel=None):
    count = max(1, int(count))
    have = bot.item_counts()
    if have.get(item, 0) < count:
        raise ToolFailed(f"have {have.get(item, 0)} {item}, need {count}")
    fuel = fuel or next((f for f in FUELS if have.get(f)), None)
    if fuel is None or not have.get(fuel):
        raise ToolFailed("no fuel: coal, charcoal, planks or logs")
    furnace, placed = nearest(bot, [FURNACE, "mcl_furnaces:furnace_active"]), False
    if furnace is None:
        furnace = place_station(bot, FURNACE)
        placed = furnace is not None
    if furnace is None:
        raise ToolFailed(f"no furnace; craft {FURNACE} (8 cobble) first")
    try:
        # bot.smelt loads count//8+1 fuel, which is right for coal but leaves
        # planks (1.5 items each) short, so top it up first
        per_item = 8 if "coal" in fuel else 1.5
        extra = max(0, math.ceil(count / per_item) - (count // 8 + 1))
        extra = min(extra, have[fuel] - 1)
        if extra > 0:
            bot.put_into_node(furnace, fuel, "fuel", extra)
        out = bot.smelt(furnace, item, fuel, count=count, timeout=count * 12 + 40)
    finally:
        kept = pick_up(bot, furnace) if placed else None
    where = (list(furnace) if not placed else
             "picked back up" if kept else f"left behind at {list(furnace)}")
    return {"output": out, "furnace": where}


def t_goto(bot, x, y, z):
    ok = bot.goto((int(x), int(y), int(z)), tolerance=1, timeout=90)
    if not ok:
        # a silent {"arrived": false} once kept the model retrying goto for
        # 15 steps from inside a sealed shelter
        hint = ("you are enclosed (solid block overhead): use climb_out"
                if not sky_open(bot) else
                "no walkable path; goto never digs -- try mine_ore/climb_out, "
                "or a nearer target")
        raise ToolFailed(f"did not arrive; now at {list(bot.node_pos)}. {hint}")
    return {"arrived": True}


def t_defend(bot):
    GUARD.watch_hp = False                 # fighting hurts; only death aborts
    killed = bot.defend(radius=8, timeout=20)
    return {"killed": killed,
            "remaining": [short(e.name) for e in bot.hostiles(8)]}


def t_take_cover(bot):
    return take_cover(bot)


def sky_open(bot, height=8):
    x, y, z = bot.node_pos
    return all(not bot.world.is_walkable((x, y + dy, z))
               for dy in range(2, height + 1))


def climb_out(bot, max_steps=10):
    """Cut a staircase up until nothing is overhead (e.g. leaving a shelter).

    The pathfinder only walks, so a bot that has sealed itself in can not
    otherwise get back to the surface.
    """
    for _ in range(max_steps):
        if sky_open(bot):
            return True
        x, y, z = bot.node_pos
        for dx, dz in Bot.HEADINGS:
            nxt = (x + dx, y + 1, z + dz)
            cells = [(x, y + 2, z), nxt, (nxt[0], nxt[1] + 1, nxt[2])]
            if not all(bot._opening_is_safe(c) for c in cells):
                continue
            if all(not bot.world.is_walkable(c) or bot.dig(c, timeout=20,
                                                         approach=False)
                   for c in cells):
                bot.goto(nxt, tolerance=0, timeout=8, replan=False)
                bot.settle(timeout=4)
                break
        if bot.node_pos == (x, y, z):
            return False
    return sky_open(bot)


def ensure_outside(bot, why):
    """Surface activities start by leaving any shelter or tunnel."""
    if not sky_open(bot) and not climb_out(bot, max_steps=40):
        raise ToolFailed(f"enclosed and could not climb out (stone needs a "
                         f"pickaxe); {why} need the open surface")


def t_climb_out(bot):
    y0 = bot.node_pos[1]
    ok = climb_out(bot, max_steps=40)
    out = {"open_sky": ok, "climbed": bot.node_pos[1] - y0}
    if not ok:
        out["hint"] = "blocked: stone needs a pickaxe, or liquid is in the way"
    return out


def t_hunt_food(bot, count=2):
    ensure_outside(bot, "animals")
    before, killed, tried = bot.item_counts(), 0, set()
    deadline = time.monotonic() + 180
    while killed < int(count) and time.monotonic() < deadline:
        prey = [e for e in bot.entities.values()
                if not e.is_player and e.id not in tried
                and any(p in e.name for p in PREY)
                and math.dist(e.position, bot.position) <= 80]
        if not prey:
            break
        target = min(prey, key=lambda e: math.dist(e.position, bot.position))
        tried.add(target.id)
        if bot.fight(target, timeout=60, max_distance=90):
            killed += 1
            bot.wait(1.0)
            collect_drops(bot, radius=6)
    if killed == 0:
        raise ToolFailed("could not catch any cow, pig, sheep or chicken "
                         "within 80 nodes")
    return {"killed": killed, "gained": delta(before, bot.item_counts())}


def t_eat(bot):
    eaten = collections.Counter()
    for food in FOODS:
        while bot.inventory.count(food) and sum(eaten.values()) < 8:
            bot.equip(bot.inventory.first_slot(food))
            have = bot.inventory.count(food)
            bot.client.interact(INTERACT_ACTIVATE)      # right-click the air
            if not bot.wait_for(lambda: bot.inventory.count(food) < have,
                                timeout=3.0):
                # VoxeLibre refuses food once the hunger bar is full
                return {"ate": dict(eaten), "stopped": "full (or not edible)"}
            eaten[food] += 1
            bot.wait(0.5)
    if not eaten:
        raise ToolFailed("nothing edible in inventory; use hunt_food")
    return {"ate": dict(eaten)}


def wait_for_day(bot, limit=900):
    bot.wait_for(lambda: not is_night(bot), timeout=limit)
    return not is_night(bot)


def t_rest(bot, seconds=30, until_day=False):
    """Stand still (ideally sheltered) and let health regenerate."""
    hp0 = bot.hp
    if until_day:
        day = wait_for_day(bot)
        return {"hp_before": hp0, "hp_after": bot.hp,
                "now": "day" if day else "still night (gave up after 15 min)"}
    bot.wait(min(max(float(seconds), 5.0), 90.0))
    return {"hp_before": hp0, "hp_after": bot.hp,
            "hint": "" if bot.hp > hp0 else
            "no regeneration: you are probably hungry -- eat first"}


def _p(t, d):
    return {"type": t, "description": d}


TOOLS = {
    "status": (t_status, "Extra detail: ground, loaded depth, craft grid size "
               "and nearest visible coal/iron/diamond ore.", {}, []),
    "gather_wood": (t_gather_wood, "Chop the nearest trees by hand.",
                    {"count": _p("integer", "logs to collect, default 4")}, []),
    "craft": (t_craft, "Craft an item. Handles the crafting table itself "
              "(places one from inventory and picks it back up). Use "
              "item='planks' for planks from any log. Known: planks, "
              "mcl_core:stick, " + ", ".join(i for i in recipes.RECIPES
                                             if i != recipes.WOOD
                                             and i != recipes.STICK),
              {"item": _p("string", "item name"),
               "times": _p("integer", "repeat the recipe, default 1")},
              ["item"]),
    "mine_to_depth": (t_mine_to_depth, "Dig a staircase down to a y level, "
                      "collecting what is dug and fighting mobs. Needs a "
                      "pickaxe. Avoids lava and water.",
                      {"target_y": _p("integer", "y to reach")}, ["target_y"]),
    "find_ore": (t_find_ore, "List the nearest visible ore of a kind.",
                 {"ore": _p("string", "diamond, iron, coal, or a node name")},
                 ["ore"]),
    "mine_ore": (t_mine_ore, "Tunnel to and mine the nearest visible ore of a "
                 "kind, collecting the drops. Refuses ore your pickaxe cannot "
                 "harvest.",
                 {"ore": _p("string", "diamond, iron, coal, or a node name"),
                  "count": _p("integer", "how many to mine, default 1")},
                 ["ore"]),
    "smelt": (t_smelt, "Smelt items in a furnace (places your furnace and "
              "picks it back up). Fuel defaults to coal, then planks/logs.",
              {"item": _p("string", "e.g. mcl_raw_ores:raw_iron"),
               "count": _p("integer", "how many"),
               "fuel": _p("string", "optional fuel item")}, ["item"]),
    "goto": (t_goto, "Walk to a coordinate along a path (does not dig).",
             {"x": _p("integer", "x"), "y": _p("integer", "y"),
              "z": _p("integer", "z")}, ["x", "y", "z"]),
    "defend": (t_defend, "Fight hostile mobs within 8 nodes.", {}, []),
    "climb_out": (t_climb_out, "Dig a staircase UP until there is open sky. "
                  "Use it to leave a shelter or tunnel; goto cannot dig.",
                  {}, []),
    "take_cover": (t_take_cover, "Dig 3 nodes straight down and seal the "
                   "hole above: an emergency shelter from mobs.", {}, []),
    "hunt_food": (t_hunt_food, "Kill the nearest cows/pigs/sheep/chickens "
                  "and collect the meat.",
                  {"count": _p("integer", "animals to kill, default 2")}, []),
    "eat": (t_eat, "Eat food from the inventory (best first) until full.",
            {}, []),
    "rest": (t_rest, "Wait in place so health regenerates (needs a full "
             "hunger bar: eat first). Safer when sheltered.",
             {"seconds": _p("integer", "5-90, default 30"),
              "until_day": _p("boolean", "wait (up to 15 min) until morning "
                              "instead; use from inside a shelter")}, []),
    "note": (None, "Save a short fact to long-term memory (coordinates, "
             "plans, what failed). Older turns are summarised, notes are not.",
             {"text": _p("string", "the note")}, ["text"]),
}

TOOL_SCHEMAS = [{"type": "function", "function": {
    "name": name, "description": desc,
    "parameters": {"type": "object", "properties": props, "required": req}}}
    for name, (_fn, desc, props, req) in TOOLS.items()]


SYSTEM = """You control a player in Luanti running VoxeLibre, a Minecraft clone, through tools.
GOAL: get mcl_core:diamond into your inventory. The run ends as soon as it arrives.

Game facts:
- Tech tree: logs by hand -> planks -> sticks + crafting table -> wooden pickaxe (3 planks, 2 sticks) -> mine stone for cobble -> stone pickaxe (3 cobble, 2 sticks) and furnace (8 cobble) -> iron ore drops mcl_raw_ores:raw_iron -> smelt 3 into mcl_core:iron_ingot -> iron pickaxe (3 ingots, 2 sticks) -> diamond.
- Keep spare sticks and planks: every pickaxe needs 2 sticks, and planks are fuel.
- Diamond ore is at y=-62..-50 (rarely a little higher). It drops a diamond ONLY when mined with an iron pickaxe or better; otherwise it is destroyed for nothing. Iron ore needs a stone pickaxe or better.
- Iron and coal are common underground. mine_to_depth also collects the cobble it digs.
- The server only sends map near you, so distant ore is invisible: descend or move to reveal more.
- Hostile mobs roam the surface at night; underground is safer. Dying drops your inventory and respawns you at the surface spawn.
- Any tool is interrupted if you die, or if hp drops to 7 while being attacked; the harness then takes cover for you automatically, even while you are thinking. At night on the surface, get underground (mine_to_depth or take_cover) rather than wandering.
- The spawn point swarms with mobs at night. If you respawn at night, take_cover immediately, then rest(until_day=true), then climb_out.
- STATE shows sky=BLOCKED when something solid is overhead. goto only walks and never digs: to go up use climb_out, to go down use mine_to_depth.
- Health only regenerates when you are well fed. Get meat with hunt_food (cows, pigs, sheep near the surface, in daylight), eat it, then rest. Cooked meat (smelt it) is worth twice as much. Keep some food with you before going deep.

How to work:
- Each tool result ends with a STATE line (position, hp, pickaxe tier, inventory). Trust it.
- Prefer one coarse tool over many small ones. Think briefly, then act.
- Older turns are compressed into one-line summaries; use note() for anything you must remember."""


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------

THINK = re.compile(r"<think>.*?</think>", re.S)


class Memory:
    """Recent steps verbatim, older steps as one-liners, plus pinned notes."""

    def __init__(self, keep=8):
        self.keep = keep
        self.steps = []          # {"assistant": msg, "results": [...], "line": str}
        self.notes = []

    def add(self, assistant, results, line):
        self.steps.append({"assistant": assistant, "results": results, "line": line})

    def messages(self):
        old, recent = self.steps[:-self.keep], self.steps[-self.keep:]
        context = ["Begin. Get a diamond."]
        if self.notes:
            context.append("Your notes:\n" + "\n".join(f"- {n}" for n in self.notes))
        if old:
            context.append("Earlier steps (summarised):\n"
                           + "\n".join(s["line"] for s in old))
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n\n".join(context)}]
        for i, s in enumerate(recent):
            a = dict(s["assistant"])
            # reasoning is only worth resending for the latest couple of steps
            if i < len(recent) - 2 and a.get("content"):
                a["content"] = THINK.sub("", a["content"]).strip()
            msgs.append(a)
            for call_id, content in s["results"]:
                if call_id is None:
                    msgs.append({"role": "user", "content": content})
                else:
                    msgs.append({"role": "tool", "tool_call_id": call_id,
                                 "content": content})
        return msgs


# --------------------------------------------------------------------------
# loop
# --------------------------------------------------------------------------

def on_bot(fn, timeout=900):
    return SESSION.submit(lambda: fn(SESSION.require()), timeout=timeout)


def connect():
    def job():
        bot = Bot(os.environ["LUANTI_HOST"], int(os.environ["LUANTI_PORT"]),
                  os.environ["LUANTI_USER"], os.environ.get("LUANTI_PASSWORD", ""))
        bot.connect(timeout=90)
        install_guard(bot)
        SESSION.bot = bot
        bot.wait(10)
        bot.clear_craft_grid()
    SESSION.submit(job, timeout=180)


def has_diamond():
    return on_bot(lambda b: b.inventory.count(DIAMOND) > 0, timeout=30)


def run_tool(memory, name, args):
    if name == "note":
        memory.notes.append(str(args.get("text", ""))[:300])
        return {"saved": True}
    if name not in TOOLS:
        raise ToolFailed(f"unknown tool {name}")
    fn = TOOLS[name][0]

    def job(bot):
        GUARD.armed, GUARD.watch_hp, GUARD.reason = True, True, None
        try:
            return fn(bot, **args)
        except Interrupted:
            pass
        finally:
            GUARD.armed = False
        # the tool was cut short: say why, and react without asking the model
        if GUARD.reason == "died":
            bot.wait_for(lambda: not bot.dead, timeout=30)
            bot.wait(3)
            raise ToolFailed(f"interrupted: you DIED during {name} and "
                             "respawned at spawn; your items were dropped "
                             "where you died")
        cover = take_cover(bot)
        raise ToolFailed(f"interrupted: hp fell to {bot.hp} with hostiles "
                         f"near during {name}. Reflex took cover: {cover}")

    return on_bot(job)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--keep", type=int, default=8,
                    help="recent steps kept verbatim in the context")
    ap.add_argument("--wait-for-day", action="store_true",
                    help="if it is night, shelter and wait for morning "
                         "before the model starts")
    opts = ap.parse_args()

    llm = LLM()
    memory = Memory(keep=opts.keep)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log_path = os.path.join(ROOT, "logs", time.strftime("run-%Y%m%d-%H%M%S.jsonl"))
    log = open(log_path, "w", encoding="utf-8")
    t_start = time.monotonic()

    def record(**kw):
        kw["t"] = round(time.monotonic() - t_start, 1)
        log.write(json.dumps(kw, default=str) + "\n")
        log.flush()

    print(f"connecting to {os.environ['LUANTI_HOST']}:{os.environ['LUANTI_PORT']} ...")
    connect()
    print(on_bot(state_line, 30))
    if opts.wait_for_day and on_bot(is_night, 30):
        print("night: taking cover and waiting for morning (no model calls) ...")
        print("  cover:", on_bot(take_cover, 120))
        on_bot(wait_for_day, 1000)
        print(on_bot(state_line, 30))
        t_start = time.monotonic()         # the run starts at sunrise
    watchdog = Watchdog()
    nudges = 0
    success = False
    try:
        for step in range(1, opts.max_steps + 1):
            watchdog.start()
            try:
                reply = llm.chat(memory.messages(), tools=TOOL_SCHEMAS)
            finally:
                events = watchdog.finish()
            # what happened while the model was thinking goes in front of
            # the next observation, so it is never silently lost
            meanwhile = ("WHILE YOU WERE THINKING: " + "; ".join(events) + "\n"
                         if events else "")
            for e in events:
                print(f"   !! while thinking: {e}")
            text = THINK.sub("", reply.get("content") or "").strip()
            calls = reply.get("tool_calls") or []
            assistant = {"role": "assistant", "content": reply.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            if text:
                print(f"\n[{step}] model: {text[:300]}")

            if not calls:
                nudges += 1
                memory.add(assistant, [(None, meanwhile + "Use a tool to act.")],
                           f"#{step} (no tool call)")
                record(step=step, text=text, tool=None)
                if nudges >= 3:
                    print("model stopped calling tools; giving up")
                    break
                continue
            nudges = 0

            results, lines = [], []
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"[{step}] -> {name}({json.dumps(args)})", flush=True)
                t0 = time.monotonic()
                try:
                    out = run_tool(memory, name, args)
                    ok = True
                except Exception as e:     # anything a tool raises goes back to the model
                    msg = str(e)
                    for prefix in ("ToolFailed: ", "BotError: "):
                        msg = msg.removeprefix(prefix)
                    out, ok = {"error": msg}, False
                state = on_bot(state_line, 30)
                content = json.dumps(out, default=str)[:2000] + "\n" + state
                if meanwhile and not results:
                    content = meanwhile + content
                results.append((call["id"], content))
                summary = json.dumps(out, default=str)[:140]
                lines.append(f"{name}({json.dumps(args)}) -> {summary}")
                print(f"     {'ok ' if ok else 'ERR'} {summary}\n     {state}")
                record(step=step, text=text, tool=name, args=args, ok=ok,
                       result=out, state=state, secs=round(time.monotonic() - t0, 1),
                       tokens_in=llm.prompt_tokens, tokens_out=llm.completion_tokens)
                if has_diamond():
                    success = True
                    break
            # every tool call needs an answer, even ones skipped by the goal
            for call in calls[len(results):]:
                results.append((call["id"], "skipped: goal reached"))
            memory.add(assistant, results, f"#{step} " + "; ".join(lines))
            if success:
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        elapsed = time.monotonic() - t_start
        print(f"\n{'DIAMOND OBTAINED' if success else 'no diamond'} after "
              f"{elapsed / 60:.1f} min, {llm.calls} model calls, "
              f"{llm.prompt_tokens} tokens in / {llm.completion_tokens} out")
        print("log:", log_path)
        record(event="end", success=success, minutes=round(elapsed / 60, 1),
               calls=llm.calls, tokens_in=llm.prompt_tokens,
               tokens_out=llm.completion_tokens)
        log.close()
        try:
            on_bot(lambda b: b.disconnect(), 30)
        except Exception:
            pass


if __name__ == "__main__":
    main()
