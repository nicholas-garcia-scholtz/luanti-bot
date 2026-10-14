"""Luanti protocol client: login, packet dispatch, and player actions.

Movement note: in Luanti the *client* is authoritative over its own position.
The server does not simulate walking from key presses -- it only sanity-checks
how far you moved (anticheat). So the bot moves simply by sending
TOSERVER_PLAYERPOS with updated coordinates at a believable speed, which is the
same thing Mineflayer does on the Minecraft side.
"""
import struct
import time

from . import commands as C
from .auth import (AUTH_MECHANISM_FIRST_SRP, AUTH_MECHANISM_LEGACY_PASSWORD,
                   AUTH_MECHANISM_SRP, describe_mechs, translate_password)
from .buffer import Reader, Writer
from .connection import Connection, Disconnected
from .srp import SRPUser, create_verifier

# Luanti's internal length unit: 1 node = BS units. Wire positions are
# BS-units * 100, i.e. nodes * 1000.
BS = 10.0
POS_SCALE = 100.0

SER_FMT_VER = 29
PROTO_MIN = 37
PROTO_MAX = 48
VERSION = (5, 16, 1)
VERSION_STRING = "5.16.1"
FORMSPEC_API_VERSION = 8

# InteractAction, src/network/networkprotocol.h
INTERACT_START_DIGGING = 0
INTERACT_STOP_DIGGING = 1
INTERACT_DIGGING_COMPLETED = 2
INTERACT_PLACE = 3
INTERACT_USE = 4
INTERACT_ACTIVATE = 5

POINTEDTHING_NOTHING = 0
POINTEDTHING_NODE = 1
POINTEDTHING_OBJECT = 2

ACCESS_DENIED_REASONS = {
    0: "wrong password", 1: "unexpected data", 2: "singleplayer only",
    3: "wrong version", 4: "wrong character in name", 5: "name too long",
    6: "name is empty", 7: "name not allowed", 8: "server is full",
    9: "banned", 10: "shutting down", 11: "crash", 12: "custom",
    13: "too many users", 14: "name taken", 15: "duplicate login",
}


class LoginError(Exception):
    pass


class LuantiClient:
    """Speaks the Luanti application protocol on top of a Connection."""

    def __init__(self, host, port=30000, username="bot", password="",
                 lang="en", verbose=False):
        self.host, self.port = host, port
        self.username, self.password = username, password
        self.lang = lang
        self.verbose = verbose

        self.conn = None
        self.srp = None
        self.chosen_mech = None

        # session state
        self.proto_ver = 0
        self.ser_ver = SER_FMT_VER
        self.map_seed = 0
        self.access_denied = None
        self.authenticated = False
        self.ready = False

        # player state
        self.position = [0.0, 0.0, 0.0]      # node coordinates
        self.velocity = [0.0, 0.0, 0.0]
        self.pitch = 0.0
        self.yaw = 0.0
        self.hp = 20
        self.breath = 11
        self.wield_index = 0
        self.fov_byte = 255                   # see _write_player_pos
        self.wanted_range = 12                # map blocks
        self.privileges = set()
        self.csm_restriction_flags = 0
        self.csm_restriction_noderange = 0
        self.movement = {}
        self.time_of_day = 0

        # world / definitions -- filled in by the World and NodeDefs modules
        self.world = None
        self.nodedefs = None
        self.itemdefs_raw = None
        self.itemdefs = None
        self.inventory_raw = None
        self.inventory = None
        self.players = set()
        from .entities import EntityTracker
        self.entities = EntityTracker()

        self.auto_respawn = True
        self.dead = False
        self._died_at = None
        self.on_death = []
        self.current_formspec = None
        self._got_nodedef = False
        self._got_itemdef = False
        self._pending_gotblocks = []
        self._last_pos_send = 0.0
        self.handlers = {}                    # command id -> list of callbacks
        self.on_chat = []
        self.on_entity_add = []
        self.on_nodemeta = []
        self.on_block = []

    # ------------------------------------------------------------------ utils

    def log(self, *a):
        if self.verbose:
            print("[client]", *a)

    def send_command(self, cmd, writer=None, channel=0, reliable=True):
        body = struct.pack(">H", cmd) + (writer.bytes() if writer else b"")
        self.conn.send(body, channel=channel, reliable=reliable)

    def on(self, cmd, fn):
        self.handlers.setdefault(cmd, []).append(fn)

    # ------------------------------------------------------------------ login

    def connect(self, timeout=45.0):
        self.conn = Connection(self.host, self.port)
        self.conn.connect()
        self.log("peer id", self.conn.peer_id)

        w = Writer().u8(SER_FMT_VER).u16(0).u16(PROTO_MIN).u16(PROTO_MAX)
        w.string(self.username)
        self.send_command(C.TOSERVER_INIT, w)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            alive = self.pump(0.2)
            # The server sends ACCESS_DENIED and then DISCO, so check the
            # reason first or it is lost behind "server sent DISCO".
            if self.access_denied:
                raise LoginError(self.access_denied)
            if not alive:
                raise Disconnected(self.conn.disconnect_reason or "connection lost")
            if self.ready:
                return True
        raise LoginError("timed out during login")

    def pump(self, timeout=0.05):
        """Process pending packets. Returns False once the link is dead."""
        if not self.conn.running:
            return False
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            pkt = self.conn.recv(timeout=max(0.0, remaining))
            if pkt is None:
                break
            self._dispatch(pkt)
            if not self.conn.running:
                return False
        self._flush_gotblocks()
        if (self.auto_respawn and self.dead and self._died_at
                and time.monotonic() - self._died_at > 2.0):
            # The death formspec is the normal trigger; this is the fallback
            # for servers that never send one.
            self._died_at = time.monotonic() + 3.0
            self.respawn()
        return self.conn.running

    def _dispatch(self, pkt):
        if len(pkt) < 2:
            return
        cmd = struct.unpack(">H", pkt[:2])[0]
        r = Reader(pkt, 2)
        fn = getattr(self, "_h_%02x" % cmd, None)
        if fn:
            try:
                fn(r)
            except Exception as e:
                self.log(f"error handling {C.TOCLIENT_NAMES.get(cmd, cmd)}: {e!r}")
        elif self.verbose:
            self.log("unhandled", C.TOCLIENT_NAMES.get(cmd, hex(cmd)), len(pkt))
        for cb in self.handlers.get(cmd, ()):
            cb(Reader(pkt, 2))

    # --------------------------------------------------------- login handlers

    def _h_02(self, r):                       # TOCLIENT_HELLO
        self.ser_ver = r.u8()
        r.u16()                               # compression mode, unused
        self.proto_ver = r.u16()
        mechs = r.u32()
        self.log(f"hello: ser={self.ser_ver} proto={self.proto_ver} "
                 f"auth={describe_mechs(mechs)}")
        self._start_auth(mechs)

    def _start_auth(self, mechs):
        name_v = self.username.lower()
        if mechs & AUTH_MECHANISM_FIRST_SRP:
            # No account yet: register by uploading a fresh verifier.
            self.chosen_mech = AUTH_MECHANISM_FIRST_SRP
            salt, verifier = create_verifier(name_v, self.password)
            w = Writer().string(salt).string(verifier).bool8(True)
            self.send_command(C.TOSERVER_FIRST_SRP, w)
            return

        if mechs & AUTH_MECHANISM_SRP:
            self.chosen_mech = AUTH_MECHANISM_SRP
            secret, based_on = self.password, 1
        elif mechs & AUTH_MECHANISM_LEGACY_PASSWORD:
            self.chosen_mech = AUTH_MECHANISM_LEGACY_PASSWORD
            secret, based_on = translate_password(name_v, self.password), 0
        else:
            raise LoginError(f"no supported auth mechanism (0x{mechs:x})")

        self.srp = SRPUser(self.username, secret, username_verifier=name_v)
        A = self.srp.start_authentication()
        w = Writer().string(A).u8(based_on)
        self.send_command(C.TOSERVER_SRP_BYTES_A, w)

    def _h_60(self, r):                       # TOCLIENT_SRP_BYTES_S_B
        salt, B = r.string(), r.string()
        M = self.srp.process_challenge(salt, B)
        self.send_command(C.TOSERVER_SRP_BYTES_M, Writer().string(M))

    def _h_03(self, r):                       # TOCLIENT_AUTH_ACCEPT
        pos = r.v3f32()
        self.position = [pos[0] / BS, pos[1] / BS, pos[2] / BS]
        self.map_seed = r.u64()
        self.send_interval = r.f32()
        self.authenticated = True
        self.log(f"auth accepted; spawn={self.position} seed={self.map_seed}")
        self.send_command(C.TOSERVER_INIT2, Writer().string(self.lang))

    def _h_0a(self, r):                       # TOCLIENT_ACCESS_DENIED
        code = r.u8()
        reason = ACCESS_DENIED_REASONS.get(code, f"code {code}")
        extra = ""
        if r.remaining() >= 2:
            try:
                extra = r.text()
            except Exception:
                pass
        self.access_denied = f"{reason}{': ' + extra if extra else ''}"
        self.log("access denied:", self.access_denied)

    def _h_35(self, r):                       # TOCLIENT_ACCESS_DENIED_LEGACY
        self.access_denied = r.wide_string()

    # ------------------------------------------------- definitions and media

    def _h_3c(self, r):                       # TOCLIENT_ANNOUNCE_MEDIA
        count = r.u16()
        # A bot needs no textures or sounds, so request nothing.
        self.log(f"server announced {count} media files; requesting none")
        self.send_command(C.TOSERVER_REQUEST_MEDIA, Writer().u16(0))
        self._maybe_ready()

    def _h_3a(self, r):                       # TOCLIENT_NODEDEF
        blob = r.long_string()
        from .nodedef import NodeDefs
        self.nodedefs = NodeDefs.parse(blob, self.ser_ver)
        from .world import World
        self.world = World(self.nodedefs)
        air = self.nodedefs.id_of("air")
        if air is not None:
            self.world.air_id = air
        self.log(f"nodedef: {len(self.nodedefs.by_id)} node types "
                 f"({self.nodedefs.inexact} inexact)")
        self._got_nodedef = True
        self._maybe_ready()

    def _h_3d(self, r):                       # TOCLIENT_ITEMDEF
        from .itemdef import ItemDefs
        blob = r.long_string()
        self.itemdefs_raw = blob
        try:
            self.itemdefs = ItemDefs.parse(blob, self.ser_ver)
            self.log(f"itemdef: {len(self.itemdefs.by_name)} items "
                     f"({self.itemdefs.inexact} unparsed)")
        except Exception as e:
            self.log(f"failed to parse itemdefs: {e!r}")
        self._got_itemdef = True
        self._maybe_ready()

    def _maybe_ready(self):
        if self.ready or not (self._got_nodedef and self._got_itemdef):
            return
        w = (Writer().u8(VERSION[0]).u8(VERSION[1]).u8(VERSION[2]).u8(0)
             .string(VERSION_STRING).u16(FORMSPEC_API_VERSION))
        self.send_command(C.TOSERVER_CLIENT_READY, w)
        self.ready = True
        self.log("sent CLIENT_READY")
        self.send_playerpos(force=True)

    # -------------------------------------------------------- world and state

    def _h_20(self, r):                       # TOCLIENT_BLOCKDATA
        bp = r.v3s16()
        if self.world is not None:
            try:
                self.world.handle_blockdata(bp, r, self.ser_ver)
            except Exception as e:
                # Never drop a block silently: an undecodable block leaves a
                # hole in the map that pathfinding will quietly route around.
                self.world.decode_errors += 1
                self.world.last_decode_error = (bp, repr(e))
                self.log(f"failed to decode block {bp}: {e!r}")
        self._pending_gotblocks.append(bp)
        for cb in self.on_block:
            cb(bp)

    def _h_21(self, r):                       # TOCLIENT_ADDNODE
        p = r.v3s16()
        param0, param1, param2 = r.u16(), r.u8(), r.u8()
        if self.world is not None:
            self.world.set_node(p, param0, param1, param2)

    def _h_59(self, r):                       # TOCLIENT_NODEMETA_CHANGED
        # Pushed when a node's metadata changes (a furnace burning, a sign
        # edited). Unlike the copy inside a map block, positions here are
        # absolute rather than an index within the block.
        from .compress import decompress_stream
        blob = r.long_string()
        plain, _ = decompress_stream(blob, self.ser_ver)
        rr = Reader(plain)
        if rr.remaining() < 1:
            return
        version = rr.u8()
        if version == 0:
            return
        if version != 2:
            self.log(f"unsupported node metadata version {version}")
            return
        count = rr.u16()
        from .mapblock import read_inventory_text
        from .inventory import Inventory
        for _ in range(count):
            pos = rr.v3s16()
            fields = {}
            for _ in range(rr.u32()):
                key = rr.text()
                value = rr.long_string()
                rr.u8()                       # private flag
                fields[key] = value
            # the node's inventory follows the variables, as plain text
            inv = None
            try:
                text = read_inventory_text(rr)
                if text:
                    inv = Inventory.parse(text.decode("utf-8", "replace"))
            except Exception:
                pass
            if self.world is not None:
                self.world.set_meta(pos, fields, inv)
            for cb in self.on_nodemeta:
                cb(pos, fields)

    def _h_22(self, r):                       # TOCLIENT_REMOVENODE
        p = r.v3s16()
        if self.world is not None:
            self.world.set_node(p, self.world.air_id, 0, 0)

    def _flush_gotblocks(self):
        if not self._pending_gotblocks:
            return
        batch, self._pending_gotblocks = self._pending_gotblocks[:255], \
            self._pending_gotblocks[255:]
        w = Writer().u8(len(batch))
        for bp in batch:
            w.v3s16(bp)
        self.send_command(C.TOSERVER_GOTBLOCKS, w, channel=2)

    def _h_34(self, r):                       # TOCLIENT_MOVE_PLAYER
        pos = r.v3f32()
        self.position = [pos[0] / BS, pos[1] / BS, pos[2] / BS]
        self.pitch, self.yaw = r.f32(), r.f32()
        self.log(f"server moved us to {self.position}")

    def _h_5d(self, r):                       # TOCLIENT_MOVE_PLAYER_REL
        d = r.v3f32()
        self.position = [self.position[i] + d[i] / BS for i in range(3)]

    def _h_33(self, r):                       # TOCLIENT_HP
        self.hp = r.u16()
        if self.hp == 0 and not self.dead:
            self.dead = True
            self._died_at = time.monotonic()
            self.log("died")
            for cb in self.on_death:
                cb()
        elif self.hp > 0:
            self.dead = False
            self._died_at = None

    def _h_44(self, r):                       # TOCLIENT_SHOW_FORMSPEC
        spec = r.long_string()
        name = r.text()
        self.current_formspec = name
        # The death screen is an ordinary formspec; closing it triggers respawn.
        if name == "__builtin:death" and self.auto_respawn:
            self.respawn()

    def _h_37(self, r):                       # TOCLIENT_DEATHSCREEN_LEGACY
        if self.auto_respawn:
            self.respawn()

    def send_formspec_fields(self, formname, fields):
        w = Writer().string(formname).u16(len(fields))
        for k, v in fields.items():
            w.string(k).long_string(v)
        self.send_command(C.TOSERVER_INVENTORY_FIELDS, w)

    def respawn(self):
        """Close the death formspec, which is what makes the server respawn us."""
        self.send_formspec_fields("__builtin:death",
                                  {"btn_respawn": "", "quit": "true"})

    def _h_4e(self, r):                       # TOCLIENT_BREATH
        self.breath = r.u16()

    def _h_29(self, r):                       # TOCLIENT_TIME_OF_DAY
        self.time_of_day = r.u16()

    def _h_45(self, r):                       # TOCLIENT_MOVEMENT
        keys = ("acceleration_default", "acceleration_air", "acceleration_fast",
                "speed_walk", "speed_crouch", "speed_fast", "speed_climb",
                "speed_jump", "liquid_fluidity", "liquid_fluidity_smooth",
                "liquid_sink", "gravity")
        self.movement = {k: r.f32() for k in keys}

    def _h_2a(self, r):                       # TOCLIENT_CSM_RESTRICTION_FLAGS
        # Limits the server places on *client-side mods*. A protocol client is
        # not bound by these -- it reads MapBlocks directly -- but they are
        # reported because they say what a CSM on this server would be allowed.
        self.csm_restriction_flags = r.u64()
        self.csm_restriction_noderange = r.u32()
        self.log(f"csm restrictions: flags=0x{self.csm_restriction_flags:x} "
                 f"noderange={self.csm_restriction_noderange}")

    def _h_41(self, r):                       # TOCLIENT_PRIVILEGES
        self.privileges = {r.text() for _ in range(r.u16())}

    def _h_27(self, r):                       # TOCLIENT_INVENTORY
        from .inventory import Inventory
        self.inventory_raw = r.rest().decode("utf-8", "replace")
        if self.inventory is None:
            self.inventory = Inventory()
        self.inventory.apply(self.inventory_raw)

    def _h_56(self, r):                       # TOCLIENT_UPDATE_PLAYER_LIST
        op = r.u8()
        names = {r.text() for _ in range(r.u16())}
        if op == 0:
            self.players = names
        elif op == 1:
            self.players |= names
        else:
            self.players -= names

    def _h_31(self, r):                       # TOCLIENT_ACTIVE_OBJECT_REMOVE_ADD
        removed, added = self.entities.handle_remove_add(r)
        for e in added:
            for cb in self.on_entity_add:
                cb(e)

    def _h_32(self, r):                       # TOCLIENT_ACTIVE_OBJECT_MESSAGES
        self.entities.handle_messages(r)

    def _h_2f(self, r):                       # TOCLIENT_CHAT_MESSAGE
        r.u8()                                # version
        msg_type = r.u8()
        sender = r.wide_string()
        text = r.wide_string()
        from .text import strip_escapes
        sender, text = strip_escapes(sender), strip_escapes(text)
        for cb in self.on_chat:
            cb(sender, text, msg_type)

    # -------------------------------------------------------------- actions

    def _write_player_pos(self, w):
        """The position block shared by PLAYERPOS and INTERACT."""
        w.v3s32([int(c * BS * POS_SCALE) for c in self.position])
        w.v3s32([int(c * BS * POS_SCALE) for c in self.velocity])
        w.s32(int(self.pitch * 100)).s32(int(self.yaw * 100))
        w.u32(0)                              # keysPressed: unused by the server
        # The server only sends map blocks that fall inside the client's view
        # cone (isBlockInSight), so a bot reports an all-round field of view to
        # receive the map in every direction. The field is fov_radians * 80,
        # and 255/80 rad is past a full turn, which makes the test always pass.
        w.u8(self.fov_byte)
        w.u8(self.wanted_range)               # view distance, in map blocks
        if self.proto_ver >= 46:
            w.u8(0)                           # camera inverted
            w.f32(0.0).f32(0.0)               # movement speed / direction
        return w

    def send_playerpos(self, force=False, min_interval=0.1):
        now = time.monotonic()
        if not force and now - self._last_pos_send < min_interval:
            return
        self._last_pos_send = now
        self.send_command(C.TOSERVER_PLAYERPOS, self._write_player_pos(Writer()),
                          channel=0, reliable=False)

    def send_chat(self, message: str):
        self.send_command(C.TOSERVER_CHAT_MESSAGE, Writer().wide_string(message))

    def interact(self, action, pointed_pos=None, above_pos=None, object_id=None):
        w = Writer().u8(action).u16(self.wield_index)
        pt = Writer().u8(0)                   # PointedThing version
        if object_id is not None:
            pt.u8(POINTEDTHING_OBJECT).u16(object_id)
        elif pointed_pos is not None:
            pt.u8(POINTEDTHING_NODE).v3s16(pointed_pos)
            pt.v3s16(above_pos if above_pos is not None else pointed_pos)
        else:
            pt.u8(POINTEDTHING_NOTHING)
        w.long_string(pt.bytes())
        self._write_player_pos(w)
        self.send_command(C.TOSERVER_INTERACT, w)

    # ------------------------------------------------------- inventory actions

    def send_inventory_action(self, action: str):
        """Send a raw InventoryAction.

        Luanti serialises these as plain text, e.g.
            Move <count> <inv> <list> <index> <inv> <list> <index>
            Drop <count> <inv> <list> <index>
            Craft <count> <inv>
        where <inv> is `current_player`, `player:<name>`, `nodemeta:x,y,z`
        or `detached:<name>`. The packet body is the bare string.
        """
        self.send_command(C.TOSERVER_INVENTORY_ACTION,
                          Writer().raw(action.encode("utf-8")))

    @staticmethod
    def nodemeta(pos):
        """Inventory location string for a node's own inventory (chest, table)."""
        return f"nodemeta:{int(pos[0])},{int(pos[1])},{int(pos[2])}"

    def inv_move(self, count, from_list, from_i, to_list, to_i,
                 from_inv="current_player", to_inv=None):
        self.send_inventory_action(
            f"Move {count} {from_inv} {from_list} {from_i} "
            f"{to_inv or from_inv} {to_list} {to_i}")

    def inv_move_somewhere(self, count, from_list, from_i, to_list,
                           from_inv="current_player", to_inv=None):
        """Move into the first slot of `to_list` that will accept the items."""
        self.send_inventory_action(
            f"MoveSomewhere {count} {from_inv} {from_list} {from_i} "
            f"{to_inv or from_inv} {to_list}")

    def inv_drop(self, count, list_name, index, inv="current_player"):
        self.send_inventory_action(f"Drop {count} {inv} {list_name} {index}")

    def inv_craft(self, count, inv="current_player"):
        self.send_inventory_action(f"Craft {count} {inv}")

    def select_slot(self, index: int):
        self.wield_index = index
        self.send_command(C.TOSERVER_PLAYERITEM, Writer().u16(index))

    def disconnect(self):
        if self.conn:
            self.conn.close()
