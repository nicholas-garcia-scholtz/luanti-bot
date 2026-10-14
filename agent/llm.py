"""Minimal OpenAI-compatible chat client for the class MiniMax gateway.

Standard library only, like the rest of the repo. Configuration comes from
`.env` in the repo root (or the real environment, which wins):

    LLM_BASE_URL=https://agent.elliottwen.info/v1
    LLM_API_KEY=sk-...
    LLM_MODEL=MiniMax-M3

Run `python -m agent.llm` to check the connection and tool calling.
"""
import json
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=os.path.join(ROOT, ".env")):
    """Read KEY=value lines into os.environ without overriding real env vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLM:
    def __init__(self, base_url=None, api_key=None, model=None, timeout=180):
        load_env()
        self.base_url = (base_url or os.environ["LLM_BASE_URL"]).rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = api_key or os.environ["LLM_API_KEY"]
        self.model = model or os.environ.get("LLM_MODEL", "MiniMax-M3")
        self.timeout = timeout
        self.prompt_tokens = 0            # running totals, for the logbook
        self.completion_tokens = 0
        self.calls = 0

    def chat(self, messages, tools=None, retries=3, **extra):
        """POST /chat/completions and return the assistant message dict.

        The returned dict can be appended to `messages` as-is. Tool calls, if
        any, are in message["tool_calls"].
        """
        body = {"model": self.model, "messages": messages, **extra}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json",
                     # Cloudflare in front of the gateway rejects the default
                     # Python-urllib User-Agent with "error code: 1010".
                     "User-Agent": "luantibot-agent/0.1"},
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                # 429 and 5xx are worth retrying; anything else is our fault
                if e.code != 429 and e.code < 500 or attempt == retries - 1:
                    raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"LLM unreachable: {e}") from None
            time.sleep(2 ** attempt)

        usage = data.get("usage") or {}
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.calls += 1
        return data["choices"][0]["message"]


if __name__ == "__main__":
    llm = LLM()
    print("model:", llm.model, "at", llm.base_url)

    reply = llm.chat([{"role": "user", "content": "Reply with exactly: pong"}])
    print("plain reply:", repr(reply.get("content")))

    tools = [{"type": "function", "function": {
        "name": "dig",
        "description": "Dig the node at the given coordinates.",
        "parameters": {"type": "object",
                       "properties": {"x": {"type": "integer"},
                                      "y": {"type": "integer"},
                                      "z": {"type": "integer"}},
                       "required": ["x", "y", "z"]}}}]
    reply = llm.chat([{"role": "user",
                       "content": "Dig the block at 3, -12, 7."}], tools=tools)
    calls = reply.get("tool_calls") or []
    for c in calls:
        print("tool call:", c["function"]["name"], c["function"]["arguments"])
    if not calls:
        print("no tool call; model said:", repr(reply.get("content")))

    print(f"tokens: {llm.prompt_tokens} in / {llm.completion_tokens} out "
          f"over {llm.calls} calls")
