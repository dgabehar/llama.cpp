#!/usr/bin/env python3
# Tool-call corruption stress test for speculative decoding under a grammar (--spec-grammar).
#
# Sends N tool-call requests to a running llama-server and checks every call:
#   - the reply parses into tool_calls (finish_reason "tool_calls")
#   - each call's arguments are valid JSON with the required keys
#   - no string argument carries a template tag (<parameter=, </parameter>, </function>, <tool_call>, ...)
# Run it once against a server started without --spec-grammar and once with it, same model and flags
# otherwise, and compare the two summaries. The 2026-09-15 failure showed up as template tags in
# string arguments.
#
#   python3 spec_grammar_stress.py --url http://127.0.0.1:8080 -n 200 --temp 0.6 --out on.jsonl

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

TAG = re.compile(r"</?(tool_call|function|parameter)\b|<parameter=|<function=")

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write a file to disk.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "cwd": {"type": "string"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace text in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["path", "old", "new"]}}},
]
REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}

# string arguments with markup, code and repetition: where the tag ambiguity bites
PROMPTS = [
    "Write an HTML page to /tmp/a.html with a table of 8 rows of fruit names and prices.",
    "Create /tmp/parse.py: a Python function that parses XML like <item><name>x</name></item> into dicts, with tests.",
    "Write a bash script /tmp/loop.sh that prints numbers 1 to 20, each on its own line, with comments.",
    "In /tmp/cfg.yaml replace 'port: 8080' with 'port: 9090'.",
    "Run `grep -rn TODO src/` in /home/user/proj.",
    "Write /tmp/notes.md: a markdown checklist of 12 steps to set up a Kubernetes cluster.",
    "Create /tmp/schema.json with a JSON schema for a user object with 10 fields.",
    "Write /tmp/tpl.jinja: a Jinja template that renders a list of tool calls as <tool_call> blocks.",
]
# prompts whose arguments legitimately contain template tags: kept for the parse check, skipped by the tag check
TAG_OK = {len(PROMPTS) - 1}


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check(resp, tags_ok=False):
    choice = resp["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    content = choice["message"].get("content") or ""
    if choice.get("finish_reason") == "length":
        return "truncated", ""  # not a corruption verdict; raise --max-tokens
    if not calls:
        return "no_call", ("tag in content" if not tags_ok and TAG.search(content) else "")
    for c in calls:
        name = c["function"]["name"]
        try:
            args = json.loads(c["function"]["arguments"])
        except Exception as e:
            return "bad_json", str(e)
        if name not in REQUIRED or not isinstance(args, dict):
            return "bad_name", name
        missing = [k for k in REQUIRED[name] if k not in args]
        if missing:
            return "missing_arg", ",".join(missing)
        for k, v in args.items():
            if not tags_ok and isinstance(v, str) and TAG.search(v):
                return "tag_in_arg", f"{name}.{k}: {TAG.search(v).group(0)}"
    return "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.6)
    # auto gives the lazy grammar (trigger buffer + reasoning-budget sampler), the 2026-09-15 setup
    ap.add_argument("--tool-choice", choices=["auto", "required"], default="auto")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out = open(a.out, "w") if a.out else None
    counts, n_draft, n_acc, n_pred, t_pred = {}, 0, 0, 0, 0.0
    t0 = time.time()
    for i in range(a.n):
        body = {
            "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            "tools": TOOLS, "tool_choice": a.tool_choice,
            "temperature": a.temp, "seed": i, "max_tokens": a.max_tokens,
        }
        try:
            resp = post(a.url, body, a.timeout)
            verdict, detail = check(resp, i % len(PROMPTS) in TAG_OK)
            tm = resp.get("timings", {})
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            # the tool-call parser refusing the output is the same failure class as a tag in an argument
            verdict = "parse_error" if "does not match" in msg else "http_error"
            resp, detail, tm = None, msg[:200], {}
        except Exception as e:
            resp, verdict, detail, tm = None, "http_error", str(e), {}
        counts[verdict] = counts.get(verdict, 0) + 1
        n_draft += tm.get("draft_n", 0)
        n_acc += tm.get("draft_n_accepted", 0)
        n_pred += tm.get("predicted_n", 0)
        t_pred += tm.get("predicted_ms", 0.0) / 1000
        if out:
            out.write(json.dumps({"i": i, "verdict": verdict, "detail": detail, "resp": resp}) + "\n")
        if verdict != "ok":
            print(f"[{i}] {verdict} {detail}", file=sys.stderr)

    summary = {
        "n": a.n, "temp": a.temp, "tool_choice": a.tool_choice, "counts": counts,
        "bad": a.n - counts.get("ok", 0) - counts.get("truncated", 0),
        "draft_n": n_draft, "draft_accepted": n_acc,
        "accept_rate": round(n_acc / n_draft, 3) if n_draft else None,
        "tok_per_s": round(n_pred / t_pred, 2) if t_pred else None,
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
