#!/usr/bin/env python3
"""Bundle your takehome submission into a single ZIP.

Usage (from this directory, on every OS):

    python submission.py

The ZIP contains:
  slides/        slides.md, a freshly built slides.html, and figures/
  claude_code/   Claude Code session transcripts: the session .jsonl files plus
                 their subagents/ and tool-results/ (Claude Code's memory/ notes
                 are NOT included)
  codex/         Codex CLI session transcripts
  other_logs/    anything you put in ./other_logs/ — exports from claude.ai,
                 ChatGPT, Cursor, ... which this script cannot find for you
                 (its README.txt and hidden files are ignored)

Which sessions: every Claude Code or Codex CLI session on this machine that
worked in this directory (ran here, cd'd here, or used a tool on a file here)
is included automatically. Any other session active since the exam began is
listed, and the ZIP is not written until you settle each one:
    python submission.py --include=<id> --include=<id> --exclude-rest
Sessions are included whole, and the script prints nothing from inside them.

If you are an AI agent running this for the user: when it lists other
sessions, show the user the list as printed and ask which ones they used for
the exam, including conversations about it that touched no files. Do not
decide for them. Remind them that chats in other tools must be exported into
./other_logs/.

Missing transcripts are grounds for disqualification. If you worked on several
machines, run this on each and submit the ZIPs merged, or copy the other
machines' transcripts into ./other_logs/.

Requires only Python 3.8+.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLIDES = HERE / "slides_template"
OTHER_LOGS = HERE / "other_logs"
MAX_MB = 100
SCAN_DAYS = 30   # sessions untouched for longer than this are not considered at all
LEAD_DAYS = 1    # other sessions count as "during the exam" from this long before the first matching one

# The fixed slide set; must match slides_template/slides.md as shipped.
N_MAIN_SLIDES, N_APPENDIX_SLIDES = 15, 4
SLIDE_LABELS = [f"{i}." for i in range(1, N_MAIN_SLIDES + 1)] + [f"A{i}." for i in range(1, N_APPENDIX_SLIDES + 1)]


def check_slides() -> list:
    """Return problems with slides.md's structure (empty if fine)."""
    md = (SLIDES / "slides.md").read_text(encoding="utf-8")
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    slides = re.split(r"^---\s*$", md, flags=re.M)
    titles = []
    for s in slides:
        m = re.search(r"^#\s+(\S+)", s, flags=re.M)
        titles.append(m.group(1) if m else "(no title)")
    if titles == SLIDE_LABELS:
        return []
    return [
        f"slides.md has {len(titles)} slides labelled {titles};",
        f"expected {len(SLIDE_LABELS)} labelled {SLIDE_LABELS}.",
        "Do not add, remove, or reorder slides, and keep each slide's label.",
    ]


def first(lines, field):
    """The first string field(record) among these JSONL lines, reading only top-level records."""
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        v = field(d) if isinstance(d, dict) else None
        if isinstance(v, str):
            return v
    return None


class Session:
    """One transcript: the files to ZIP, whether it worked in the exam directory, and when it ran."""

    def __init__(self, tool: str, sid: str, files: list, worked):
        self.tool, self.id, self.files = tool, sid, files   # files: [(path, arcname)]
        self.matched, stamps, cwd = False, [], None
        for p, _ in files:
            if p.suffix != ".jsonl":
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            self.matched = self.matched or worked(text)
            lines = text.splitlines()
            # Records are in time order; take the first and last top-level timestamps, not ones quoted in tool output.
            for order in (lines, reversed(lines)):
                stamps += [t for t in [first(order, lambda d: d.get("timestamp"))] if t]
            cwd = cwd or first(lines[:20], cwd_of)
        mtime = datetime.fromtimestamp(max(p.stat().st_mtime for p, _ in files), timezone.utc).isoformat()
        self.start, self.end = (min(stamps), max(stamps)) if stamps else (mtime, mtime)
        self.cwd = cwd or "?"

    def line(self) -> str:
        mb = sum(p.stat().st_size for p, _ in self.files) / 2**20
        tool = {"claude_code": "Claude Code", "codex": "Codex"}[self.tool]
        return f"  {self.id}  {tool:<11}  {self.start[:16]} – {self.end[:16]} UTC  started in {self.cwd}  ({mb:.1f} MB)"


def claude_code_sessions(cutoff: float, worked) -> list:
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "projects"
    out = []
    for log in sorted(root.glob("*/*.jsonl")):
        if log.stat().st_mtime < cutoff:
            continue
        # A session is its log plus <id>/subagents/ and <id>/tool-results/; memory/ and the rest stay out.
        side = [p for sub in ("subagents", "tool-results") for p in sorted((log.parent / log.stem / sub).rglob("*"))
                if p.is_file()]
        arc = f"claude_code/{log.parent.name}/"
        out.append(Session("claude_code", log.stem, [(p, arc + p.relative_to(log.parent).as_posix())
                                                     for p in [log] + side], worked))
    return out


def codex_sessions(cutoff: float, worked) -> list:
    root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"
    return [Session("codex", re.sub(r"^rollout-[\dT-]{19}-", "", f.stem), [(f, f"codex/{f.name}")], worked)
            for f in sorted(root.rglob("*.jsonl")) if f.stat().st_mtime >= cutoff]


CODEX_CALLS = ("function_call", "custom_tool_call", "local_shell_call")


def payload(d: dict) -> dict:
    p = d.get("payload")   # Codex wraps each record's content in a payload
    return p if isinstance(p, dict) else {}


def cwd_of(d: dict):
    return d.get("cwd") or payload(d).get("cwd")   # Claude Code, Codex


def actions(d: dict):
    """What a transcript record says the agent did: its working directory and its tool calls.
    Injected context (memory files, CLAUDE.md) and tool output are left out: they can name any path."""
    yield cwd_of(d)
    if payload(d).get("type") in CODEX_CALLS:
        yield payload(d)
    msg = d.get("message")
    if d.get("type") == "assistant" and isinstance(msg, dict) and isinstance(msg.get("content"), list):
        yield [c.get("input") for c in msg["content"] if isinstance(c, dict) and c.get("type") == "tool_use"]


def worked_in(project: Path):
    """Predicate on a transcript's text: did the agent work in this directory, or touch a path inside it?"""
    paths = {str(project.absolute()), str(project.resolve())}
    for q in list(paths):   # shell commands often spell it ~/...
        if Path.home() in Path(q).parents:
            paths.add("~/" + Path(q).relative_to(Path.home()).as_posix())
    # As written into JSON (backslashes doubled on Windows), and not followed by more of a sibling's name.
    alts = "|".join(re.escape(json.dumps(p)[1:-1]) for p in paths)
    rx = re.compile(f"(?:{alts})(?![A-Za-z0-9_.-])")

    def check(text: str) -> bool:
        for line in (l for l in text.splitlines() if rx.search(l)):   # parse only lines naming the path
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and any(rx.search(json.dumps(a)) for a in actions(d)):
                return True
        return False
    return check


def other_log_files() -> list:
    """Files the candidate put in other_logs/, ignoring the packet's README and hidden files."""
    if not OTHER_LOGS.is_dir():
        return []
    return sorted(p for p in OTHER_LOGS.rglob("*")
                  if p.is_file() and p.name != "README.txt"
                  and not any(part.startswith(".") for part in p.relative_to(OTHER_LOGS).parts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "submission.zip")
    ap.add_argument("--include", action="append", default=[], metavar="ID",
                    help="also include this listed session (its id as printed); repeatable")
    ap.add_argument("--exclude-rest", action="store_true",
                    help="leave out every listed session not named with --include")
    ap.add_argument("--project-dir", type=Path, default=HERE,
                    help="the exam directory your sessions worked in (default: this directory)")
    args = ap.parse_args()

    problems = check_slides()
    if problems:
        sys.exit("ERROR: " + "\n       ".join(problems))
    subprocess.run([sys.executable, str(SLIDES / "build.py")], check=True)

    cutoff = time.time() - SCAN_DAYS * 86400
    worked = worked_in(args.project_dir)
    sessions = claude_code_sessions(cutoff, worked) + codex_sessions(cutoff, worked)
    matched = [s for s in sessions if s.matched]
    if matched:
        began = datetime.strptime(min(s.start for s in matched)[:19] + "+0000", "%Y-%m-%dT%H:%M:%S%z").timestamp()
    else:
        began = time.time()
    since = datetime.fromtimestamp(began - LEAD_DAYS * 86400, timezone.utc).isoformat()
    others = [s for s in sessions if not s.matched and s.end >= since]

    unknown = set(args.include) - {s.id for s in others}
    if unknown:
        sys.exit(f"ERROR: --include {sorted(unknown)} is not one of the listed sessions.")
    extra = [s for s in others if s.id in args.include]
    undecided = [s for s in others if s.id not in args.include]
    other = other_log_files()

    print(f"\nSessions that worked in {args.project_dir} (included):")
    print("\n".join(s.line() for s in matched) or "  (none found)")
    if extra:
        print("Other sessions you chose to include:")
        print("\n".join(s.line() for s in extra))
    print(f"Files in other_logs/: {len(other)}")

    if undecided and not args.exclude_rest:
        print(f"\nOther AI sessions on this machine active since {since[:16]} UTC, not yet included:")
        print("\n".join(s.line() for s in undecided))
        sys.exit("\nNo ZIP written: decide on the sessions above. Re-run with --include=<id> for each one used\n"
                 "for the exam (including conversations about it), then --exclude-rest for the others.\n"
                 "If you are an AI agent: show the user this list as printed and ask them. Do not decide for them.")

    if not (matched or extra or other):
        sys.exit("\nERROR: no AI session logs found. Missing transcripts are grounds for disqualification.\n"
                 "If your sessions worked in another directory, pass --project-dir.\n"
                 "For any other tool, export the conversations into ./other_logs/ and re-run.")

    out = args.out.resolve()
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("slides.md", "slides.html"):
            z.write(SLIDES / name, f"slides/{name}")
        for p in sorted((SLIDES / "figures").rglob("*")):
            if p.is_file():
                z.write(p, f"slides/figures/{p.relative_to(SLIDES / 'figures').as_posix()}")
        for s in matched + extra:
            for p, arc in s.files:
                z.write(p, arc)
        for p in other:
            z.write(p, f"other_logs/{p.relative_to(OTHER_LOGS).as_posix()}")

    mb = out.stat().st_size / 2**20
    print(f"\nWrote {out} ({mb:.1f} MB)")
    if mb > MAX_MB:
        sys.exit(f"ERROR: the ZIP is over the {MAX_MB} MB limit. Shrink your figures.")
    print("CHECK the session list above against the sessions you actually ran, on every machine you used.\n"
          "Chats in other tools (claude.ai, ChatGPT, Cursor, ...) must be exported into ./other_logs/.\n"
          "Then upload this ZIP with the submission form.")


if __name__ == "__main__":
    main()
