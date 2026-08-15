#!/usr/bin/env python3
import pathlib, sys, re, json, os
from typing import List, Tuple

RUFF_PATH = os.getenv("RUFF_ERRORS", "ruff-errors.txt")
PYTEST_PATH = os.getenv("PYTEST_RESULTS", "pytest-results.xml")

def parse_ruff(path: pathlib.Path) -> List[Tuple[int, str, str]]:
    fixes = []
    txt = path.read_text()
    for line in txt.splitlines():
        m = re.match(r".+:\d+:\d+:\s*(.+)", line)
        if not m:
            continue
        msg = m.group(1).strip()
        if "undefined name" in msg.lower() or "F821" in msg:
            fixes.append((line, "REMOVE_F821_PLACEHOLDER", "ADD_MISSING_IMPORT_PLACEHOLDER"))
        if "F402" in msg or "loop‑var‑shadowing" in msg.lower():
            fixes.append((line, "REMOVE_F402_PLACEHOLDER", "ADD_CORRECT_LOOP_VAR_PLACEHOLDER"))
        if "F601" in msg:
            fixes.append((line, "REMOVE_F601_PLACEHOLDER", "FIX_DICT_KEY_PLACEHOLDER"))
    return fixes

def parse_pytest(path: pathlib.Path) -> List[Tuple[int, str, str]]:
    fixes = []
    txt = path.read_text()
    for m in re.finditer(r'(\S+:\d+):\s*AssertionError', txt):
        fname, line = m.group(1).split(":")
        fixes.append((int(line), f"FIX_{fname}_LINE_{line}", f"FIXED_{fname}_LINE_{line}"))
    return fixes

def main():
    ruff_path = pathlib.Path(RUFF_PATH)
    pytest_path = pathlib.Path(PYTEST_PATH)

    if not ruff_path.exists():
        print(f"⚠️  Ruff errors file not found: {RUFF_PATH}")
    else:
        fixes = parse_ruff(ruff_path)
        for line_no, old, new in fixes:
            print(f"🔧 Fix ruff line {line_no}: {old!r} → {new!r}")

    if not pytest_path.exists():
        print(f"⚠️  Pytest results file not found: {PYTEST_PATH}")
    else:
        fixes += parse_pytest(pytest_path)
        for line_no, old, new in fixes:
            print(f"🔧 Fix pytest line {line_no}: {old!r} → {new!r}")

    print("\n✅  Fix‑script finished. Review the printed replacements above.")
    print("If you want the bot to actually edit files, edit the `auto-fix-and-pr` job")
    print("to call `subprocess.run([...])` with a custom Python script that does the edits.")

if __name__ == "__main__":
    main()