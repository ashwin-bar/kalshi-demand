"""Pre-publication check (row 149): scans every file tracked by git for local paths, credential-like text,
data files and large files. Run from the project folder:   .venv\\Scripts\\python src\\check_publish.py"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()

CHECKS = {
    "local path": re.compile(r"[A-Za-z]:\\(Users|kalshi)", re.IGNORECASE),
    "credential-like": re.compile(r"(api[_-]?key|secret|password|private[ _]key|bearer|KALSHI-ACCESS)", re.IGNORECASE),
}
DATA_EXT = {".parquet", ".csv", ".db", ".sqlite", ".pem", ".key", ".log"}
BIG_MB = 5

problems = 0
print(f"Scanning {len(files)} tracked files\n")
for name in files:
    p = ROOT / name
    if not p.is_file():
        continue
    size_mb = p.stat().st_size / 1e6
    if p.suffix.lower() in DATA_EXT:
        print(f"[data file]  {name} ({size_mb:.1f} MB)")
        problems += 1
    if size_mb > BIG_MB:
        print(f"[large file] {name} ({size_mb:.1f} MB)")
        problems += 1
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        continue
    for label, rx in CHECKS.items():
        hits = [i for i, line in enumerate(lines, 1) if rx.search(line)]
        if hits:
            print(f"[{label}] {name}: {len(hits)} line(s), e.g. line {hits[0]}: {lines[hits[0] - 1].strip()[:100]}")
            problems += 1
print(f"\n{problems} item(s) to review" if problems else "\nClean: nothing to review")