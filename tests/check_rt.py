"""Sprawdza czasy reakcji w najnowszej sesji testowej.

Nie wymaga PsychoPy ani pandas; uruchamiany z PyCharm: Run 'check_rt'.
"""
from pathlib import Path
import csv
import math

ROOT = Path(__file__).resolve().parent
folder = ROOT / "data"
paths = list(folder.glob("*/trials.csv"))
if not paths:
    raise SystemExit("Brak danych: najpierw uruchom demo_session.py.")
latest = max(paths, key=lambda p: p.stat().st_mtime)
print("Sprawdzam:", latest)
with latest.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

bad = []
measured = 0
for r in rows:
    for key in ("rt_cue_seconds", "rt_reward_seconds"):
        val = r.get(key, "")
        if val is None or val.strip() == "":
            if r.get("trial_status") == "completed":
                bad.append((r["phase"], r["trial_index"], key, "BRAK"))
            continue
        measured += 1
        try:
            number = float(val)
        except ValueError:
            number = float("nan")
        if not math.isfinite(number) or number < 0:
            bad.append((r["phase"], r["trial_index"], key, val))

print("Liczba zapisanych prob:", len(rows))
print("Liczba RT sprawdzonych:", measured)
if bad:
    print("UWAGA - bledne czasy reakcji:")
    for phase, trial, field, val in bad:
        print(f"  {phase}, proba {trial}: {field}={val}")
    raise SystemExit("WYNIK: BLAD RT - sprawdz powyzsze proby.")
print("OK - wszystkie zapisane czasy reakcji sa nieujemne i skonczone.")
