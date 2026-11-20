"""Train argmedian h=4 L=3 seeds in batches that can be spread over several days.

Built for a fanless MacBook Air that is only left training while nobody uses it:
  - one seed at a time, one thread, low priority (nice 10);
  - a batch stops after --count seeds or before a seed would overrun --hours;
  - resumes where it left off: finished seeds are skipped, a seed cut half-way
    is deleted and trained again from scratch (AlgZoo cannot resume a run);
  - `touch data/trained/STOP` stops it after the seed in progress;
  - keeps the Mac awake while it runs (caffeinate -i; the screen may turn off,
    but closing the lid still sleeps the Mac);
  - heat: a pause between seeds; if a seed takes more than --slow seconds
    (the Mac slows itself down when hot; a normal seed takes ~310 s), a long
    pause; if macOS records a thermal warning, it stops.

Every seed is logged to data/trained/batch_log.tsv.

Usage: PYTHONPATH=src python scripts/train_batch_h4.py --first 25 --last 124 (--count 50 | --hours 1.5)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINED = REPO / "data" / "trained"
STOP = TRAINED / "STOP"
LOG = TRAINED / "batch_log.tsv"

parser = argparse.ArgumentParser()
parser.add_argument("--first", type=int, default=25)
parser.add_argument("--last", type=int, default=124)
parser.add_argument("--count", type=int, help="train at most this many seeds")
parser.add_argument("--hours", type=float, help="do not start a seed that would end after this many hours")
parser.add_argument("--rest", type=int, default=60, help="pause between seeds (s)")
parser.add_argument("--slow", type=int, default=400, help="a seed slower than this (s) triggers --cooldown")
parser.add_argument("--cooldown", type=int, default=600, help="pause after a slow seed (s)")
args = parser.parse_args()
if args.count is None and args.hours is None:
    sys.exit("give --count or --hours")


def final(seed):
    return TRAINED / f"argmedian_4_3_seed{seed}" / f"argmedian_4_3_{2**30}_{seed}.pth"


def thermal_warning():
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    warn = [ln.strip() for ln in out.splitlines() if "warning level" in ln.lower() and not ln.strip().startswith("Note: No ")]
    for ln in out.splitlines():  # Intel-style report: CPU_Speed_Limit = 100 means no throttling
        if "CPU_Speed_Limit" in ln and ln.split("=")[-1].strip().isdigit() and int(ln.split("=")[-1]) < 100:
            warn.append(ln.strip())
    return warn


def log(*fields):
    new = not LOG.exists()
    with LOG.open("a") as f:
        if new:
            f.write("time\tseed\tseconds\tevent\n")
        f.write("\t".join(str(x) for x in fields) + "\n")
    print(*fields, flush=True)


TRAINED.mkdir(parents=True, exist_ok=True)
(TRAINED / "seed_logs").mkdir(exist_ok=True)
STOP.unlink(missing_ok=True)
subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
todo = [s for s in range(args.first, args.last + 1) if not final(s).exists()]
print(f"{len(todo)} seeds left in {args.first}-{args.last}", flush=True)
start, done, expected = time.time(), 0, 330.0

for seed in todo:
    if args.count is not None and done >= args.count:
        break
    if args.hours is not None and time.time() - start + expected > args.hours * 3600:
        log(datetime.now().isoformat(timespec="seconds"), seed, "", "time budget reached")
        break
    if STOP.exists():
        STOP.unlink()
        log(datetime.now().isoformat(timespec="seconds"), seed, "", "stopped by STOP file")
        break
    warn = thermal_warning()
    if warn:
        log(datetime.now().isoformat(timespec="seconds"), seed, "", "thermal warning, stopping: " + " | ".join(warn))
        break
    d = final(seed).parent
    if d.exists():  # cut half-way last time
        shutil.rmtree(d)
    t = time.time()
    with open(TRAINED / "seed_logs" / f"seed{seed}.log", "w") as out:
        p = subprocess.Popen(["nice", "-n", "10", sys.executable, str(REPO / "scripts" / "train_seed_h4.py"),
                              "--seed", str(seed), "--train-only"], stdout=out, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONPATH": str(REPO / "src")})
        try:
            code = p.wait()
        except KeyboardInterrupt:
            p.terminate()
            p.wait()
            shutil.rmtree(d, ignore_errors=True)
            log(datetime.now().isoformat(timespec="seconds"), seed, "", "interrupted, partial seed deleted")
            sys.exit(1)
    secs = time.time() - t
    if code != 0 or not final(seed).exists():
        log(datetime.now().isoformat(timespec="seconds"), seed, f"{secs:.0f}", f"FAILED (exit {code}), see seed_logs/seed{seed}.log")
        break
    done += 1
    expected = max(expected, secs + args.rest)
    log(datetime.now().isoformat(timespec="seconds"), seed, f"{secs:.0f}", "done")
    if secs > args.slow:
        log(datetime.now().isoformat(timespec="seconds"), seed, "", f"slow seed (Mac hot?), cooling {args.cooldown} s")
        time.sleep(args.cooldown)
    else:
        time.sleep(args.rest)

left = sum(not final(s).exists() for s in range(args.first, args.last + 1))
print(f"batch over: {done} seeds trained in {(time.time() - start) / 3600:.2f} h; {left} left in {args.first}-{args.last}")
