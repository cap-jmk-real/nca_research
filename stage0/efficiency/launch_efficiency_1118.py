#!/usr/bin/env python3
"""Engine launcher for run_efficiency_1118.py (all phases, resumable).

1. train, 1 thread: one subprocess per (benchmark, model) running all 5 seeds, at most NPAR at a time,
   each pinned (taskset) to its own physical core from CORES; nice 10, ionice idle.
2. train, 8 threads: one subprocess per benchmark, run alone, pinned to the 8 physical cores of CORES8.
3. latency, 1 thread: one subprocess per benchmark, run alone, pinned to LAT_CORE.
Logs: EFF_OUT/logs/. Marker: EFF_OUT/DONE (or FAILED with the failing step).
"""
import os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ["EFF_OUT"])
PY = os.environ.get("PY", sys.executable)
NPAR = int(os.environ.get("NPAR", 5))
CORES = [int(c) for c in os.environ.get("CORES", "9,10,11,12,13").split(",")]
CORES8 = os.environ.get("CORES8", "8-15")
LAT_CORE = os.environ.get("LAT_CORE", "14")
SCRIPT = str(HERE / "run_efficiency_1118.py")
BENCHES = ["curated", "bfcl_e1v2", "toolbench", "apibank"]
HEAVY = ["hgb_cell", "hgb", "rf_cell", "rf", "transformer", "knn", "gnca_chain", "gnca_star", "gnca_learned", "gnca_tree",
         "gnn_untied", "mlp", "logreg"]
LOGS = OUT / "logs"; LOGS.mkdir(parents=True, exist_ok=True)


def cmd(core, *args):
    return ["nice", "-n", "10", "ionice", "-c3", "taskset", "-c", str(core), PY, SCRIPT, *args]


def log(m):
    with open(LOGS / "launcher.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}\n")


def fail(m):
    (OUT / "FAILED").write_text(m); log("FAILED " + m); sys.exit(1)


def phase_t1():
    jobs = [(b, m) for b in BENCHES for m in HEAVY]
    free, running = list(CORES), {}
    while jobs or running:
        while jobs and free and len(running) < NPAR:
            b, m = jobs.pop(0); c = free.pop(0)
            fh = open(LOGS / f"t1_{b}_{m}.log", "a")
            p = subprocess.Popen(cmd(c, "--phase", "train", "--threads", "1", "--bench", b, "--models", m), stdout=fh, stderr=subprocess.STDOUT)
            running[p] = (b, m, c, fh); log(f"start t1 {b}/{m} core {c} pid {p.pid}")
        time.sleep(2)
        for p in list(running):
            if p.poll() is not None:
                b, m, c, fh = running.pop(p); fh.close(); free.append(c)
                log(f"end t1 {b}/{m} rc={p.returncode}")
                if p.returncode != 0:
                    fail(f"t1 {b}/{m} rc={p.returncode}")


def phase_alone(tag, core, extra):
    for b in BENCHES:
        with open(LOGS / f"{tag}_{b}.log", "a") as fh:
            log(f"start {tag} {b}")
            rc = subprocess.call(cmd(core, *extra, "--bench", b), stdout=fh, stderr=subprocess.STDOUT)
            log(f"end {tag} {b} rc={rc}")
            if rc != 0:
                fail(f"{tag} {b} rc={rc}")


if __name__ == "__main__":
    (OUT / "FAILED").unlink(missing_ok=True)
    log(f"== start commit={os.environ.get('EFF_COMMIT', 'n/a')}")
    phase_t1()
    phase_alone("t8", CORES8, ["--phase", "train", "--threads", "8"])
    phase_alone("lat", LAT_CORE, ["--phase", "latency", "--threads", "1"])
    (OUT / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S")); log("== done")
