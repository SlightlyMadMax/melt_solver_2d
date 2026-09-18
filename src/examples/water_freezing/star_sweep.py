#!/usr/bin/env python
"""
Star-design parameter study: vary one parameter at a time around a baseline.

Each axis holds the other three parameters at the baseline, which is what a
convergence study needs — a full cartesian product would be both wasteful and
harder to read.

Runs are independent, so they are executed in a process pool. Each case writes its
own summary.json and the results are merged at the end, which avoids the race that
concurrent appends to one CSV would cause.

Baseline (cold start, verified stable with a 5x margin at 151 and 2.5x at 201):
    grid 151x151, dt = 0.02 s, eps = eps_flow = 0.1 K, C = 1e6 1/s

Example
-------
    python -m src.examples.water_freezing.star_sweep --jobs 4
    python -m src.examples.water_freezing.star_sweep --jobs 4 --dry-run
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT_ROOT = HERE / "data" / "cold_start" / "parametric"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-at-a-time parameter study around a baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    base = p.add_argument_group("baseline")
    base.add_argument("--grid", type=int, default=151)
    base.add_argument("--dt", type=float, default=0.02)
    base.add_argument("--eps", type=float, default=0.1)
    base.add_argument("--penalty-c", type=float, default=1e6)
    base.add_argument("--start", choices=["cold", "warm"], default="cold")
    base.add_argument("--cold-wall-ramp", type=float, default=0.0)
    base.add_argument("--end-time", type=float, default=2340.0)

    ax = p.add_argument_group("axes")
    ax.add_argument("--grids", type=int, nargs="*", default=[51, 101, 151, 201])
    ax.add_argument("--dts", type=float, nargs="*", default=[0.08, 0.04, 0.02, 0.01])
    ax.add_argument("--eps-values", type=float, nargs="*",
                    default=[0.05, 0.1, 0.2, 0.4])
    ax.add_argument("--penalty-values", type=float, nargs="*",
                    default=[1e4, 1e5, 1e6])
    ax.add_argument(
        "--split-eps", action=argparse.BooleanOptionalAction, default=True,
        help="add two off-diagonal points that separate the heat-equation "
        "smoothing from the penalty smoothing",
    )

    run = p.add_argument_group("execution")
    run.add_argument("--jobs", type=int, default=4, help="runs in parallel")
    run.add_argument("--amg-rebuild-every", type=int, default=50)
    run.add_argument("--vorticity-bc-order", type=int, choices=(1, 2), default=2)
    run.add_argument("--timeout", type=float, default=None, help="per run [s]")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--log-interval", type=float, default=60.0,
        help="how often each run logs its step, in seconds of model time",
    )
    run.add_argument(
        "--progress-interval", type=float, default=120.0,
        help="how often the sweep prints where its running cases have got to, "
        "in seconds of wall clock; 0 turns the heartbeat off",
    )
    run.add_argument(
        "--smoke-time", type=float, default=0.0,
        help="before the real sweep, run every case for this many seconds of "
        "model time in a throwaway directory and abort unless all of them "
        "survive. A warm start that is going to blow up does so in its first "
        "steps, so a couple of minutes here buys back hours",
    )
    run.add_argument(
        "--smoke-only", action="store_true",
        help="stop after the smoke test instead of starting the real sweep",
    )
    run.add_argument(
        "--amg-rebuild-warmup", type=int, default=0,
        help="rebuild the AMG hierarchy on every one of the first N steps of each run",
    )
    run.add_argument(
        "--heat-convection", choices=("deferred", "central-div"), default="deferred",
        help="convective term of the heat equation, passed to run.py",
    )
    run.add_argument(
        "--no-latent-convection", action="store_true",
        help="multiply the convective term by c instead of c + lambda*delta",
    )
    run.add_argument(
        "--penalty-time-scheme", choices=("cn", "dr", "implicit"), default="cn",
        help="time discretisation of the penalty term, passed to run.py",
    )
    run.add_argument(
        "--warm-start-file", type=Path, default=None,
        help="precursor field for --start warm; run.py resolves it by grid otherwise",
    )
    run.add_argument(
        "--energy-budget", action="store_true",
        help="have every run keep its domain energy budget, passed to run.py",
    )
    run.add_argument(
        "--save-interval", type=float, default=0.0,
        help="checkpoint the fields every this many seconds of model time, "
        "passed to run.py; 0 keeps only the final state",
    )
    run.add_argument(
        "--out-root", type=Path, default=OUT_ROOT,
        help="directory for the run subdirectories and the merged summary",
    )
    return p.parse_args(argv)


def build_cases(a: argparse.Namespace) -> list[dict]:
    """Baseline plus one deviation per axis; duplicates collapse to a single run."""
    cases, seen = [], set()

    def add(axis, grid, dt, eps, eps_flow, c):
        key = (grid, round(dt, 12), round(eps, 12), round(eps_flow, 12), round(c, 6))
        if key in seen:
            return
        seen.add(key)
        cases.append({
            "axis": axis, "grid": grid, "dt": dt,
            "eps": eps, "eps_flow": eps_flow, "c": c,
        })

    add("baseline", a.grid, a.dt, a.eps, a.eps, a.penalty_c)
    for g in a.grids:
        add("grid", g, a.dt, a.eps, a.eps, a.penalty_c)
    for dt in a.dts:
        add("dt", a.grid, dt, a.eps, a.eps, a.penalty_c)
    for e in a.eps_values:
        add("eps", a.grid, a.dt, e, e, a.penalty_c)
    for c in a.penalty_values:
        add("penalty", a.grid, a.dt, a.eps, a.eps, c)
    if a.split_eps:
        add("eps_split", a.grid, a.dt, 2 * a.eps, a.eps, a.penalty_c)
        add("eps_split", a.grid, a.dt, a.eps, 2 * a.eps, a.penalty_c)
    return cases


# Cases currently in flight, so the heartbeat knows what to look at
_RUNNING: dict[str, Path] = {}
_RUNNING_LOCK = threading.Lock()
_STEP_RE = re.compile(r"^Step (\d+) / (\d+):.*?est remaining = ([\d.]+)", re.M)


def last_step(log: Path) -> str:
    """Where a case has got to, from the tail of its own log."""
    try:
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        return "no log yet"
    hits = _STEP_RE.findall(tail)
    if not hits:
        return "starting"
    n, total, rem = hits[-1]
    return f"{int(n):>7}/{total:<7} {float(rem) / 60:5.1f} min left"


def heartbeat(interval: float, stop: threading.Event) -> None:
    while not stop.wait(interval):
        with _RUNNING_LOCK:
            snapshot = sorted(_RUNNING.items())
        if not snapshot:
            continue
        print(f"  --- {len(snapshot)} running ---", flush=True)
        for label, log in snapshot:
            print(f"      {label}  {last_step(log)}", flush=True)


def case_dir(c: dict, root: Path | None = None) -> Path:
    return (root or OUT_ROOT) / (
        f"{c['grid']}x{c['grid']}_dt{c['dt']:g}"
        f"_e{c['eps']:g}_ef{c['eps_flow']:g}_C{c['c']:.0e}"
    )


def run_case(c: dict, a: argparse.Namespace, end_time: float | None = None,
             root: Path | None = None) -> dict:
    d = case_dir(c, root)
    cmd = [
        sys.executable, "-u", "-m", "src.examples.water_freezing.run",
        "--nx", str(c["grid"]), "--ny", str(c["grid"]),
        "--dt", str(c["dt"]),
        "--end-time", str(a.end_time if end_time is None else end_time),
        "--start", a.start,
        "--eps-t", str(c["eps"]),
        "--eps-flow", str(c["eps_flow"]),
        "--penalty-c", str(c["c"]),
        "--vorticity-bc-order", str(a.vorticity_bc_order),
        "--amg-rebuild-every", str(a.amg_rebuild_every),
        "--log-interval", str(a.log_interval),
        "--outdir", str(d),
        "--summary-csv", str(d / "row.csv"),
    ]
    if a.cold_wall_ramp > 0:
        cmd += ["--cold-wall-ramp", str(a.cold_wall_ramp)]
    if a.penalty_time_scheme != "cn":
        cmd += ["--penalty-time-scheme", a.penalty_time_scheme]
    if a.heat_convection != "deferred":
        cmd += ["--heat-convection", a.heat_convection]
    if a.no_latent_convection:
        cmd += ["--no-latent-convection"]
    if a.amg_rebuild_warmup > 0:
        cmd += ["--amg-rebuild-warmup", str(a.amg_rebuild_warmup)]
    if a.warm_start_file is not None:
        cmd += ["--warm-start-file", str(a.warm_start_file)]
    if a.energy_budget:
        cmd += ["--energy-budget"]
    if a.save_interval > 0:
        cmd += ["--save-interval", str(a.save_interval)]

    label = (f"{c['axis']:<10} {c['grid']}x{c['grid']} dt={c['dt']:<6g} "
             f"eps={c['eps']:<5g}/{c['eps_flow']:<5g} C={c['c']:.0e}")
    if a.dry_run:
        print(f"  [dry] {label}", flush=True)
        return {**c, "ok": True, "detail": "dry-run"}

    print(f"  START {label}", flush=True)
    d.mkdir(parents=True, exist_ok=True)
    log = d / "run.log"
    with _RUNNING_LOCK:
        _RUNNING[label] = log
    t0 = time.perf_counter()
    try:
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=REPO, timeout=a.timeout, stdout=fh,
                                  stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        with _RUNNING_LOCK:
            _RUNNING.pop(label, None)
        print(f"  TIMEOUT {label}", flush=True)
        return {**c, "ok": False, "detail": "timeout"}
    finally:
        with _RUNNING_LOCK:
            _RUNNING.pop(label, None)
    el = time.perf_counter() - t0
    if proc.returncode != 0:
        blob = log.read_text(encoding="utf-8", errors="replace")
        detail = "diverged" if "FloatingPointError" in blob else (
            (blob.strip().splitlines() or ["failed"])[-1][:70])
        print(f"  FAIL  {label}  [{el/60:.1f} m] {detail}", flush=True)
        return {**c, "ok": False, "detail": detail}
    print(f"  OK    {label}  [{el/60:.1f} m]", flush=True)
    return {**c, "ok": True, "detail": f"{el/60:.1f} m"}


def collect(cases: list[dict]) -> None:
    rows = []
    for c in cases:
        f = case_dir(c) / "summary.json"
        if not f.exists():
            continue
        d = json.load(open(f, encoding="utf-8"))
        d["axis"] = c["axis"]
        rows.append(d)
    if not rows:
        return
    out = OUT_ROOT / "summary.csv"
    keys = list(rows[0])
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nMerged {len(rows)} result(s) -> {out}")


def main(argv=None) -> None:
    global OUT_ROOT
    a = parse_args(argv)
    OUT_ROOT = a.out_root
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = build_cases(a)

    print(f"{len(cases)} run(s), start={a.start}, ramp={a.cold_wall_ramp:g} s, "
          f"end_time={a.end_time:g} s, {a.jobs} in parallel")
    print(f"baseline: {a.grid}x{a.grid} dt={a.dt:g} eps={a.eps:g} C={a.penalty_c:.0e}\n")

    if a.smoke_time > 0 and not a.dry_run:
        print(f"=== smoke test: {a.smoke_time:g} s of model time per case ===")
        smoke_root = Path(tempfile.mkdtemp(prefix="star_smoke_"))
        try:
            with ThreadPoolExecutor(max_workers=a.jobs) as pool:
                smoke = list(pool.map(
                    lambda c: run_case(c, a, a.smoke_time, smoke_root), cases))
            bad = [r for r in smoke if not r["ok"]]
            if bad:
                print(f"\n{len(bad)} case(s) failed the smoke test; "
                      "not starting the sweep")
                for r in bad:
                    print(f"  {r['axis']:<10} {r['grid']}x{r['grid']} "
                          f"dt={r['dt']:g} eps={r['eps']:g}/{r['eps_flow']:g} "
                          f"C={r['c']:.0e}   {r['detail']}")
                raise SystemExit(1)
            print("all cases survived\n")
        finally:
            shutil.rmtree(smoke_root, ignore_errors=True)
        if a.smoke_only:
            return

    # Cheapest first, with the baseline ahead of everything because every other
    # case is measured against it. Cost goes as steps times cells, so the long
    # time steps and the coarse grids land while the finest grid is still on its
    # first thousand steps. The axes that are cheap to compute are also the ones
    # worth looking at first, so this is not only about finishing sooner.
    cases.sort(key=lambda c: (c["axis"] != "baseline",
                              (a.end_time / c["dt"]) * c["grid"] ** 2))

    t0 = time.perf_counter()
    stop = threading.Event()
    pulse = None
    if a.progress_interval > 0 and not a.dry_run:
        pulse = threading.Thread(target=heartbeat, args=(a.progress_interval, stop),
                                 daemon=True)
        pulse.start()
    try:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            results = list(pool.map(lambda c: run_case(c, a), cases))
    finally:
        stop.set()
        if pulse is not None:
            pulse.join(timeout=2.0)
    el = time.perf_counter() - t0

    print("\n" + "=" * 74)
    print(f"DONE in {el/3600:.2f} h")
    print("=" * 74)
    for r in results:
        print(f"  {'OK  ' if r['ok'] else 'FAIL'}  {r['axis']:<10} "
              f"{r['grid']}x{r['grid']} dt={r['dt']:<6g} "
              f"eps={r['eps']:g}/{r['eps_flow']:g} C={r['c']:.0e}   {r['detail']}")
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(results)} succeeded")
    if not a.dry_run:
        collect(cases)


if __name__ == "__main__":
    main()
