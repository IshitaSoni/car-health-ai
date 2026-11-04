"""
car-health-ai - run the whole project from the project root
===========================================================
    python src/run_all.py            # everything, in order
    python src/run_all.py --from 4   # restart from step 4
    python src/run_all.py --list     # show the steps

Order: data -> EDA -> Week 3 models -> Week 4 health score -> Week 7 monotone
models -> Week 5 explanations and reports -> tests. Stops at the first failure.
Scripts may live in the project root (Weeks 1-4) or in src/ (later weeks); both are found.
Modules that src/ scripts import (health_score, train_models, generate_data) are
copied into src/ if missing there.
Then start the dashboard with:  streamlit run src/app.py
"""
import os, shutil, subprocess, sys, time

ROOT = os.getcwd()
SHARED = ["health_score.py", "train_models.py", "generate_data.py"]


def find(name):
    for d in (".", "src"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in the project root or src/")


def sync_shared():
    os.makedirs("src", exist_ok=True)
    for m in SHARED:
        if not os.path.exists(os.path.join("src", m)) and os.path.exists(m):
            shutil.copy(m, os.path.join("src", m))
            print(f"copied {m} into src/")


STEPS = [
    ("Generate the dataset (skipped if it exists)", ["generate_data.py"],
     lambda: os.path.exists("data/car_health_ai_dataset.csv")),
    ("Exploratory data analysis (Week 2)", ["eda.py"], None),
    ("Train the Week 3 models", ["train_models.py"], None),
    ("Health Score, calibration, validation (Week 4)", ["week4_health_score.py"], None),
    ("Monotone-constrained models (Week 7)", ["week7_monotonic.py"], None),
    ("Explanations and example reports (Week 5)", ["week5_reports.py"], None),
    ("Automated tests", ["-m", "unittest", "discover", "-s", "tests"], None),
]


def main():
    if "--list" in sys.argv:
        for i, (name, *_r) in enumerate(STEPS, 1):
            print(i, name)
        return
    start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 1
    os.makedirs("data", exist_ok=True)
    sync_shared()
    for i, (name, cmd, skip) in enumerate(STEPS, 1):
        if i < start:
            continue
        if skip and skip():
            print(f"[{i}/{len(STEPS)}] {name}: skipped")
            continue
        args = cmd if cmd[0] == "-m" else [find(cmd[0])] + cmd[1:]
        print(f"\n[{i}/{len(STEPS)}] {name}")
        t = time.time()
        r = subprocess.run([sys.executable] + args, cwd=ROOT)
        if r.returncode != 0:
            print(f"\nStep {i} failed (exit {r.returncode}). Fix it, then rerun: python src/run_all.py --from {i}")
            sys.exit(r.returncode)
        print(f"    done in {time.time() - t:.0f}s")
    print("\nAll steps finished. Start the dashboard with: streamlit run src/app.py")


if __name__ == "__main__":
    main()