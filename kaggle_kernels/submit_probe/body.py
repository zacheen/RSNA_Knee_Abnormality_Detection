# Leaderboard probe on CPU. Scores a chosen subset of the WindowNet checkpoints, so one submission answers one question
# (a single model, a backbone, a label set). CPU keeps it off the two-session GPU limit and the GPU quota, and one
# member of 3 checkpoints is small enough for the 9 h limit. MEMBERS is the only line that changes between versions.
import os  # noqa: E402

MEMBERS = ["W8"]
ckpts = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "*kneenet*.pt")
                if os.path.basename(p).split("_kneenet")[0] in MEMBERS})
print("members", MEMBERS, "checkpoints", len(ckpts), ckpts, flush=True)
subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts,
                "--out", "/kaggle/working/submission.csv"], check=True, env={**os.environ, **V4})
