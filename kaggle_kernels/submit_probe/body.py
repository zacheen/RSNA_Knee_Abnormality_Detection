# Leaderboard probe. Scores a chosen subset of the WindowNet checkpoints, so one submission answers one question
# (a single model, a backbone, a label set). It needs a GPU, since one member of 3 checkpoints projected 21.9 h on
# the CPU against the 9 h limit. MEMBERS is the only line that changes between versions.
import os  # noqa: E402

MEMBERS = ["W8", "W9", "P1", "W11", "W12", "W22", "W23", "W25", "W28", "W27", "W5", "W14", "W26", "W30", "W33", "W34"]
ckpts = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "*kneenet*.pt")
                if os.path.basename(p).split("_kneenet")[0] in MEMBERS})
print("members", MEMBERS, "checkpoints", len(ckpts), ckpts, flush=True)
subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts,
                "--out", "/kaggle/working/submission.csv"], check=True, env={**os.environ, **V4})
