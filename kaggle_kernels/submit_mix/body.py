# Two-family ensemble. The older KneeNet members run on a cache v3 layout, the efficientnet WindowNet members on cache v4,
# and the two submissions are merged with fixed shares, so the older family gets a small vote instead of the share their
# checkpoint count would claim. OLD_SHARE is the line that changes between versions.
import os  # noqa: E402

OLD_SHARE = 0.2
W_MEMBERS = ["W8", "W9", "P1", "W11", "W12"]
found = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "*kneenet*.pt")})
window = [p for p in found if "window-models" in p and os.path.basename(p).split("_kneenet")[0] in W_MEMBERS]
old = [p for p in found if "window-models" not in p]
print("old", len(old), "window", len(window), flush=True)
runs = [("old", old, V3, 3.5), ("window", window, V4, 5.0)]
for name, ckpts, env, hours in runs:
    subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts, "--budget-hours", str(hours),
                    "--out", f"/kaggle/working/sub_{name}.csv"], check=True, env={**os.environ, **env})
subprocess.run([sys.executable, str(ASSETS / "merge_submissions.py"), "--out", "/kaggle/working/submission.csv",
                "/kaggle/working/sub_old.csv", "/kaggle/working/sub_window.csv",
                "--weights", str(OLD_SHARE), str(1 - OLD_SHARE)], check=True)
