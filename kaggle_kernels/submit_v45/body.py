# Two-cache ensemble. The ten WindowNet members behind public 0.944 run on cache v4, V5A (3 FS slots of 32 slices) on
# cache v5, and the two submissions merge with fixed shares, so the different input view gets a vote of its own size
# rather than the 1/11 its checkpoint count would give. V5_SHARE is the line that changes between versions.
import os  # noqa: E402

V5_SHARE = 0.15
V4_MEMBERS = ["W8", "W9", "P1", "W11", "W12", "W22", "W23", "W25", "W28", "W27"]
V5_MEMBERS = ["V5A"]
found = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "*kneenet*.pt")})
pick = lambda names: [p for p in found if os.path.basename(p).split("_kneenet")[0] in names]  # noqa: E731
v4, v5 = pick(V4_MEMBERS), pick(V5_MEMBERS)
print("v4", len(v4), "v5", len(v5), flush=True)
runs = [("v4", v4, V4, 6.0), ("v5", v5, V5, 1.5)]
for name, ckpts, env, hours in runs:
    subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts, "--budget-hours", str(hours),
                    "--out", f"/kaggle/working/sub_{name}.csv"], check=True, env={**os.environ, **env})
subprocess.run([sys.executable, str(ASSETS / "merge_submissions.py"), "--out", "/kaggle/working/submission.csv",
                "/kaggle/working/sub_v4.csv", "/kaggle/working/sub_v5.csv",
                "--weights", str(1 - V5_SHARE), str(V5_SHARE)], check=True)
