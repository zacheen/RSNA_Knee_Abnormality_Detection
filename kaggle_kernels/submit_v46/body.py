# Two-cache ensemble. The ten WindowNet members behind public 0.944 run on cache v4, W43 and W44 (160 mm crop) on
# cache v6, and the two submissions merge with fixed shares, so the different input view gets a vote of its own size
# rather than the 1/6 its checkpoint count would give. V6_SHARE is the line that changes between versions.
import os  # noqa: E402

V6_SHARE = 0.2
V4_MEMBERS = ["W8", "W9", "P1", "W11", "W12", "W22", "W23", "W25", "W28", "W27"]
V6_MEMBERS = ["W43", "W44"]
found = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "*kneenet*.pt")})
pick = lambda names: [p for p in found if os.path.basename(p).split("_kneenet")[0] in names]  # noqa: E731
v4, v6 = pick(V4_MEMBERS), pick(V6_MEMBERS)
print("v4", len(v4), "v6", len(v6), flush=True)
runs = [("v4", v4, V4, 6.0), ("v6", v6, V6, 2.0)]
for name, ckpts, env, hours in runs:
    subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts, "--budget-hours", str(hours),
                    "--out", f"/kaggle/working/sub_{name}.csv"], check=True, env={**os.environ, **env})
subprocess.run([sys.executable, str(ASSETS / "merge_submissions.py"), "--out", "/kaggle/working/submission.csv",
                "/kaggle/working/sub_v4.csv", "/kaggle/working/sub_v6.csv",
                "--weights", str(1 - V6_SHARE), str(V6_SHARE)], check=True)
