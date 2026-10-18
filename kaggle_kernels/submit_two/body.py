# Two-group ensemble on cache v4. Each group is averaged over its own checkpoints, then the two group predictions merge
# with fixed shares, so a small group of a stronger recipe can get half the vote. B_SHARE and the member lists are the
# lines that change between versions.
import os  # noqa: E402

A_MEMBERS = ["W8", "W9", "P1", "W11", "W12", "W22", "W23", "W25", "W28", "W27"]
B_MEMBERS = ["W39", "W41"]
B_SHARE = 0.5
found = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "*kneenet*.pt")})
pick = lambda names: [p for p in found if os.path.basename(p).split("_kneenet")[0] in names]  # noqa: E731
a, b = pick(A_MEMBERS), pick(B_MEMBERS)
print("a", len(a), "b", len(b), flush=True)
for name, ckpts, hours in [("a", a, 6.0), ("b", b, 2.0)]:
    subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts, "--budget-hours", str(hours),
                    "--out", f"/kaggle/working/sub_{name}.csv"], check=True, env={**os.environ, **V4})
subprocess.run([sys.executable, str(ASSETS / "merge_submissions.py"), "--out", "/kaggle/working/submission.csv",
                "/kaggle/working/sub_a.csv", "/kaggle/working/sub_b.csv",
                "--weights", str(1 - B_SHARE), str(B_SHARE)], check=True)
