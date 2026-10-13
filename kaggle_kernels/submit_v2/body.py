# V2 (common/v2_env.py) must match what train_v2 used, or infer.py refuses the checkpoints.
import os  # noqa: E402

ckpts = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "kneenet*.pt")})
print("checkpoints", ckpts, flush=True)
subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts,
                "--out", "/kaggle/working/submission.csv"], check=True, env={**os.environ, **V2})
