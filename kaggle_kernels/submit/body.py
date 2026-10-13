# Averages every kneenet*.pt found, so a stale epoch file in an attached dataset would silently join the ensemble.
ckpts = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "kneenet*.pt")})
print("checkpoints", ckpts, flush=True)
subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts,
                "--out", "/kaggle/working/submission.csv"], check=True)
