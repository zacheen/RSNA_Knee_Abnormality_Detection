# Ensemble of rounds 1b, 2b, 5a, 5b, 1c, 5c (Kaggle) and L4, L5 (local, depth-conv pooling, seeds 0 and 1), all on cache v1, 3 checkpoints each with equal weight. The full set is kept rather than the top-scoring subset, because the top combinations differ by less than seed noise.
# Averages every *kneenet*.pt found, so a stale epoch file in an attached dataset would silently join the ensemble.
ckpts = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "*kneenet*.pt")})
print("checkpoints", len(ckpts), ckpts, flush=True)
subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts,
                "--out", "/kaggle/working/submission.csv"], check=True)
