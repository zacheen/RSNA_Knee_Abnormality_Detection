# Ensemble of rounds 1b, 2b, 5a, 5b, 1c, 5c (Kaggle), L4, L5 (local resnet18 depth-conv), L7 (local resnet34 depth-conv) and C1 (Colab, 5-slot resnet18 depth-conv), 3 checkpoints each with equal weight. The full set is kept rather than the top-scoring subset, because the top combinations differ by less than seed noise.
# The test cache is built with 5 slots because C1 needs them. 3-slot members read only the first 3, which match cache v1 byte for byte.
# Averages every *kneenet*.pt found, so a stale epoch file in an attached dataset would silently join the ensemble.
import os  # noqa: E402

ckpts = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "*kneenet*.pt")})
print("checkpoints", len(ckpts), ckpts, flush=True)
subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts,
                "--out", "/kaggle/working/submission.csv"], check=True, env={**os.environ, **V3})
