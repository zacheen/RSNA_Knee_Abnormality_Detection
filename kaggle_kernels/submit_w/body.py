# WindowNet ensemble on cache v4. W1 (convnext_tiny, epochs 7 to 9 of a run cut short), W2 (resnet34), W3 (convnext_tiny seed 1), W8 and W9 (efficientnet_b0 seeds 0 and 1), 3 checkpoints each with equal weight.
# The older KneeNet members are left out. On the 58 gold studies mixing them in at weights 0 to 0.5 moved the score only within noise, and they would need a second pass over a cache v3 layout.
import os  # noqa: E402

ckpts = sorted({p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "*kneenet*.pt")})
print("checkpoints", len(ckpts), ckpts, flush=True)
subprocess.run([sys.executable, str(ASSETS / "infer.py"), "--ckpt", *ckpts,
                "--out", "/kaggle/working/submission.csv"], check=True, env={**os.environ, **V4})
