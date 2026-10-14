# W15. P1 recipe (efficientnet_b0 on pseudo labels round 1) with seed 1. P1 scored 0.939 public alone, W8 0.938.
# Cache v4 comes from the four build kernels, unzipped into /kaggle/tmp, which does not count toward the output cap.
import os  # noqa: E402
import shutil  # noqa: E402
import zipfile  # noqa: E402

zips = sorted(p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "cache_train_*.zip"))
print("cache zips", zips, "free GB", shutil.disk_usage("/kaggle/tmp").free // 2**30 if os.path.isdir("/kaggle/tmp") else None, flush=True)
for z in zips:
    zipfile.ZipFile(z).extractall("/kaggle/tmp/cache")
print("free GB after extract", shutil.disk_usage("/kaggle/tmp").free // 2**30, flush=True)
subprocess.run([sys.executable, str(ASSETS / "train.py"), "--cache", "/kaggle/tmp/cache/train_*",
                "--epochs", "12", "--workers", "4", "--out", "/kaggle/working/kneenet.pt", "--labels", str(ASSETS / "report_labels_pseudo1.jsonl"), *"--arch windows --backbone efficientnet_b0 --bs 2 --lr 1e-3 --backbone-lr 2e-4 --wd 0.02 --dropout 0 --seed 1".split()],
               check=True, env={**os.environ, **V4})
