# Trains on cache v2 from both shard kernels. Round-2 hyperparameters, 18 epochs, because round 2 was still rising at 10.
import os  # noqa: E402
import zipfile  # noqa: E402

zips = [p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "cache_train_*.zip")]
print("cache zips", zips, flush=True)
for z in zips:
    zipfile.ZipFile(z).extractall("/kaggle/tmp/cache")
subprocess.run([sys.executable, str(ASSETS / "train.py"), "--cache", "/kaggle/tmp/cache/train_*",
                "--labels", str(ASSETS / "report_labels.jsonl"), "--epochs", "18", "--bs", "4", "--workers", "4",
                "--out", "/kaggle/working/kneenet.pt"], check=True, env={**os.environ, **V2})
