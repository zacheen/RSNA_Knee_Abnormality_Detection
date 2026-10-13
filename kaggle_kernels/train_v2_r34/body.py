# Round 4a. resnet34 on cache v2 for backbone diversity in the final ensemble. 10 epochs to save weekly GPU quota, since one is about twice resnet18.
import os  # noqa: E402
import zipfile  # noqa: E402

zips = [p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "cache_train_*.zip")]
print("cache zips", zips, flush=True)
for z in zips:
    zipfile.ZipFile(z).extractall("/kaggle/tmp/cache")
subprocess.run([sys.executable, str(ASSETS / "train.py"), "--cache", "/kaggle/tmp/cache/train_*",
                "--labels", str(ASSETS / "report_labels.jsonl"), "--epochs", "10", "--bs", "4", "--backbone", "resnet34", "--workers", "4",
                "--out", "/kaggle/working/kneenet.pt"], check=True, env={**os.environ, **V2})
