# Round 5c. Round 5a (1b recipe + v2s labels) with seed 1. Seed spread on the 58 is about 0.04, so more seeds steady the ensemble.
import zipfile  # noqa: E402

zips = [p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "cache_train_*.zip")]
print("cache zips", zips, flush=True)
for z in zips:
    zipfile.ZipFile(z).extractall("/kaggle/tmp/cache")
subprocess.run([sys.executable, str(ASSETS / "train.py"), "--cache", "/kaggle/tmp/cache/train_*",
                "--labels", str(ASSETS / "report_labels_v2s.jsonl"), "--epochs", "12", "--bs", "4", "--workers", "4",
                "--lr", "3e-4", "--backbone-lr", "3e-4", "--wd", "1e-4", "--dropout", "0", "--seed", "1",
                "--out", "/kaggle/working/kneenet.pt"], check=True)
