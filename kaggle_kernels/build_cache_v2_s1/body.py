# Cache v2, shard 1 of 2. The zipped set is about 21 GB, over one kernel's 20 GB output cap.
import os  # noqa: E402

env = {**os.environ, "RSNA_WORK": "/kaggle/tmp/work", **V2}
subprocess.run([sys.executable, str(ASSETS / "build_cache.py"), "--split", "train", "--shard", "1", "--num-shards", "2",
                "--zip", "--zip-dir", "/kaggle/working"], check=True, env=env)
