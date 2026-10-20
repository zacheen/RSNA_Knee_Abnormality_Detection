# Cache v6, shard 0 of 4. Four shards keep each zip near 11 GB, under the 20 GB output cap.
import os  # noqa: E402

env = {**os.environ, "RSNA_WORK": "/kaggle/tmp/work", **V6}
subprocess.run([sys.executable, str(ASSETS / "build_cache.py"), "--split", "train", "--shard", "0", "--num-shards", "4",
                "--zip", "--zip-dir", "/kaggle/working"], check=True, env=env)
