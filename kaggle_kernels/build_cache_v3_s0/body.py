# Cache v3, shard 0 of 2. Two shards keep each zip well under the 20 GB output cap and each run well under 12 h.
import os  # noqa: E402

env = {**os.environ, "RSNA_WORK": "/kaggle/tmp/work", **V3}
subprocess.run([sys.executable, str(ASSETS / "build_cache.py"), "--split", "train", "--shard", "0", "--num-shards", "2",
                "--zip", "--zip-dir", "/kaggle/working"], check=True, env=env)
