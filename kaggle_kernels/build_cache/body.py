# Split the train set across several kernel copies only if one kernel exceeds 12 h or the 20 GB output cap.
SHARD, NUM_SHARDS, LIMIT = 0, 1, 0

import os  # noqa: E402

env = {**os.environ, "RSNA_WORK": "/kaggle/tmp/work"}
cmd = [sys.executable, str(ASSETS / "build_cache.py"), "--split", "train",
       "--shard", str(SHARD), "--num-shards", str(NUM_SHARDS), "--zip", "--zip-dir", "/kaggle/working"]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
subprocess.run(cmd, check=True, env=env)
