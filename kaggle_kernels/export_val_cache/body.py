# Pull only the 58 radiologist-labeled studies out of every attached cache zip, so ensembles can be scored locally.
# zipfile reads single members, so the 21 GB of shards are never extracted in full.
import zipfile  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, str(ASSETS))
from config import DATA_ROOT, LABELS  # noqa: E402

tr = pd.read_csv(DATA_ROOT / "train.csv")
gt = set(tr[tr[LABELS].notna().all(axis=1)].StudyInstanceUID)
for z in sorted(p for d in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * d + "cache_train_*.zip")):
    kind = "v2" if "-v2-" in z else "v1"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(f"/kaggle/working/val_cache_{kind}_{Path(z).stem}.zip", "w") as dst:
        names = [n for n in src.namelist() if Path(n).stem in gt or n.endswith("meta.csv")]
        for n in names:
            dst.writestr(n, src.read(n))
    print(z, kind, len(names) - 1, "studies", flush=True)
