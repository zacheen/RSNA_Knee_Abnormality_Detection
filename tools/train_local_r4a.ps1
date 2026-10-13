# Round 4a on the local GPU. resnet34 on cache v2 with round-3 labels and hyperparameters, so it can join round 3 in
# one ensemble. Batch 2, because batch 4 at 48x224 needs about 10 GiB and the local card has 8.
$root = Split-Path $PSScriptRoot -Parent
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:RSNA_CROP_FG = '1'; $env:RSNA_IMG_SIZE = '224'; $env:RSNA_N_SLICES = '48'  # must match kaggle_kernels/common/v2_env.py
$args_ = 'run', '--no-capture-output', '-n', 'rsna_knee', 'python', '-u', 'src/train.py',
    '--cache', 'data/work/cache/train_s*of2', '--labels', 'data/work/report_labels_v2.jsonl',
    '--backbone', 'resnet34', '--epochs', '10', '--bs', '2', '--workers', '4',
    '--out', 'data/work/models/r4a/kneenet.pt'
$p = Start-Process -FilePath 'conda' -ArgumentList $args_ -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$root\data\logs\train_r4a.out.log" -RedirectStandardError "$root\data\logs\train_r4a.err.log"
"started round 4a, pid $($p.Id) at $(Get-Date -Format 'HH:mm')"
