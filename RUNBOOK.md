# RSNA Knee 執行手冊

這份手冊從零開始，一步一步做到 Kaggle 上的一份提交，並且在同一題上跑通用 AI agent 當對照組。每一步都附上指令，以及本機用小量資料實測過的結果。

所有指令都在 repo 根目錄 `RSNA_Knee_Abnormality_Detection/` 的 Git Bash 裡執行。

## 目錄

0. 整體架構
1. 一次性前置作業
2. 取得資料
3. 本機冒煙測試（小量資料）
4. 雲端完整執行
5. Agent 對照實驗
6. 實測紀錄
7. 已知限制

## 0. 整體架構

原始資料 570 GB，本機放不下，所以分工如下。

| 位置 | 做什麼 | 為什麼放這裡 |
|---|---|---|
| 本機 | 報告抽標籤、寫程式、冒煙測試、跑 agent | 抽標籤只需要幾 MB 的 csv，agent 迴圈要長時間連網 |
| Kaggle CPU kernel | DICOM 轉成壓縮 cache | 比賽資料已掛載在雲端，不用下載 |
| Kaggle GPU kernel | 訓練、推論、提交 | 免費 GPU，而且提交本來就必須在 Kaggle notebook 上 |

資料流向。

```text
train.csv 的報告 ──report_labels.py（本機, LLM）──> report_labels.jsonl ─┐
                                                                         ├─> train kernel ──> kneenet_epNN.pt ──> submit kernel ──> submission.csv
train_series/ DICOM ──build_cache kernel（雲端 CPU）──> cache zip ────────┘
```

每個 study 被壓成 `(3 個切面, 32 張切片, 192, 192)` 的 uint8 陣列，約 3.5 MB。切面依序是 Sagittal、Coronal、Axial，每個切面優先挑 fluid-sensitive 的 series，同類中再挑切片最多的。

程式檔案。

| 檔案 | 用途 |
|---|---|
| `src/config.py` | 路徑與常數，自動判斷本機或 Kaggle |
| `src/dicom_io.py` | 讀一個 series，只解碼要用的那 32 張 |
| `src/build_cache.py` | 每個 study 轉成 cache，`infer.py` 共用同一個函式 |
| `src/report_labels.py` | 報告轉 12 個三態標籤（1 有、0 明確沒有、-1 沒提到） |
| `src/model.py` | 2.5D 模型，每張切片共用一個 2D encoder，每個切面做 attention pooling |
| `src/train.py` | 訓練，-1 不算進 loss，58 筆放射科醫師標註當驗證集 |
| `src/infer.py` | 產生 submission.csv，超過時間預算的 study 填 0.5 |
| `kaggle_kernels/*/` | 四支雲端 kernel（打包子集、建 cache、訓練、提交）的設定與主體 |
| `tools/push_assets.sh` | 把 `src/` 和標籤同步到私有 Kaggle dataset |
| `tools/push_kernel.sh` | 組出 kernel 的 `run.py` 並推上 Kaggle |
| `agents/` | Agent 對照實驗的工作區準備、任務說明、執行與評分 |

## 1. 一次性前置作業

### 1.1 Kaggle 帳號

1. 在比賽頁面按 Join Competition 並接受規則。沒接受的話 API 與 kernel 都讀不到資料。
2. 在 Kaggle 帳號設定完成手機驗證。沒驗證的帳號在 kernel 裡不能開 GPU 與網路。

### 1.2 Kaggle API token

在 https://www.kaggle.com/settings/api 產生 token，存成 `~/.kaggle/access_token`，檔案內容只有那串 token。

```bash
mkdir -p ~/.kaggle && printf '%s' '<你的 token>' > ~/.kaggle/access_token
```

驗證。最後一欄 `userHasEntered` 要是 `True`。

```bash
conda run -n rsna_knee kaggle competitions list -s rsna-knee
```

`kaggle_kernels/*/kernel-metadata.json` 與 `kaggle_datasets/rsna-knee-assets/dataset-metadata.json` 裡寫死了 Kaggle 帳號名稱。換帳號的話這四個檔案都要改。

### 1.3 Python 環境

這個專案使用獨立的 conda 環境 `rsna_knee`，不碰 `ML`。

```bash
conda create -n rsna_knee python=3.12 -y
```

```bash
conda run -n rsna_knee pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
conda run -n rsna_knee pip install kaggle pandas numpy pyarrow pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm timm scikit-learn opencv-python-headless tqdm
```

RTX 50 系列（Blackwell）需要 cu128 版的 torch 才有 `sm_120`。

驗證 GPU。

```bash
conda run -n rsna_knee python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Windows 的 console 是 cp950。輸出含非 ASCII 字元時 `conda run` 會在轉印時崩潰，所以會印出報告文字的指令一律加 `--no-capture-output`。

## 2. 取得資料

### 2.1 五個 csv

```bash
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do conda run -n rsna_knee kaggle competitions download -c rsna-knee-abnormality-detection -f "$f" -p data/raw; done
```

### 2.2 小量 DICOM 子集

Kaggle CLI 不能下載整個資料夾，逐檔下載大約 100 個檔案就會被限流。所以改在雲端用一個 kernel 把子集打包成單一 zip，再下載 kernel 的輸出。

預設打包 20 個有放射科醫師標註的 study，加上 3 個 test 範例 study。數量由 `kaggle_kernels/export_subset/export_subset.py` 的 `N_LABELED` 控制。

```bash
conda run -n rsna_knee kaggle kernels push -p kaggle_kernels/export_subset
```

等狀態變成 COMPLETE。

```bash
conda run -n rsna_knee kaggle kernels status <帳號>/rsna-knee-export-subset
```

下載並解壓到 `data/raw/`，解壓後的目錄結構與比賽原始資料相同。

```bash
conda run -n rsna_knee kaggle kernels output <帳號>/rsna-knee-export-subset -p data/subset_zip && unzip -q -o data/subset_zip/subset.zip -d data/raw
```

實測。20 個 train study 加 3 個 test study 共 4,080 個檔案，zip 3.15 GB。雲端打包約 30 分鐘，下載約 11 分鐘，解壓 31 秒。

一個 slice 的完整路徑約 280 字元，超過 Windows 預設的 260 字元上限。`src/dicom_io.py` 的 `long_path()` 會自動加上 `\\?\` 前綴處理，不需要改系統設定。但用 Python 以外的工具（例如檔案總管）開這些檔案可能會失敗。

## 3. 本機冒煙測試（小量資料）

這一節只驗證程式能從頭跑到尾，數字沒有意義。所有輸出都寫到 `data/work_smoke/`，不會污染正式結果。

```bash
export RSNA_WORK=data/work_smoke
```

### 3.1 建 cache

```bash
conda run -n rsna_knee python src/build_cache.py --split train --only-present
```

```bash
conda run -n rsna_knee python src/build_cache.py --split test
```

實測。20 個 study 用 16 個 worker 共 9 秒。單一 worker 每個 study 平均 2.17 秒，中位數 1.95 秒，最慢 5.44 秒。三個切面的覆蓋率都是 100%。

每個 cache 檔 3.5 MB，用 deflate level 1 壓縮後剩 65%。

確認 cache 內容正確的方法是把切片拼成一張圖來看。三列應該依序是 Sagittal、Coronal、Axial，每列從一側掃到另一側，而且是脂肪抑制的亮水訊號序列。

```bash
conda run -n rsna_knee python -c "import numpy as np,cv2,glob; v=np.load(sorted(glob.glob('data/work_smoke/cache/train_*/*.npy'))[0]); cv2.imwrite('data/work_smoke/montage.png', np.concatenate([np.concatenate([v[p,i] for i in range(0,32,4)],1) for p in range(3)],0))"
```

### 3.2 報告抽標籤

離線版，不呼叫 LLM，只用來確認流程。

```bash
conda run -n rsna_knee python src/report_labels.py --backend keyword --only-gt
```

LLM 版，走本機的 `claude` CLI，計入 Claude Code 訂閱額度，不需要 API key。

```bash
conda run --no-capture-output -n rsna_knee python src/report_labels.py --backend claude_cli --only-gt --limit 10
```

跟 58 筆放射科醫師標註比對。

```bash
conda run -n rsna_knee python src/report_labels.py --score
```

實測。`claude_cli` 一批 10 份報告約 91 秒，結果都是合法的三態值。換算全量 4,407 份約 441 批，單一 worker 約 11 小時，4 個 worker 約 3 小時（這是推算，還沒跑過全量）。

在第一批 10 份裡，Effusion 有被報告提到的 study 中只有一半跟放射科醫師一致。一個西班牙文報告寫輕微積液，醫師標的是陽性，但 prompt 的門檻是中量以上。樣本只有 10 份，不足以下結論，但全量跑完後 `--score` 要特別看這一欄。

關鍵字版拿來當 B 層的下限。在 B 層藏起來的 29 筆上，macro AUC 是 0.6496。

### 3.3 訓練

本機只有 20 個 study，所以加 `--train-on-gt` 讓訓練集與驗證集重疊。正式訓練不能加這個參數。

```bash
conda run -n rsna_knee python src/train.py --cache "data/work_smoke/cache/train_*" --labels data/work_smoke/report_labels.jsonl --epochs 2 --train-on-gt
```

實測。RTX 5060 Laptop 8 GB，batch size 2，每個 epoch 約 25 秒，相當於每個 study 每個 epoch 約 1.25 秒。換算全量 4,349 個 study，本機一個 epoch 約 90 分鐘。

這一步的 AUC 約 0.42，低於亂猜，這是預期中的結果。20 個 study 配關鍵字標籤只能證明流程能跑，不能說明模型好壞。

### 3.4 推論

用 3 個 test 範例 study 產生 submission.csv，並印出換算成 1300 個 study 的預估時間。

```bash
conda run -n rsna_knee python src/infer.py --ckpt data/work_smoke/models/kneenet_ep*.pt
```

實測。3 個 study 共 18 秒，輸出的欄位名稱與 `sample_submission.csv` 完全相同。

這裡印出的 1300 個 study 預估時間（約 2.1 小時）不可靠，因為 3 個 study 的時間大部分花在開 process pool 與載入模型。可靠的數字要等第 4.5 節在雲端跑。

### 3.5 雲端冒煙測試

本機跑通之後，用很小的量把三支雲端 kernel 各跑一次，確認雲端環境沒有問題。這一步用關鍵字版標籤，正式執行前會被覆蓋。

```bash
conda run -n rsna_knee python src/report_labels.py --backend keyword
```

```bash
LABELS=data/work_smoke/report_labels.jsonl bash tools/push_assets.sh create
```

把 `kaggle_kernels/build_cache/body.py` 第二行改成 `SHARD, NUM_SHARDS, LIMIT = 0, 1, 30`，再依序推三支 kernel，每支等前一支 COMPLETE 再推下一支。

```bash
bash tools/push_kernel.sh build_cache
```

```bash
bash tools/push_kernel.sh train
```

```bash
bash tools/push_kernel.sh submit
```

查 log 的方法。

```bash
conda run -n rsna_knee kaggle kernels output <帳號>/rsna-knee-build-cache -p data/kernel_logs --file-pattern ".*\.log"
```

實測，三支都跑到 COMPLETE。

| kernel | 結果 |
|---|---|
| build_cache | 30 個 study 29 秒，每個 study 0.96 秒（4 個 worker），zip 0.07 GB，沒有 `[warn]` |
| train | 29 個訓練 study，每個 epoch 約 3 秒（batch size 4）。驗證集只有 1 筆標註，所以 AUC 是 nan，這是預期中的 |
| submit | 離線環境從 wheel 裝好 pylibjpeg 三個套件，3 個 study 11 秒，產出合法的 submission.csv |

雲端的掛載路徑跟官方文件寫的不同，程式已經自動處理，這裡記下來方便查 log。

| 掛載的東西 | 實際路徑 |
|---|---|
| 比賽資料 | `/kaggle/input/competitions/rsna-knee-abnormality-detection` |
| 自己的 dataset | `/kaggle/input/datasets/<帳號>/rsna-knee-assets` |
| 另一支 kernel 的輸出 | `/kaggle/input/notebooks/<帳號>/<kernel 名稱>` |

## 4. 雲端完整執行

第 3.5 節跑完後，雲端留著兩個冒煙測試的設定，正式執行前都要換掉。

- `rsna-knee-assets` 裡的 `report_labels.jsonl` 是關鍵字版。第 4.2 節重新上傳時會換成第 4.1 節的 LLM 版。
- `kaggle_kernels/build_cache/body.py` 的 `LIMIT` 是 30。第 4.3 節會改回 0。

正式結果寫到 `data/work/`，先把冒煙測試的環境變數清掉。

```bash
unset RSNA_WORK
```

### 4.1 全量報告抽標籤（本機）

4,407 份報告，每批 10 份，4 個 worker 推算約 3 小時。中斷後重跑同一行指令會從上次停下的地方接續，因為已完成的 study 會被跳過。最後如果還有 study 沒標到，指令會以非零狀態結束並印出剩幾個，這時再跑一次同一行即可。

這一步會消耗 Claude Code 訂閱額度。額度用完時批次會失敗，等額度恢復後重跑即可。

實測第一次執行在 1 小時 40 分內標完 2,950 份（每秒約 0.5 份），之後連續 146 批失敗，`claude` 以狀態 1 結束而 stderr 是空的，模式符合撞到訂閱額度上限。幾個小時後重跑同一行指令就恢復正常。現在的錯誤訊息會連 stdout 一起印出，下次可以直接看到原因。

建議用背景方式啟動，關掉終端機也會繼續跑。PowerShell 的寫法如下。

```powershell
Start-Process conda -ArgumentList 'run','--no-capture-output','-n','rsna_knee','python','-u','src/report_labels.py','--backend','claude_cli','--workers','4' -RedirectStandardOutput data\logs\report_labels.out.log -RedirectStandardError data\logs\report_labels.err.log -WindowStyle Hidden
```

```bash
conda run --no-capture-output -n rsna_knee python src/report_labels.py --backend claude_cli --workers 4
```

```bash
conda run -n rsna_knee python src/report_labels.py --score
```

`--score` 的結果要先看過再往下走。如果某個標籤的 recall 很低，代表 prompt 對那個標籤的門檻跟放射科醫師不一致，要回頭改 `src/report_labels.py` 的 `CRITERIA` 再重抽那個標籤。

### 4.2 上傳程式與標籤到 Kaggle

`tools/push_assets.sh` 會把 `src/*.py`、`data/work/report_labels.jsonl`、`data/wheels/*.whl` 平鋪到私有 dataset `rsna-knee-assets`。三支雲端 kernel 都掛載這個 dataset。

提交時 kernel 不能連網，所以 Kaggle 映像檔裡沒有的套件要先下載成 wheel 一起上傳。實測 Kaggle 映像檔是 Python 3.13.15，已經有 pydicom、timm、cv2，缺的是 pylibjpeg 這三個解碼套件。其他三個也一起帶著，以免映像檔更新後消失。

`--no-deps` 不能省略。少了它，timm 會把 torch 也一起拉下來，多出將近 1 GB。

```bash
conda run -n rsna_knee pip download --no-deps --only-binary=:all: --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 --python-version 3.13 -d data/wheels pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg timm opencv-python-headless
```

第一次建立。

```bash
bash tools/push_assets.sh create
```

之後每次改了 `src/` 或標籤都要更新，否則雲端跑的是舊程式。

```bash
bash tools/push_assets.sh version "說明這次改了什麼"
```

### 4.3 建 cache（雲端 CPU）

先把 `kaggle_kernels/build_cache/body.py` 第二行的 `LIMIT` 改回 0，否則只會處理 30 個 study。

```bash
bash tools/push_kernel.sh build_cache
```

輸出是 `/kaggle/working/cache_train_s0of1.zip`。

單一 kernel 有 12 小時與 20 GB 輸出兩個上限。雲端冒煙測試量到每個 study 0.96 秒、壓縮後 2.3 MB，換算全量約 1.2 小時、約 10 GB，單一 kernel 放得下。如果之後改大解析度或切片數超過上限，把 `kaggle_kernels/build_cache/` 複製成多份，每份改 `body.py` 第一行的 `SHARD`、`NUM_SHARDS` 與 `kernel-metadata.json` 的 `id`，再把 train kernel 的 `kernel_sources` 改成列出全部分片。

### 4.4 訓練（雲端 GPU）

```bash
bash tools/push_kernel.sh train
```

train kernel 會開網路，因為 timm 要下載 ImageNet 預訓練權重。輸出是最後 3 個 epoch 的 `/kaggle/working/kneenet_epNN.pt`，推論時三個一起平均。log 裡每個 epoch 的 `val_macro_auc` 與最後一行 `averaged last 3 epochs` 都是在 58 筆放射科醫師標註上的分數，後者才是實際提交的模型。

epoch 數設成 12。T4 上全量每個 epoch 實測約 9 分鐘，12 個 epoch 約 1.8 小時。`train.py` 另有 `--budget-hours`（預設 11），每個 epoch 結束時如果預估下一個 epoch 會超過預算就提前停下並保留目前最好的權重，因為被 12 小時上限砍掉的 kernel 不會留下任何輸出。提前停下時 learning rate 排程沒有走完，分數會比跑完差，所以它只是保險，不能拿來取代調 epoch 數。

### 4.5 推論與提交（雲端 GPU, 離線）

```bash
bash tools/push_kernel.sh submit
```

這一步在 commit 時只會跑 3 個 test 範例。真正的 1300 個 study 只在提交後的重跑才出現，所以先看 log 裡的 `projected 1300 studies` 預估時數，確認低於 9 小時再提交。

提交會占用每日提交次數，所以這一步由你自己決定時機。

也可以用 `tools/chain_submit.sh` 一次做完。它會等 train kernel 結束、跑 submit kernel、讀出預估時數，低於 8 小時才提交，任何一步失敗就停下不提交。

```bash
bash tools/chain_submit.sh "這次提交的說明"
```

```bash
conda run -n rsna_knee kaggle competitions submit rsna-knee-abnormality-detection -k <帳號>/rsna-knee-submit -v <kernel 版本號> -f submission.csv -m "baseline resnet18 2.5D"
```

查分數。

```bash
conda run -n rsna_knee kaggle competitions submissions rsna-knee-abnormality-detection
```

### 4.6 Colab 與本機訓練

Kaggle 的 GPU 額度留給提交，訓練放在 Colab 與本機。

Colab 的程式、標籤與 runner 由下面這行同步到 Drive 的 `rsna_knee` 資料夾。runner.ipynb 若可能還開著，加上 `KEEP_RUNNER=1`，因為 Colab 的自動存檔會蓋掉剛同步過去的版本。

```bash
bash tools/sync_colab.sh
```

- `runner_A.ipynb` 讀 `jobs_A.txt`，`runner_B.ipynb` 讀 `jobs_B.txt`，兩個可以同時開，不會搶同一個工作。每行是 `<名稱> <標籤檔> <seed> [train.py 參數]`，有 `runs/<名稱>/DONE` 的工作會被跳過。
- `cache_kernels.txt` 決定下載哪一份 cache。cache v4 需要的切片數、尺寸、裁切與範圍以 `env KEY=VALUE` 行寫在裡面，train.py 會拒絕設定不符的 cache。
- runner 用 `--resume` 訓練，checkpoint 與 `resume_state.pt` 每個 epoch 直接寫進 Drive 的 `runs/<名稱>/`。runtime 斷線後重新全部執行，未完成的工作會從最後一個完成的 epoch 接著跑，最多損失一個 epoch。
- cache 的 zip 第一次從 Kaggle 下載後，會複製一份到 Drive 的 `cache_zips/<kernel>/`，旁邊的 `.ok` 記錄檔案大小。之後的 session 直接從 Drive 複製，不必再花約 30 分鐘從會中途斷線的 Kaggle 下載。Drive 上的大小和 `.ok` 對不上時，會改回從 Kaggle 下載。cache kernel 重建過的話，要手動刪掉 `cache_zips` 底下那個 kernel 的資料夾。
- runner 最後一格在工作做完後釋放 runtime，所以整晚放著也不會一直耗運算單元。前面的格子出錯時不會跑到最後一格，這時 runtime 不會自動釋放。
- GPU 選 L4。A100 的運算單元耗用約是 L4 的 2 到 3 倍，T4 約慢一倍。沒有可用 GPU 時 Colab 會問要不要用 CPU，要選取消。

Colab 的進度由 Claude 自己定時檢查，不靠使用者回報。runner 開始後，在背景執行下面這行，它在工作完成或 log 停止更新 40 分鐘時結束，後者也會跳出 Windows 通知。

```bash
bash tools/watch_colab.sh W4 W5
```

需要使用者動手的事，例如打開 runner、選 GPU、重新連線，都要另外跳出 Windows 通知。通知用 Daily_Task 的共用腳本，訊息裡不能有雙引號與冒號。和通知有關的事都先讀 `D:\dont_move\git_save\Daily_Task\shared\NOTIFICATIONS.md`。
- `-Category "RSNA Knee"` 讓通知在通知中心自成一組，不會和其他專案的通知混在一起。
- `-Title` 寫這則通知的性質，例如「需要你操作」、「警告」或「結果」。
- 每個分類用自己的 AppId 送出，第一次送出時腳本會在開始功能表建立 `RSNA Knee.lnk`，Windows 才會接受這個 AppId 的通知。所以「RSNA Knee」在通知中心與「設定 → 系統 → 通知」裡都是獨立的應用程式，可以單獨關掉。
- 通知收不到時，先查 `[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('ClaudeCode.RSNAKnee').Setting` 是不是 `Enabled`，不要先懷疑呼叫端。

```bash
powershell.exe -NoProfile -File "D:\dont_move\git_save\Daily_Task\shared\notify.ps1" -Message "請打開 runner_A" -Title "RSNA Knee 需要你操作" -Category "RSNA Knee"
```

本機訓練用 `tools/train_local_queue.sh`。有三件事要注意。
- 從 Claude Code 啟動的程序會在 Claude Code 重新啟動時一起結束，所以長時間的訓練改用 WMI 啟動，例如 `data/logs/run_W9.sh`。
- numpy 底層的 Intel Fortran 程式庫收到 Windows 的關閉視窗事件時會自行中止，log 裡是 `forrtl: error (200)`。佇列腳本已經設好 `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` 來關掉這個處理。
- WindowNet 在本機用 `--workers 0`，因為 DataLoader 的子程序在 Windows 上曾經整個卡住而不報錯。

## 5. Agent 對照實驗

### 5.1 實驗設計

通用 agent 吃不下原始資料的形狀，所以把題目切成幾層，每一層交給 agent 的東西不同，量到的差距也不同。

| 層 | agent 拿到什麼 | agent 要自己想出什麼 | 怎麼評分 | 量到的是什麼 |
|---|---|---|---|---|
| A 全生 | 比賽名稱、Kaggle CLI、空資料夾 | 全部，包括怎麼在雲端處理 570 GB | Kaggle 公開榜分數 | agent 能不能獨立參賽 |
| B 報告標籤 | 4,407 份報告、29 筆看得到的標註 | 怎麼從多語言報告抽 12 個標籤 | 另外 29 筆藏起來的標註上的 macro AUC | agent 在它擅長的子任務上的能力 |
| C 建模 | 下載好的 cache、標籤 | 模型架構與訓練方式 | 58 筆標註上的驗證分數，或公開榜 | 純建模能力 |
| D copilot | 全部，由你下指令 | 只負責執行 | 公開榜 | agent 幫你加速多少 |

人的對照組就是第 3、4 節這條 pipeline 的產出。B 層用同一個評分腳本評 `data/work/report_labels.jsonl`，C 層與 A 層比公開榜分數。

C 層有一個要注意的地方。cache 本身已經包含「挑哪些 series、取幾張切片」這些領域判斷，所以 C 層會低估差距。

### 5.2 B 層

先建工作區。工作區放在 repo 外面，降低 agent 順手讀到 `data/raw/train.csv` 裡完整 58 筆標註的機會。

```bash
conda run -n rsna_knee python agents/prepare_B.py --ws D:/rsna_agent_ws/B_claude
```

這一步同時把藏起來的 29 筆寫到 `data/agent_hidden/B_hidden_gt.csv`。切分固定用 `random_state=0`，所以每個 agent 與人的對照組都在同樣的 29 筆上評分。

跑 agent，最後一個參數是時數上限。

```bash
bash agents/run_agent.sh claude D:/rsna_agent_ws/B_claude 2
```

```bash
bash agents/run_agent.sh codex D:/rsna_agent_ws/B_codex 2
```

Codex 預設用 `gpt-5.5`，可以用環境變數 `CODEX_MODEL` 換掉。這台機器的 Codex 設定檔預設模型是 `gpt-6.1-sol`，但 ChatGPT 帳號登入時不支援，直接用會回 400。Codex 的 `--full-auto` sandbox 預設會關掉網路，所以 Codex agent 不能呼叫雲端 LLM，只能靠自己讀報告或寫規則，這也是比較時要記得的條件差異。

AIDE 需要 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`，並且要先 `conda run -n rsna_knee pip install aideml`。

```bash
bash agents/run_agent.sh aide D:/rsna_agent_ws/B_aide 2
```

評分，agent 的與人的用同一支腳本。

```bash
conda run -n rsna_knee python agents/grade_B.py D:/rsna_agent_ws/B_claude/predictions.csv
```

```bash
conda run -n rsna_knee python agents/grade_B.py data/work/report_labels.jsonl
```

跑完後檢查 agent 有沒有讀到不該讀的檔案。

```bash
grep -c "agent_hidden\|data/raw/train.csv" D:/rsna_agent_ws/B_claude/agent_log.jsonl
```

### 5.3 C 層

前提是第 4.3 節的 cache 已經建好。把 cache zip 下載到本機並解壓，大約 15 GB。

```bash
conda run -n rsna_knee kaggle kernels output <帳號>/rsna-knee-build-cache -p data/cloud_cache && unzip -q -o "data/cloud_cache/*.zip" -d data/work/cache
```

任務說明照 `agents/tasks/B_report_labels.md` 的格式寫一份 C 版本，交給 agent 的資料是 `data/work/cache/`、`report_labels.jsonl` 與 29 筆看得到的標註，評分用另外 29 筆。

### 5.4 A 層

給 agent 一個空資料夾與一段任務說明，要求它用 Kaggle CLI 在雲端完成整條流程並提交。這一層 agent 會拿到你的 Kaggle token，所以任務說明裡要限制提交次數，並且要求所有 kernel 與 dataset 維持私有。

A 層一次要燒掉很多 token 與 Kaggle GPU 時數，建議 B 層結果出來之後再決定要不要跑。

## 6. 實測紀錄

每一行都是在某個版本的程式上量到的，日期用來判斷數字是否已經過時。

| 日期 | 項目 | 數字 |
|---|---|---|
| 2026-10-09 | train.csv | 4,407 個 study，58 個有 12 欄標註，其餘 4,349 個 12 欄全是 NaN，沒有只填一部分的 |
| 2026-10-09 | 每個 study 的 series 數 | 平均 5.53，中位 5，最少 3，最多 14 |
| 2026-10-09 | 切面分布 | Sagittal 9,864、Coronal 8,609、Axial 5,898 |
| 2026-10-09 | Fluid_Sensitive 與 Fat_Suppression | 在 train 裡完全一致，沒有任何一個 series 兩者不同 |
| 2026-10-09 | 報告文字 | 拉丁字母 3,866、希臘字母 321、西里爾字母 220。拉丁字母裡約一半是英文 |
| 2026-10-09 | 58 筆標註的陽性數 | ACL 24、MCL 9、Medial Meniscus 26、Lateral Meniscus 23、Medial OA 15、Lateral OA 11、PF OA 21、Effusion 35、Synovitis 27、Baker's 12、Contusion 19、Fracture 18 |
| 2026-10-09 | 子集下載 | 23 個 study，4,080 個檔案，3.15 GB |
| 2026-10-09 | 本機建 cache | 單一 worker 每 study 2.17 秒，壓縮率 65% |
| 2026-10-09 | 雲端建 cache（30 個 study） | 4 個 worker 每 study 0.96 秒，壓縮後每 study 2.3 MB |
| 2026-10-09 | 雲端建 cache（全量） | 4,407 個 study，處理 3,151 秒（每 study 0.72 秒），加上壓縮整支 kernel 約 1 小時 37 分。zip 10.46 GB，0 個 `[warn]`，三個切面覆蓋率都是 100% |
| 2026-10-09 | claude_cli 抽標籤 | 10 份報告一批 91 秒 |
| 2026-10-09 | B 層關鍵字下限 | 藏起來的 29 筆上 macro AUC 0.6496 |
| 2026-10-09 | 全量抽標籤（prompt v1） | 4,407 份全部完成，分三次執行，中間兩次撞到訂閱額度上限，實際執行時間合計約 2.5 小時 |
| 2026-10-09 | B 層人工對照組（prompt v1） | 藏起來的 29 筆上 macro AUC 0.7471 |
| 2026-10-09 | prompt v1 對 58 筆標註的 recall | ACL 0.92、Medial Meniscus 0.88 很好。PF OA 0.48、MCL 0.44、Synovitis 0.44、Effusion 0.60 偏低，報告寫沒有或沒提到，但醫師看影像標成有 |
| 2026-10-09 | 本機訓練 | RTX 5060 8 GB，每 study 每 epoch 約 1.25 秒 |
| 2026-10-09 | 雲端訓練 | Tesla T4，29 個 study 每 epoch 約 3 秒 |
| 2026-10-09 | 雲端推論 | 3 個 study 兩次分別 11 秒與 14 秒 |

全量推算。

| 步驟 | 推算時間 | 依據 |
|---|---|---|
| 全量抽標籤 | 約 3 小時 | 441 批，4 個 worker |
| 雲端建 cache | 約 1.2 小時，約 10 GB | 4,407 × 0.96 秒 |
| 雲端推論 1300 個 study | 1.7 小時以內 | 兩次 3 個 study 的換算值是 1.28 與 1.65 小時，含啟動時間，所以是上限 |

## 7. 已知限制

1. 左右膝方向沒有統一。左膝與右膝在影像上的內外側方向相反，而 Medial 與 Lateral 是不同的標籤。第 1 輪改成訓練時隨機做左右鏡像、推論時把原圖與鏡像平均，讓模型兩種方向都看過，但沒有依 DICOM 方向標籤把影像翻成同一個方向。鏡像本身不需要交換 Medial 與 Lateral 標籤，因為鏡像後的左膝就是右膝，內側半月板還是內側。
2. 非正方形影像會被壓扁。子集裡有 640×1280 的切片，`cv2.resize` 直接縮成 192×192，沒有保留長寬比。
3. 58 筆標註太少。B 層每一邊只有 29 筆，C 層驗證也只有 58 筆，所以 AUC 差 0.02 以內的比較沒有統計意義。
4. Agent 隔離只靠任務說明與事後檢查。`data/raw/train.csv` 含完整 58 筆標註，`claude` 以 bypassPermissions 執行時技術上讀得到。真正隔離要用 Docker 或另一台機器，目前用第 5.2 節的 grep 事後檢查代替。
5. 報告標籤與影像標註的門檻可能不同。放射科醫師是看影像標的，報告寫的是另一位醫師的判讀，兩者本來就會不一致，Effusion 是目前看到的例子。
6. 本機子集裡抽樣的 503 張全部是 Explicit VR Little Endian，所以 JPEG Lossless 與 JPEG 2000 的解碼沒有在本機實際碰到。雲端全量 4,407 個 study 跑完沒有任何 `[warn]`，代表被選中的 series 都解得開。沒被選中的 series 沒有讀過，所以這不代表整份資料都解得開。
7. Kaggle 帳號名稱寫死在 `kaggle_kernels/*/kernel-metadata.json` 與 `kaggle_datasets/rsna-knee-assets/dataset-metadata.json`。

## 8. 優化紀錄

每一輪先分別問 Codex 與 Antigravity，再綜合成結論，原始回覆與綜合結論放在 `data/rounds/r<N>/`。

在拿到排行榜分數之前，每一輪用 58 筆放射科醫師標註上的驗證 AUC 判斷。這 58 筆同時是挑 checkpoint 的依據，所以差 0.02 以內只能當成雜訊。報告標籤的 prompt 只能用 B 層看得到的 29 筆調整。

| 輪 | 日期 | 改了什麼 | 驗證 macro AUC | 排行榜 |
|---|---|---|---|---|
| 0 | 2026-10-09 | baseline，resnet18 2.5D，prompt v1 標籤，6 epoch | 0.8167 | 帳號未完成身分驗證，無法提交 |
| 1 | 2026-10-10 | 左右鏡像增強與鏡像 TTA，12 epoch，標籤仍是 v1 | 0.8523（第 8 epoch 最高，之後過擬合，第 12 epoch 0.8402） | 同上，無法提交。推論推算 0.76 小時 |
| 2 | 2026-10-10 | v2 軟標籤，dropout 0.25 放在 head 前，weight decay 1e-3，backbone lr 1e-4，10 epoch 並平均最後 3 個 epoch 的預測 | 0.8208（最後 3 個 epoch 平均）。每個 epoch 都還在上升，第 10 個是 0.8223，屬於訓練不足 | 無法提交。推論推算 2.69 小時（3 個 checkpoint） |
| 2b | 2026-10-10 | 第 2 輪設定改成 18 epoch，仍用 cache v1。這是第 3 輪的對照組 | 0.8287（最後 3 個 epoch 平均）。第 14 到 17 個 epoch 都在 0.83 左右，已經到高原 | 無法提交，submit kernel 第 6 版，推論推算 0.96 小時 |
| 1b | 2026-10-11 | 第 1 輪的設定（硬標籤 v1、學習率全部 3e-4、weight decay 1e-4、不加 dropout），cache v1，12 epoch 並平均最後 3 個 | **0.8409**。每個 epoch 的分數與第 1 輪完全相同，因為 seed 固定，可重現 | |
| 3（結果） | 2026-10-11 | 見上方第 3 列 | 0.8184，比第 2b 輪低。在軟標籤的設定下，cache v2 沒有幫助 | 無法提交，submit-v2 第 1 版，推論推算 2.07 小時 |
| 1b+2b | 2026-10-11 | 第 1b 輪與第 2b 輪的 ensemble，各 3 個 checkpoint 等權重。兩者都是 cache v1，所以推論只需建一種 cache | **0.8555**，比單獨最好的高 0.015。混合權重在 0.5 到 0.7 之間幾乎一樣（0.855 到 0.857），所以用等權重，不調 | 帳號驗證後的首選。`rsna-knee-submit-ens` 第 1 版已在雲端跑完，6 個 checkpoint 都載入，推論推算最多 1.59 小時。提交仍是 403 |

帳號完成身分驗證後，用這一行提交目前最好的版本。第 3 版是六輪的 ensemble，18 個 checkpoint 都已載入，推論推算最多 2.29 小時。

```bash
conda run --no-capture-output -n rsna_knee python tools/submit_code.py 3 "ensemble of 6 rounds, cache v1" zacheenoao/rsna-knee-submit-ens
```
| 5a | 2026-10-11 | 第 1b 輪的設定，只把標籤換成 v2s | **0.8425**，與第 1b 輪相當。在第 5 和第 8 個 epoch 就達到 0.852 | |
| 1b+2b+5a | 2026-10-11 | 三輪各 3 個 checkpoint 等權重，全部是 cache v1 | **0.8630** | 尚未建推論 kernel |
| 5b | 2026-10-11 | 第 1b 輪的設定，標籤換成完整的 v3 | 0.8390 | |
| 1c | 2026-10-11 | 第 1b 輪的設定與標籤，seed 改成 1 | **0.8019** | |
| 五輪 | 2026-10-11 | 1b、2b、5a、5b、1c 各 3 個 checkpoint 等權重，全部是 cache v1 | **0.8653** | `rsna-knee-submit-ens` 第 2 版 |

**第 1c 輪改變了前面所有比較的解讀。** 它和第 1b 輪只差 seed，分數卻從 0.8409 掉到 0.8019，差了 0.04。所以在 58 筆驗證上，單一模型本身就會有 ±0.02 到 0.04 的波動。前面各輪之間 0.01 到 0.02 的差距，例如 1b 對 2b、cache v1 對 v2，都在這個範圍內，只能算方向，不能算結論。

| 5c | 2026-10-11 | 第 5a 輪的設定，seed 改成 1。增加 seed 數讓 ensemble 更穩 | **0.8524**，目前最好的單一模型。和第 5a 輪只差 seed，又是 0.01 的差距 | |
| 六輪 | 2026-10-11 | 再加上第 5c 輪，共 18 個 checkpoint | **0.8704** | `rsna-knee-submit-ens` 第 3 版 |
| 5d | 2026-10-11 | 第 5b 輪的設定（v3 標籤），seed 改成 1。讓三種標籤都各有兩個 seed | 0.8392。七輪 ensemble 是 0.8704，與六輪相同，再加 seed 已經沒有幫助 | 沒有加進提交 |
| L1 | 2026-10-11 | 本機 GPU。第 1b 輪的設定，v2s 標籤，seed 2 | 0.8418。加進去成為八輪，ensemble 是 0.8689，比六輪的 0.8704 略低但在雜訊內，所以不放進提交 | 沒有上傳 |
| L2、L3 | 2026-10-11 | 本機 GPU，v1 與 v3 標籤，seed 2 | 使用者要求暫停，沒有開始。之後也不再跑，因為同一配方的 seed 已經飽和 | |
| L4 | 2026-10-12 | 本機 GPU。第 1b 輪的設定與 v2s 標籤，pooling 改成 `--pool conv`（切片序列上兩層 1D 卷積，再接 attention pooling） | **0.8559**，目前最好的單一模型。六輪加上 L4 成為第七個等權成員，ensemble 從 0.8704 升到 **0.8748**。只取 1b、5c、L4 三個是 0.8751，差距在雜訊內，所以不挑子集 | checkpoint 上傳為私人 dataset `rsna-knee-local-models`，由 `rsna-knee-submit-ens` 第 4 版掛載，共 21 個 checkpoint |
| L5 | 2026-10-12 | L4 的設定，seed 改成 1。確認 L4 的提升不是 seed 的運氣 | **0.8564**，與 L4 幾乎相同。八個成員等權是 **0.8772**。attention 組與 depth-conv 組的權重在 0.25 到 0.75 之間都落在 0.877 到 0.880，差距在雜訊內，所以維持每個 checkpoint 等權 | 加進 `rsna-knee-local-models`，由 `rsna-knee-submit-ens` 第 5 版掛載，共 24 個 checkpoint |
| L6 | 2026-10-12 | L4 的設定，seed 改成 2 | 0.8562。九個成員是 0.8778，只比八個多 0.0006，depth-conv 的 seed 也已飽和 | 沒有加進提交 |
| L7 | 2026-10-12 | 本機 GPU。L4 的設定，骨幹改成 resnet34，cache v1 | **0.8624**，目前最好的單一模型。每個 epoch 約 15.5 分鐘，是 resnet18 的兩倍 | 還沒上傳 |
| C1 | 2026-10-12 | Colab L4 GPU。L4 的設定，cache v3 的 5 個 slot，seed 0。每個 epoch 452 秒 | 0.8394，比同設定 3 個 slot 的 L4、L5 低約 0.017，但在 seed 雜訊的範圍內 | 還沒上傳 |
| 十成員 | 2026-10-13 | 所有成員在 cache v3 上評估。3 個 slot 的模型只讀前 3 個 slot，所以分數與 cache v1 上相同 | 八個成員 0.8772，加 L7 是 0.8815，再加 C1 是 **0.8842**，再加 L6 反而是 0.8838。C1 單獨較弱卻能拉高 ensemble，表示多出來的兩個序列讓它的錯誤和其他成員不同。前幾名子集約 0.887，差距在雜訊內，照慣例不挑子集 | `rsna-knee-submit-ens` 第 7 版，推論端建 5 個 slot，30 個 checkpoint，預估 3.9 小時。public **0.893**（第 5 版 0.888），和 58 筆 GT 的方向一致 |
| L8 | 2026-10-13 | 本機 GPU。L7 的設定，標籤換成混合標籤 `report_labels_blend.jsonl`（我們的 v2s 加上 7 組 CC0 公開標籤的平均，對 58 筆 GT 0.8923，v2s 是 0.8655） | 0.8571，和 L7 的 0.8629 在雜訊內，所以較好的標籤沒有直接讓單一模型變好。加進十成員後是 0.8877（十成員 0.8842），標籤不同的成員仍能拉高 ensemble | 還沒上傳 |
| L9 | 2026-10-13 | 本機 GPU。L8 的設定改用 cache v3 的 5 個 slot | 第一個 epoch 跑了 71 分鐘，整輪要 6 小時以上，所以中止，改用 GPU 測 WindowNet | 中止 |
| W1 | 2026-10-13 | 本機 GPU。新架構 WindowNet（相鄰 3 張切片疊成 RGB、每個 finding 一組 attention、隨機裁 288），骨幹 convnext_tiny，cache v4（5 個 slot 各 20 張、2% 到 98%、128 mm 物理裁切、320 px），混合標籤，bs 2，lr 1e-3，骨幹 lr 1e-4，wd 0.02。每個 epoch 約 15 分鐘 | 第 9 個 epoch 結束時 Claude Code 重新啟動，訓練被一起結束。第 7 到 9 個 epoch 的平均是 **0.9029**。每個 epoch 的分數從第 1 個的 0.829 一路升到 0.906，舊架構要到第 8 個 epoch 才到 0.85 | 放進 `rsna-knee-window-models` |
| W2 | 2026-10-13 | Colab T4。W1 的設定，骨幹改成 resnet34，骨幹 lr 2e-4。每個 epoch 約 18 分鐘 | **0.9071**，目前最好的單一模型，比舊架構最好的 L7（0.8624）高 0.045 | 同上 |
| W3 | 2026-10-13 | Colab L4。W1 的設定，seed 1。每個 epoch 約 12 分鐘 | **0.9064** | 同上 |
| W 三成員 | 2026-10-13 | W1、W2、W3 等權 | **0.9110**，舊的十成員是 0.8842。把舊的 11 組平均後以 0 到 0.5 的權重混入，分數都在 0.909 到 0.9135 之間，差距在雜訊內，所以不混入，推論只需要 cache v4 | `rsna-knee-submit-w` 第 1 版，ref 57159249，預估推論 4.4 小時。public **0.939**，十個 KneeNet 成員是 0.893 |
| W8 | 2026-10-13 | 本機 GPU。W1 的設定，骨幹改成 efficientnet_b0，骨幹 lr 2e-4。每個 epoch 約 9 分鐘，是三種骨幹裡最快的 | **0.9159**，目前最好的單一模型。四個 W 成員是 0.9171，W2 加 W8 是 0.9207，子集差距在雜訊內 | 放進 `rsna-knee-window-models` |
| W9 | 2026-10-13 | 本機 GPU。W8 的設定，seed 1。第一次執行在第 1 個 epoch 後被 numpy 的 Intel Fortran 程式庫在關閉視窗事件時中止，加上 `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` 後重跑 | 0.9096 | 同上 |
| W 五成員 | 2026-10-13 | W1、W2、W3、W8、W9 等權，15 個 checkpoint | **0.9181**，三成員是 0.9110。加 seed 的提升已經變小 | `rsna-knee-submit-w` 第 2 版，ref 57167561，預估推論 1.5 到 6.7 小時。public **0.942**，三成員是 0.939 |
| P1 | 2026-10-14 | 本機 GPU。W8 的設定，標籤換成偽標籤 `report_labels_pseudo1.jsonl`。偽標籤是混合標籤與 W 五成員對訓練資料預測的各半平均，用 `tools/predict_cache.py` 產生。這些預測是 in-sample 的，作用接近平滑過的標籤。偽標籤本身對 58 筆 GT 是 0.9299，混合標籤 0.8922，五成員預測 0.9181 | 0.9136，W8 是 0.9159，差距在雜訊內。前幾個 epoch 學得較快，第 3 個 epoch 0.883 對 W8 的 0.842。加進五成員是 0.9190 | 放進 `rsna-knee-window-models`，排進排行榜探測 |
| P2 | 2026-10-14 | 本機 GPU。W2 的設定（resnet34），標籤換成同一份偽標籤 | 0.9043，W2 是 0.9071。加進五成員是 0.9161，比五成員的 0.9181 略低。偽標籤在 58 筆 GT 上對兩種骨幹都沒有幫助 | 沒有上傳 |
| W13 | 2026-10-14 | 本機 GPU。W8 的設定改成公開 0.949 模型的訓練方式，梯度累積 8 步讓有效 batch 從 2 變成 16，head lr 2e-3，骨幹 lr 1e-4，wd 0.05，16 個 epoch。兩邊的切片數（約 100）與視野（約 115 mm）已經相同，剩下的差異主要在 batch 與訓練長度 | 0.9022，比 W8 的 0.9159 低。大 batch 與較長的訓練沒有幫助，之後維持 batch 2 的設定 | 沒有上傳 |
| 五裁切 TTA | 2026-10-14 | 不重新訓練，推論時把 volume 上下左右平移 16 px 再取中央 288，與原本的中央裁切共 5 個位置平均 | W8 0.9159 變 0.9167，W9 0.9096 變 0.9070，P1 0.9136 變 0.9169，三者 ensemble 0.9154 變 0.9164。方向不一致，又要 5 倍的推論時間，所以不採用 | |
| W14 | 2026-10-14 | 本機 GPU。W8 的設定，seed 3 | 訓練中，05:20 開始 | |
| W4 | 2026-10-14 | Kaggle T4，`rsna-knee-train-w4`。W8 的設定，標籤換成我們自己的 v2s。每個 epoch 約 11.5 分鐘 | 0.9100。efficientnet 五成員在 58 筆 GT 上是 0.9202（排行榜 0.943），加上 W4 是 0.9208 | 放進 `rsna-knee-window-models` |
| W5 | 2026-10-14 | Kaggle T4，`rsna-knee-train-w5`。W8 的設定，seed 2 | 0.9148 | 同上 |
| W15、W16 | 2026-10-14 | Kaggle T4。W15 是 P1 的設定（偽標籤）seed 1，W16 是 W4 的設定（v2s）seed 1 | W15 0.9119，W16 0.9073 | 放進 `rsna-knee-window-models` |
| W14 結果 | 2026-10-14 | 見上方 W14 列，本機 W8 的設定 seed 3 | 0.9185，目前最好的單一模型。efficientnet 五成員加 W4、W5、W14 是 0.9200，與五成員的 0.9202 相同，58 筆 GT 已經飽和 | 放進 `rsna-knee-window-models` |
| W17、W18 | 2026-10-14 | W8 的設定加 `--resize 384`，把 288 的裁切放大到 384 再進骨幹。W17 在本機 seed 0，W18 在 Colab L4 seed 1 | W18 0.9101，低於 b0 的平均（約 0.915），所以放大輸入沒有幫助。W17 在本機每個 epoch 要 75 分鐘，顯存用到 7.75 GB，第 2 個 epoch 後中止 | W18 沒有上傳 |
| W22 | 2026-10-14 | Kaggle T4，`rsna-knee-train-w22`。W8 的設定，骨幹換成 tf_efficientnetv2_b0，與 b0 大小相近但屬於不同系列。每個 epoch 約 9.6 分鐘 | **0.9161**，與 b0 最好的幾個相當 | 放進 `rsna-knee-window-models` |
| W19 | 2026-10-14 | Colab L4。W8 的設定，骨幹換成 efficientnet_b1 | 第 2、3 個 epoch 是 0.769、0.755，loss 開始上升，第 4 個 epoch 預測變成 NaN，驗證時以 `ValueError: Input contains NaN` 中止。b0 用同一組 lr 沒有發散，b1 應該要更低的 lr。已從 `jobs_A.txt` 註解掉，避免重跑時接續發散的狀態 | 失敗 |
| W23、W24 | 2026-10-14 | Colab L4。tf_efficientnetv2_b0，W23 是 seed 1，W24 改用 v2s 標籤 | W23 0.9132，W24 0.9082 | 還沒上傳 |
| 5-fold OOF | 2026-10-14 | W8 的設定加 `--folds 5 --fold k`。依報告文字的雜湊分成 5 份，同樣的報告一定在同一份。每個模型不看其中一份，在最後 3 個 epoch 對那一份預測並平均，寫成 `kneenet_oof.csv`。fold 0、1 在 Kaggle，fold 3 在 Colab T4，fold 2、4 在本機。Kaggle 這週 45 小時的 GPU 額度在 fold 2 推送時用完 | 5 個 fold 在 58 筆 GT 上是 0.9088、0.9123、0.9044、0.9018、0.9021，因為各少了 20% 的訓練資料。合起來的 OOF 共 4,349 筆，對混合報告標籤（以 0.5 二值化）的 macro AUC 是 **0.8845**，這是第一個用全部報告標籤算出的 CV 指標。OOF 與混合標籤各半寫成 `report_labels_oof1.jsonl`，用它重訓還在等使用者同意 | fold 模型在 `data/models/F0` 到 `F4` |
| W11 | 2026-10-14 | Kaggle T4，`rsna-knee-train-w11`。W8 的設定，骨幹改成 efficientnet_b2。每個 epoch 約 17 分鐘，`/kaggle/tmp` 解壓 cache v4 後還有約 1 TB 可用 | 0.9092，沒有比 b0 好。加進五成員是 0.9191（五成員 0.9181），在雜訊內，所以不再往更大的骨幹試 | 放進 `rsna-knee-window-models` |
| 排行榜探測 | 2026-10-14 | 單一成員各提交一次，用約 400 筆 public test 回答 58 筆 GT 分不出的問題。`tools/probe_submit.sh` 送出。CPU 推論一個成員要 21.9 小時，所以探測要用 GPU | W8 單獨 public **0.938**，P1 單獨 **0.939**，W2（resnet34）單獨 **0.920**，五成員 0.942。三個結論。第一，efficientnet_b0 一個模型就接近整個 ensemble。第二，resnet34 在排行榜上比 b0 低 0.018，在 58 筆 GT 上只低 0.009，58 筆 GT 會低估骨幹的差距。第三，偽標籤在排行榜上與 W8 相同，沒有害處也沒有明顯好處 | 下一個提交改成只用 efficientnet |
| W12 | 2026-10-14 | Kaggle T4，`rsna-knee-train-w12`。W8 的設定，訓練時不再隨機裁成 288，直接用 320 的整個畫面，每步 48 個 window。每個 epoch 約 18 分鐘 | 0.9085，與 W8 在雜訊內。訓練 loss 比 W8 低（0.404 對 0.432），可能較容易過擬合 | 放進 `rsna-knee-window-models` |
| efficientnet 五成員 | 2026-10-14 | W8、W9、P1、W11、W12，15 個 checkpoint，拿掉 resnet34 與 convnext | 58 筆 GT 上 W8、W9、P1、W11 是 0.9173，與原本五成員的 0.9181 相同 | `rsna-knee-submit-probe` 第 5 版，ref 57180317。public **0.943**，目前最好，原本含 resnet34 與 convnext 的五成員是 0.942。之後的 ensemble 以 efficientnet 為主 |
| 第 8 輪調研 | 2026-10-15 | 上網整理本比賽公開 repo 與論文的做法，結論在 `data/rounds/r8/research_web.md`。只看方法，沒有下載任何 weight | 混合標籤在 58 筆 GT 上已經接近校準，只有 Synovitis 在低標籤區仍有 33% 陽性，單調重映射改不了 AUC，所以不做標籤校準。pairwise ranking loss 需要一個 batch 有多筆 study，我們是 batch 2，也不做 | |
| V5A | 2026-10-16 | 本機 GPU（cache 在 Kaggle CPU 上建好後下載到本機，`rsna-knee-train-v5a` 沒有使用）。W8 的設定改用 cache v5，拿掉兩個非 FS 的 slot，把切片預算移到三個 FS slot，每個 32 張，總切片數和 v4 差不多。依據是第三方消融顯示拿掉非 FS 序列不掉分。cache v5 由四個 CPU kernel `rsna-knee-build-cache-v5-s0` 到 `-s3` 建 | 0.9086，同 seed 的 W8 是 0.9159。Lateral OA 0.779、Lateral Meniscus 0.845 掉得最多，非 FS 序列對外側結構仍有用。加進 efficientnet 五成員是 0.9189，五成員本身 0.9202，所以維持 cache v4 | 沒有上傳 |
| P3、P4 | 2026-10-16 | 本機 GPU（`rsna-knee-train-p3` 沒有使用）。W8 的設定，標籤換成 `report_labels_oof1.jsonl`，也就是五折 OOF 預測與混合標籤各半，P3 是 seed 0，P4 是 seed 1。每個 epoch 約 530 秒 | P3 0.9164，P4 0.9095，平均 0.913，與混合標籤的 b0 各 seed 相當。efficientnet 五成員加 P3 再加 W4、W5、W14、W15、W16、W22 是 0.9200，加 P3、P4 是 0.9187，五成員本身 0.9202。58 筆 GT 已經飽和，OOF 偽標籤是否有用只能靠排行榜判斷 | 沒有上傳 |
| W25、W26 | 2026-10-16 | 本機 GPU。替最後的 ensemble 多準備成員。W25 是 W22 的設定（tf_efficientnetv2_b0）seed 2，W26 是 W8 的設定 seed 4 | W25 0.9140，W26 0.9167 | 還沒上傳 |
| W27、W28 | 2026-10-16 | 本機 GPU。W27 是 W8 的設定，骨幹換成 convnext_nano，骨幹 lr 用 W1 的 1e-4，用來測不同系列的骨幹能不能增加 ensemble 多樣性。W28 是 W22 的設定 seed 3 | W27 0.9140（convnext_tiny 的 W1 是 0.903），W28 0.9204，目前最好的單一模型。tf_efficientnetv2_b0 四個（W22、W23、W25、W28）是 0.9209，再加 W27 是 **0.9245**，之前的組合都停在 0.920 附近。efficientnet 五成員加這五個是 0.9238，五成員加 W27 是 0.9210，所以增益主要來自 effv2 與 convnext_nano 的組合 | 放進 `rsna-knee-window-models`，額度重置後探測排行榜 |
| W29 到 W32 | 2026-10-16 | Colab 一次只能成功跑一個 runner，所以只用 runner_A，W31、W32 改到本機，排在 W28 之後。runner_A 跑 W29（b0，OOF 偽標籤 seed 2）再跑 W30（tf_efficientnetv2_b0，OOF 偽標籤 seed 0）。本機跑 W31（b0 seed 5，加 `--train-on-gt`，58 筆 GT 也拿來訓練，所以它的驗證分數是 in-sample，只能用排行榜判斷）再跑 W32（tf_efficientnetv2_b0 seed 4） | W29 0.9213，目前最好的 b0。OOF 偽標籤的三個 b0（P3、P4、W29）平均 0.9157，混合標籤的 b0 各 seed 平均約 0.915，仍在雜訊內。W31 0.9188，但它把 58 筆 GT 拿來訓練，這個分數是 in-sample。W30 在 Colab 第 11 個 epoch 後斷線，改在本機從 resume state 跑完最後一個 epoch，最後 3 個 checkpoint 是 0.9165。W32 0.9122 | 還沒上傳 |
| W33、W34 | 2026-10-16 | W27 的設定（convnext_nano，骨幹 lr 1e-4）換 seed。W33 在 Colab runner_A，seed 1，W34 在本機，seed 2。目的是確認 effv2 加 convnext_nano 在 58 筆 GT 上的 0.9245 不是靠 W27 一個 seed 的運氣 | W33 0.9127，W34 0.9074，W27 是 0.9140。convnext_nano 單獨比 b0 低一點，價值在於多樣性 | 放進 `rsna-knee-window-models` |
| W35 | 2026-10-16 | Colab runner_A。W8 的設定，骨幹換成 regnety_016，大小與 b0 相近、系列不同。排行榜顯示成員越多元越好（10 個成員 0.944），所以再加一個新系列 | 0.9144，與 b0 各 seed 相當 | 還沒上傳 |
| W36、W37 | 2026-10-17 | 本機 GPU。W35 的設定（regnety_016）seed 1 與 2，讓新系列也有三個 seed | W36 0.9014，W37 0.9079，W35 0.9144。regnety_016 三個 seed 平均 0.908，比 b0 低，不放進 ensemble | 放進 `rsna-knee-window-models` |
| 第 9 輪 | 2026-10-17 | Codex 這輪不能用，這個 ChatGPT 帳號對所有模型都回 400 not supported。改問 Antigravity（Gemini 3.8 Flash），回覆在 `data/rounds/r8/agy.md`。採用的是 full-frame 訓練帶來的輸入多樣性，以及停止增加同質 seed。另外把輸入切法不同的 V5A 當作多樣性來源 | |
| W38、W39 | 2026-10-17 | W12 的 full-frame 設定（整張 320 px、每步 48 個 window），W38 是 convnext_nano（本機），W39 是 tf_efficientnetv2_b0（Kaggle，`rsna-knee-train-w39`） | W39 **0.9301**，目前最好的單一模型，之前最好的是 W29 的 0.9213。每個 epoch 約 940 秒。在 58 筆 GT 上加進任何組合都比它單獨低。W38 在本機每個 epoch 要 70 分鐘，第 6 個 epoch 0.9002 時中止 | W39 放進 `rsna-knee-window-models` |
| W40、W41 | 2026-10-17 | W39 的設定換 seed。W40 seed 1 在 Kaggle，W41 seed 2 在本機 | W41 0.9062，同設定的 W39 是 0.9301，只差 seed 就差 0.024，W39 的分數有一部分是運氣。本機每個 epoch 約 750 秒，比 Kaggle T4 快。W40 訓練中 | W41 放進 `rsna-knee-window-models` |
| v45 | 2026-10-17 | `rsna-knee-submit-v45`。0.944 的 10 個成員在 cache v4 上，V5A 在 cache v5 上，兩份預測以 0.85 比 0.15 合併 | public 0.944（ref 57330296），與 10 個成員相同，V5A 沒有幫助 | `rsna-knee-submit-v45` 第 1 版 |
| 第 8 輪排行榜探測 | 2026-10-16 | 額度重置後提交四次。efficientnet 五成員加 KneeNet 十成員（0.2 的比重），effv2 四個（W22、W23、W25、W28）加 W27，efficientnet 五成員加這五個共 10 個成員，以及 P3 單獨 | 10 個成員 public **0.944**，目前最好（ref 57288977）。effv2 加 W27 是 0.942，在 58 筆 GT 上卻是最高的 0.9245，所以 GT 的排序不可靠，真正有效的是兩組合在一起的多樣性。KneeNet 混入是 0.942，比 0.943 略低，不再使用。P3 單獨 0.937，W8 單獨 0.938、P1 0.939，OOF 偽標籤沒有幫助 | `rsna-knee-submit-probe` 第 6 到 8 版，`rsna-knee-submit-mix` 第 1 版 |
| 16 成員探測 | 2026-10-17 | 0.944 的 10 個成員再加 b0 的 W5、W14、W26，effv2 的 W30，convnext_nano 的 W33、W34，共 48 個 checkpoint。commit run 約 3 小時，以前約 5 分鐘，推算最多 15.39 小時、最少 3.78 小時 | public 0.944，與 10 個成員相同（ref 57303091），實際計分約 4 小時跑完。加成員在排行榜上已經飽和 | `rsna-knee-submit-probe` 第 9 版 |
| W20 | 2026-10-15 | Colab L4。W8 的設定，epoch 從 12 拉長到 18。中途斷線一次，從第 13 個 epoch 之後用 `--resume` 接回 | 0.9124，落在 b0 各 seed 的 0.910 到 0.919 之間。第 15 個 epoch 單點 0.9175，之後回落，拉長訓練沒有幫助，之後維持 12 epoch | 還沒上傳 |
| W21 | 2026-10-16 | Colab。W8 的設定，訓練時每步編碼的 window 從 32 增加到 48 個（每個 study 共約 90 個）。每個 epoch 約 600 秒 | 0.9051，比 b0 各 seed 的 0.910 到 0.919 低。訓練時多看 window 沒有幫助，推論時本來就用全部 window | 還沒上傳 |

**第一次拿到排行榜分數（2026-10-13）。** 完成身分驗證後，`rsna-knee-submit-ens` 第 3 版（六個 attention 成員）public 是 0.882，第 5 版（再加 L4 與 L5）是 **0.888**。兩者在 58 筆 GT 上的差距方向相同，所以 58 筆 GT 可以當作排行榜的參考。排行榜第 1 名是 0.966，第 29 名是 0.961，差距約 0.07。

討論區的高分做法顯示，差距主要來自輸入的建構，詳見 `data/rounds/r7/research.md`。每個 study 平均有 5.5 個序列，我們每個切面只取一個 fluid-sensitive 序列。cache v3 因此多加兩個非 fluid-sensitive 的 slot（矢狀面與冠狀面），其餘設定與 cache v1 相同，一次只改一個變數。前 3 個 slot 與 cache v1 逐位元相同，所以舊的 checkpoint 可以直接讀 5 個 slot 的影像，新舊成員可以放在同一個 ensemble。建置用 `rsna-knee-build-cache-v3-s0` 與 `-s1` 兩個 CPU kernel，設定在 `kaggle_kernels/common/v3_env.py`。

L4 是第一個改變架構的成員。之前所有成員都用 attention pooling，它不看切片順序，這是所有成員共同的盲點。新的 pooling 把最後一層卷積初始化為 0，所以剛開始時與舊版完全相同，訓練後才開始利用切片順序。判斷它有沒有用，要看加進 ensemble 後的分數，不看它單獨的分數。

本機實測每個 epoch 482 秒，與 Kaggle T4 相當，比前面以隨機張量量到的 4.9 分鐘慢。量測當下 GPU 使用率是 0%，所以瓶頸在 Windows 上的資料讀取，不在 GPU 運算。

**2026-10-11 16:10 暫停。** 目前可提交的最佳版本仍是 `rsna-knee-submit-ens` 第 3 版（六輪，驗證 0.8704）。恢復時可以接著做的事有兩件。第一，L1 跑完後用 `tools/eval_ensemble.py` 加進去評估，若有幫助，要先把 checkpoint 上傳成 Kaggle dataset 才能放進提交。第二，七輪已經飽和，再加 seed 不划算，下一步應該換方向，例如換架構或改進弱標籤，而不是再加 seed。

Kaggle 這週的 GPU 額度約剩 5 小時，所以後續成員改在本機訓練。本機以第 1b 輪的設定，每個 epoch 約 4.9 分鐘（Kaggle T4 約 8 分鐘），記憶體 3.4 GiB。cache v1 先用 `tools/fetch_cache.sh` 下載。它會不斷換新的簽名網址續傳，直到檔案大小與伺服器回報的一致，因為伺服器會中途斷線，curl 自己的重試次數在 10 GB 的檔案上會用完。訓練佇列如下。

```bash
bash tools/train_local_queue.sh "L1 report_labels_v2s.jsonl 2" "L2 report_labels.jsonl 2" "L3 report_labels_v3.jsonl 2"
```

本機訓練出來的成員要加進提交，必須先把 checkpoint 上傳成 Kaggle dataset，再掛到 `submit_ens`，因為提交只能在 Kaggle notebook 上跑。

ensemble 不受這個問題影響，五個成員一路加上去，分數穩定往上走（0.8555、0.8630、0.8653）。前幾名組合之間只差 0.002，所以最後用五個全部，不挑分數最高的子集，以免挑到的只是運氣。

第 5a 輪回答了第 2b 輪留下的問題。在同一組超參數下，軟標籤與硬標籤分數相當，所以第 2b 輪變差是正則化與較低學習率造成的，不是軟標籤造成的。另一個規律是組合的成員越不一樣，ensemble 越好。第 5b 輪換標籤、第 1c 輪換 seed，就是沿著這兩個方向再加成員。

到這裡為止最重要的發現有兩個。第一，報告標籤和放射科醫師比起來比較準（v2），訓練出來的模型反而比較差。第 2b 輪同時改了標籤和正則化，所以第 5a 輪只換標籤，用來分辨是哪一個造成的。第二，硬標籤模型和軟標籤模型的錯誤不一樣，合起來比任何一個單獨都好。加上第 3 輪（cache v2）則沒有再進步。

ensemble 的評估在本機進行，不花 Kaggle 額度。驗證用的 cache 由 `rsna-knee-export-val-cache` 從雲端 cache 只取出 58 筆，約 600 MB，checkpoint 從各個 train kernel 下載。

```bash
conda run --no-capture-output -n rsna_knee python tools/eval_ensemble.py r1b="data/models/r1b/*.pt"@"data/val_cache/v1/train_s0of1" r2b="data/models/r2b/*.pt"@"data/val_cache/v1/train_s0of1"
```

完整 cache 下載到本機一直失敗。伺服器會中途斷線（schannel `server closed abruptly`），5.5 小時只下載了 4.1 GB，所以本機訓練暫時做不到。

第 1 輪的 checkpoint 已經被後面幾輪覆蓋，而且它的 0.8523 是從單一 epoch 挑出來的，不能和其他輪比。第 1b 輪用同樣的平均方式重跑，才能和第 2b 輪公平比較，也能讓 ensemble 有一個硬標籤的成員。
| 4a | 2026-10-11 | resnet34，cache v2，10 epoch，在本機 GPU 訓練。目的是讓最後的 ensemble 有不同架構 | 沒有執行。cache 下載到本機一直失敗，而且第 3 輪顯示 cache v2 沒有比較好，所以取消 | |
| 3 | 2026-10-10 | cache v2。先裁到前景再補成正方形，224×224，每個切面 48 張。第 2 輪設定，18 epoch | 訓練中 | |

第 2 輪的分數低於第 1 輪，但兩者不能直接比。第 1 輪的 0.8523 是從 12 個 epoch 裡挑最高的那一個，本身就偏樂觀。第 2 輪則是平均最後 3 個 epoch，而且明顯還沒訓練完。兩位顧問都判斷第 2 輪是訓練不足，不是設定錯誤，所以第 2b 輪與第 3 輪都沿用同一組設定，只把 epoch 拉長到 18。

cache v2 實測 4,407 個 study 全部完成，沒有任何 `[warn]`，兩個分片各 10.61 GB。本機 60 個 series 的量測顯示，背景強度接近 0，裁切框的中位數佔原圖面積的 89%，其中 25% 的 series 幾乎沒有被裁到。所以第 3 輪的改善主要會來自補成正方形、提高解析度和增加切片數。

第 3 輪用另一組 kernel，`rsna-knee-build-cache-v2-s0`、`-s1`、`rsna-knee-train-v2`、`rsna-knee-submit-v2`，所以可以和第 2 輪同時進行。cache v2 壓縮後約 21 GB，超過單一 kernel 20 GB 的輸出上限，所以分成兩個分片。新的 cache 設定靠環境變數 `RSNA_CROP_FG`、`RSNA_IMG_SIZE`、`RSNA_N_SLICES` 開啟，三個 v2 kernel 的 `body.py` 都寫死了同一組值，checkpoint 也會記錄 `crop_fg`，設定不一致時推論會直接拒絕。預設值產生的 cache 與先前逐位元組相同。

```bash
bash tools/chain_submit.sh "r3 cache v2" rsna-knee-train-v2 submit_v2
```

報告標籤 prompt 的比較，都在 B 層看得到的 29 筆上量。

| prompt | 日期 | 輸出 | macro AUC |
|---|---|---|---|
| v1 | 2026-10-09 | 1、0、-1 | 0.7918 |
| v2 | 2026-10-10 | 0 到 1 的機率，依描述的嚴重程度分級，-1 代表沒提到 | 0.8767 |

| v3 | 2026-10-11 | v2 再加上 Synovitis、Lateral Meniscus、MCL、PF OA 四個標籤的細則 | 0.8827 |
| v2s | 2026-10-11 | v2，只有 Synovitis 改用 v3 的值 | 0.8870 |

v3 只讓 Synovitis 明顯變好（0.691 升到 0.814，有給值的比例從 38% 升到 100%），PF OA 反而從 0.958 降到 0.907。所以最後用 v2s，也就是只採用有機制可以解釋的那一欄。

v3 全量標完後，在藏起來的 29 筆上各量一次。v2 是 0.8065，v3 是 0.8285，v2s 是 0.8262。兩者都比 v2 高約 0.02，代表改善不是在調整用的那一半上過度擬合出來的。v3 在這裡略高於 v2s，但差距在雜訊內，而且選 v2s 是在看到這組數字之前就決定的，事後改選等於用測試資料挑結果，所以維持 v2s。合併指令如下。

```bash
conda run -n rsna_knee python src/report_labels.py --merge data/work/report_labels_v2.jsonl data/work/report_labels_v3.jsonl --merge-labels Synovitis --out data/work/report_labels_v2s.jsonl
```

v2 全量標完後，在調 prompt 時從沒看過的 B 層藏起來的 29 筆上量一次，作為 B 層人工對照組的最終數字。v1 是 0.7471，v2 是 0.8065。看得到的 29 筆上的進步（0.085）比這裡（0.059）大，差距就是在調整用的那一半上過度擬合的部分。
