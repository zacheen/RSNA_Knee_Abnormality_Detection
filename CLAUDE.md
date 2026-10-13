# Project instructions

## Git in this repo

This project overrides the "Staging and committing" section of the user-level CLAUDE.md. That section does not apply here.

- Commit directly after every update, without asking first. Do not stop at staging and do not hand over a commit message for the user to commit.
- Skip the GitKraken template, the change file and the commit-msg guard. They exist to let the user commit by hand, which this repo does not do.
- Open branches freely. Prefer one branch per experiment or feature, such as `exp/cache-v3-slots`, and merge back into `main` once the result is known and recorded in RUNBOOK.md section 8.
- Stage by naming paths. Never use `git add -A` or `git add .`, so data or local-only files cannot slip in even if `.gitignore` misses them.
- The repo has no remote. Never add one or push without asking, because pushing publishes and the privacy rules in the user-level CLAUDE.md still apply.
- Everything else in the user-level CLAUDE.md still applies, including the writing style for commit messages and the rules for `Known_concern.md`, which stays untracked.

## One source of truth

This folder is the only place code is edited. Kaggle and Colab are deployment targets that receive copies of `src/`.

- `tools/push_assets.sh` uploads `src/` to the Kaggle dataset `rsna-knee-assets`.
- `tools/sync_colab.sh` copies `src/` to the Drive folder `rsna_knee` that `colab/runner.ipynb` reads.
- Both write a `VERSION` file holding the current commit hash, and `train.py` stores it in every checkpoint, so any model traces back to the commit that trained it. Commit before pushing or syncing, otherwise the hash names code that differs from what ran.
- Never edit code inside `kaggle_datasets/` or on Drive. A fix made there is lost at the next push or sync.
