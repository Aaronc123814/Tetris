# Training on Google Colab (with checkpoints pushed to GitHub)

`tetris_cnn.py` auto-detects CUDA, full-state checkpoints to `tetris_cnn.pt`
every `--save-every` episodes, and resumes with `--resume`. Colab sessions are
ephemeral, so `run_colab.py` runs the trainer and pushes the checkpoint back to
this repo every 10 minutes — a disconnect costs at most ~10 minutes of progress,
and a fresh session just re-clones and resumes.

Everything below is single-line cells: paste-safe, nothing multi-line to mangle.

## 1. Enable the GPU

`Runtime → Change runtime type → GPU (T4)`.

## 2. Clone (public repo — no token needed to read; brings the latest checkpoint)

```python
!git clone https://github.com/Aaronc123814/Tetris.git /content/Tetris
```

## 3. Configure git identity + a push token

Pushing the checkpoint back needs a Personal Access Token (`repo` scope, or a
fine-grained token with read+write **Contents** on this repo). Create one at
GitHub → Settings → Developer settings.

```python
from getpass import getpass
USER  = "Aaronc123814"
TOKEN = getpass("GitHub personal access token: ")
!cd /content/Tetris && git config user.email "aaronchacko98@gmail.com" && git config user.name "{USER}" && git remote set-url origin https://{USER}:{TOKEN}@github.com/{USER}/Tetris.git
```

Torch and numpy are preinstalled on Colab, so no separate install cell is needed.

## 4. Train + auto-push the checkpoint

```python
!cd /content/Tetris && python run_colab.py
```

`run_colab.py` runs `tetris_cnn.py train --resume tetris_cnn.pt` and, every 10
minutes while training is alive, commits + pushes `tetris_cnn.pt`. You'll see
training logs scroll and `[run_colab] push (checkpoint) exit=0` lines every 10
minutes.

**Options** (env vars on the same line):

```python
# push every 2 min + checkpoint every 10 episodes — verifies pushing quickly
!cd /content/Tetris && PUSH_EVERY=120 SAVE_EVERY=10 python run_colab.py

# longer run / bigger batches
!cd /content/Tetris && EPISODES=20000 BATCH_SIZE=2048 python run_colab.py
```

Defaults: `EPISODES=5000  BATCH_SIZE=1024  SAVE_EVERY=50  PUSH_EVERY=600`.

## Resuming after a disconnect

The VM is wiped on disconnect, so re-clone (no `git pull` needed — the clone is
current). Re-run **cells 2, 3, 4**:

```python
!git clone https://github.com/Aaronc123814/Tetris.git /content/Tetris
# then cell 3 (token) and cell 4 (python run_colab.py)
```

`--resume tetris_cnn.pt` restores full state and `--episodes` is the *total*
budget (resume counts toward it), so training continues seamlessly. Look for
`Resumed from tetris_cnn.pt: episode N/5000` in the output.

## Watching the trained agent

`play` needs a display, so run it **locally**, not on Colab, after pulling the
latest checkpoint:

```bash
git pull
python tetris_cnn.py play
```

## Notes

- **Checkpoint size / repo bloat.** With the replay buffer saved, `tetris_cnn.pt`
  is ~20 MB and every push adds a new copy to git history. Add `--no-save-buffer`
  to the `tetris_cnn.py train` call in `run_colab.py` for tiny checkpoints (resume
  starts with an empty buffer, costing a short re-warmup). Otherwise, `git gc` /
  squash history later.
- **Architecture flags must match on resume.** A checkpoint records its
  `--features` / `--head` / `--norm`; resuming requires the same values (the code
  errors clearly if they differ). The current `tetris_cnn.pt` uses the defaults
  (`rich` / `pool` / `group`), so no extra flags are needed.
- **Colab limits.** Idle disconnect ~90 min, max session ~12 h. The auto-push
  caps lost progress at ~10 min.
- **Never commit your PAT** into the repo.
