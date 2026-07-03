# Training on Google Colab (with checkpoints pushed to GitHub)

`tetris_cnn.py` auto-detects CUDA, full-state checkpoints to `tetris_cnn.pt`
every `--save-every` episodes, and resumes with `--resume`. Colab sessions are
ephemeral, so the workflow below pushes the checkpoint back to this repo
periodically — a disconnect costs at most ~10 minutes of progress, and a fresh
session just re-clones and resumes.

## 1. Enable the GPU

`Runtime → Change runtime type → GPU (T4)`.

## 2. Clone (repo is public, so no token needed to read)

```python
REPO = "Tetris"
!git clone https://github.com/Aaronc123814/Tetris.git /content/$REPO
%cd /content/$REPO
```

## 3. Configure git identity + a push token

Pushing the checkpoint back needs a Personal Access Token (`repo` scope, or a
fine-grained token with read+write **Contents** on this repo). Create one at
GitHub → Settings → Developer settings.

```python
from getpass import getpass
USER  = "Aaronc123814"
TOKEN = getpass("GitHub personal access token: ")
!git config user.email "aaronchacko98@gmail.com"
!git config user.name  "{USER}"
!git remote set-url origin https://{USER}:{TOKEN}@github.com/{USER}/Tetris.git
```

## 4. Dependencies (usually a no-op on Colab)

```python
!pip -q install numpy torch
```

## 5. Train + auto-push the checkpoint every 10 minutes

```python
import subprocess, time, threading

repo = "/content/Tetris"
train = subprocess.Popen(
    ["python", "tetris_cnn.py", "train",
     "--episodes", "10000",
     "--device", "cuda",
     "--batch-size", "1024",        # T4 has the VRAM; cuts gradient noise
     "--resume", "tetris_cnn.pt",   # picks up where the last session left off
     "--save-every", "50"],
    cwd=repo)

def push_loop():
    while train.poll() is None:
        time.sleep(600)  # every 10 minutes
        subprocess.run(["git", "add", "tetris_cnn.pt"], cwd=repo)
        # commit is a no-op (nonzero exit) when nothing changed — that's fine
        subprocess.run(["git", "commit", "-q", "-m", "checkpoint"], cwd=repo)
        subprocess.run(["git", "push", "-q"], cwd=repo)

threading.Thread(target=push_loop, daemon=True).start()
train.wait()

# final push once training finishes
subprocess.run(["git", "add", "tetris_cnn.pt"], cwd=repo)
subprocess.run(["git", "commit", "-q", "-m", "checkpoint (final)"], cwd=repo)
subprocess.run(["git", "push", "-q"], cwd=repo)
```

## Resuming after a disconnect

Start a fresh Colab, re-run cells 2–5. Cell 2 re-clones the latest
`tetris_cnn.pt`, and `--resume` restores full state. `--episodes` is the *total*
budget (resume counts toward it), so training continues seamlessly.

## Notes

- **Checkpoint size / repo bloat.** With the replay buffer saved, `tetris_cnn.pt`
  is ~20 MB and every push adds a new copy to git history. Add `--no-save-buffer`
  to the train command for tiny checkpoints (resume starts with an empty buffer,
  costing a short re-warmup). Otherwise, `git gc` / squash history later.
- **Colab limits.** Idle disconnect ~90 min, max session ~12 h. The auto-push
  caps lost progress at ~10 min.
- **Never commit your PAT** into the repo.
