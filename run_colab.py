#!/usr/bin/env python3
"""Colab runner: train the Tetris CNN agent and push checkpoints to GitHub.

Runs `tetris_cnn.py train` and, every PUSH_EVERY seconds while training is
alive, commits + pushes `tetris_cnn.pt` to the repo's origin. Designed so the
whole Colab cell is a single line (nothing multi-line to mis-paste):

    !cd /content/Tetris && python run_colab.py

Assumes git identity + a tokened `origin` remote are already configured in the
Colab session (see COLAB.md, cell 3). Env overrides (all optional):
    EPISODES=5000  BATCH_SIZE=1024  SAVE_EVERY=50  PUSH_EVERY=600
"""
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.abspath(__file__))
CKPT = "tetris_cnn.pt"
EPISODES = os.environ.get("EPISODES", "5000")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "1024")
SAVE_EVERY = os.environ.get("SAVE_EVERY", "50")
PUSH_EVERY = int(os.environ.get("PUSH_EVERY", "600"))


def git_push(msg):
    """Commit + push the checkpoint. Commit is a harmless no-op when the file
    hasn't changed; we print the push exit code either way."""
    subprocess.run(["git", "add", CKPT], cwd=REPO)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=REPO)
    r = subprocess.run(["git", "push", "-q"], cwd=REPO)
    print(f"[run_colab] push ({msg}) exit={r.returncode}", flush=True)


def main():
    train = subprocess.Popen(
        [sys.executable, "tetris_cnn.py", "train",
         "--episodes", EPISODES,
         "--device", "cuda",
         "--batch-size", BATCH_SIZE,
         "--resume", CKPT,
         "--save-every", SAVE_EVERY],
        cwd=REPO)

    def push_loop():
        while train.poll() is None:
            time.sleep(PUSH_EVERY)
            if train.poll() is None:
                git_push("checkpoint")

    threading.Thread(target=push_loop, daemon=True).start()
    train.wait()
    git_push("checkpoint-final")


if __name__ == "__main__":
    main()
