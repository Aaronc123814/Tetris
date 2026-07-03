# Tetris CNN Agent

A DQN-style Tetris agent that learns a board evaluator with a small **CNN
over the afterstate board**, instead of hand-crafted features. It keeps the
afterstate Q-learning trick (the thing that makes Tetris RL tractable) and
swaps in a convolutional value network.

## Files

| File | Purpose |
|---|---|
| `tetris_cnn.py` | The agent: env wrapper, CNN, training, and play. Main entry point. |
| `tetris_rl.py` | Tetris environment (`Tetris`, `TETROMINOES`). **Required** — `tetris_cnn.py` imports it. |
| `tetris_cnn.pt` | Saved weights (best 100-episode rolling-average checkpoint). |

## Quick start

```bash
pip install torch numpy

# Train (GPU strongly recommended — see "Training cost" below)
python tetris_cnn.py train

# Watch the trained agent (ASCII, animates in place)
python tetris_cnn.py play
python tetris_cnn.py play --no-lookahead
```

## How it works

The **afterstate** formulation collapses Tetris's "which action sequence?"
problem into "which resulting board?", so the network only has to learn a
board evaluator `V(s')`:

1. When a piece spawns, enumerate every legal final placement (rotation ×
   column), ~30–40 options.
2. For each, simulate the drop and clear completed lines → the *afterstate*.
3. Build the input maps for each afterstate (`--features rich`, default):
   - channel 0: post-clear afterstate board (20×10 binary)
   - channel 1: the just-placed piece's cells, shifted by the same line
     clears so it lines up with channel 0
   - channel 2: column envelope (filled from each column's top cell down) —
     a direct height map
   - channel 3: holes (empty cells with a filled cell somewhere above)

   Channels 2–3 are the long-range *vertical* signals the 3×3 conv stack
   (7×7 receptive field) can't compute from raw cells but that dominate
   board quality. `--features basic` drops them for the original 2-channel
   input.
4. A CNN estimates the value of each afterstate.
5. The agent picks the highest-value placement (ε-greedy during training).
6. Train with **Double DQN**: the online net selects the best next
   afterstate, the target net supplies its value — this curbs the
   max-operator overestimation that plain `target.max()` suffers from.
   Target network synced every 10 episodes.

## Network

Three conv layers (C→32→64→64, 3×3, padding 1) with **GroupNorm** + ReLU,
then a head over the conv activations. `C` is the input-channel count
(4 for `rich`, 2 for `basic`).

The head has two variants (`--head`):

| `--head` | head | params | notes |
|---|---|---|---|
| `pool` (default) | `MaxPool2d(2)` → `Linear(64·10·5 → 256) → ReLU → Linear(256 → 1)` | **~0.88M** | 2×2 pool halves 20×10→10×5 before the flatten |
| `wide` | `Linear(64·20·10 → 256) → ReLU → Linear(256 → 1)` | ~3.33M | the original; first Linear is ~98% of the net |

The `wide` head's first Linear held ~3.28M weights and overfit the replay
buffer — a ~4.6k-param Dellacherie MLP outscores it. `pool` cuts the head 75%
with a **2×2 pool, not global average pooling**: vertical position (height) is
the dominant Tetris signal, so GAP would discard the very thing the value
depends on, whereas 2×2 pooling keeps coarse position. Pair with `--weight-decay`
(Adam L2, default `1e-4`) regularizing the head. Use `--head wide
--weight-decay 0` to reproduce the original net; sweep the two flags apart to
attribute each effect.

GroupNorm rather than BatchNorm on purpose: DQN *acts* on tiny per-piece
placement batches (~34) but *learns* on a 512 batch; BatchNorm's
batch-dependent statistics are unstable across that mismatch, while
GroupNorm is batch-independent and behaves identically acting or learning.

## Key hyperparameters (defaults)

| Param | Value |
|---|---|
| input | `rich` = 4 channels, 20×10 (`--features basic` for 2) |
| batch size | 512 |
| buffer size | 30,000 (persisted in the checkpoint; resume restores it) |
| learning rate | 2.5e-4 (Adam) |
| weight decay | 1e-4 (Adam L2; `--weight-decay 0` to disable) |
| head | `pool` (2×2 pool, ~0.88M; `--head wide` for the ~3.33M original) |
| γ (discount) | 0.99 |
| ε schedule | 1.0 → 0.01 over 2000 episodes (linear) |
| target sync | every 10 episodes |
| train frequency | one gradient step per 4 pieces |
| conv norm | GroupNorm (`--norm batch\|none` to A/B test) |
| train lookahead | off (`--train-lookahead` to match play's 2-ply policy) |
| episodes | 10,000 (default; stop when `roll100` plateaus) |

## Reward shaping

```
reward = 1.0                 # survival bonus per piece
       + 10 · (lines)²       # multi-line clears strongly rewarded
       − 5.0                 # one-time penalty on game over
```

## Two-ply lookahead (play time, no retrain)

At play/eval time, `select(lookahead=True)` does a deterministic 2-ply
search: for each placement of the current piece it enumerates the next
piece's placements on the resulting board and scores by the best leaf
value (one batched forward pass). The 7-bag randomizer means the next
piece is known except at bag boundaries, where it falls back to 1-ply.
`play` uses lookahead by default; pass `--no-lookahead` to disable.

## CLI reference

```
python tetris_cnn.py train [--episodes N] [--batch-size N] [--device D]
                           [--lr LR] [--resume PATH] [--save-path PATH]
                           [--save-every N] [--norm group|batch|none]
                           [--features basic|rich] [--head pool|wide]
                           [--weight-decay F] [--train-lookahead]
                           [--eps-end F] [--eps-decay-steps N]
                           [--no-save-buffer]
python tetris_cnn.py play  [--weights PATH] [--device D] [--no-lookahead]
```

`--device` accepts `cuda | mps | cpu`; default auto-detects **cuda → mps →
cpu**. `--episodes` is the *total* budget — a resumed run counts toward it,
not on top of it. The checkpoint at `--save-path` is written every
`--save-every` episodes (the resume point) and on every new rolling-average
best; it stores the full training state (weights, optimizer, ε, history) for
resuming plus the best weights for `play`. Safe to Ctrl-C anytime.

## Training cost — use a GPU

This config is heavy: each gradient step re-enumerates ~512×34 next-state
boards and runs them through two 3.3M-param nets (Double DQN). On a GPU a
full run is a matter of hours; on a CPU laptop it is **not practical**
(benchmarked at ~0.5 pieces/sec → weeks to months). Train on Colab or
Kaggle.

### Google Colab

1. Runtime → Change runtime type → GPU.
2. Upload `tetris_cnn.py` **and** `tetris_rl.py` to `/content/`.
3. (Optional but recommended) mount Drive so checkpoints survive disconnects.
4. Run via the Python API (avoids shell line-wrapping issues):

```python
import tetris_cnn
tetris_cnn.train(batch_size=1024, save_path='/content/drive/MyDrive/tetris_cnn.pt')
```

### Resuming across sessions (Kaggle's 12h limit, Colab disconnects)

A full 10,000-episode run won't finish in one Kaggle/Colab session. Training
saves its **complete state** (weights, optimizer, ε, episode count, history)
every `save_every` episodes, so you can stop and continue without losing
progress or resetting exploration. Run the **same call each session** with
`resume` and `save_path` pointing at the same persistent file:

```python
import tetris_cnn
# Run this identical cell every session. First time it starts fresh;
# afterwards it picks up exactly where it left off and stops at 10,000.
tetris_cnn.train(num_episodes=10000, batch_size=1024,
                 resume=True, save_path='/kaggle/working/tetris_cnn.pt')
```

On Kaggle, point the path at `/kaggle/working/` and **Save Version (Commit)**
so the file persists as notebook output; next session, add that output as an
input dataset and set `resume='/kaggle/input/<that>/tetris_cnn.pt'` with
`save_path='/kaggle/working/tetris_cnn.pt'`. On Colab, point both at the same
Drive path. `resume=True` is shorthand for "resume from `save_path` if it
exists, else start fresh."

### Watching progress

Watch the `roll100` (100-episode rolling average lines) column in the log —
that's the signal that matters; individual games swing wildly. Stop when it
plateaus rather than waiting for episode 10,000.

## Status

Work in progress. The goal is to beat the original Dellacherie-feature MLP
baseline (~150 lines / 100-ep rolling average). The first CNN plateaued
around 125 lines.

The larger reworked network (3 conv + GroupNorm + 256 head, Double DQN)
initially *regressed* to ~15 lines. Root cause: action selection scored
candidates as `r + γ·V(σ)` while the value head is trained so V already
includes the arrival reward (`V(σ) ≈ r + γ·max V(next)`) — double-counting
the immediate reward. Selection now ranks by `V(σ)` directly, matching both
the training target and the 150-line agent in `tetris_rl.py`.

Regularization push (the plateau looks like overfitting, not under-capacity —
a ~4.6k-param Dellacherie MLP beats the 3.3M CNN that's *handed the same
height/holes signals*): the value head now defaults to **`pool`** (2×2 pool
before the flatten, ~0.88M params, −75% on the head that held ~98% of the net)
plus **Adam weight decay 1e-4**. Both are independent flags (`--head`,
`--weight-decay`) so each effect can be attributed; `--head wide
--weight-decay 0` reproduces the prior net. Watch `roll100` and `V̄`.

Quality push (compute is the cheap axis): the input is now the **`rich`
4-channel** set by default — adding a column-height map and a holes map, the
long-range vertical signals the small-receptive-field conv can't compute
itself and that dominate board quality. Also: lr 2.5e-4, ε decay over 2000
eps, buffer persistence, optional `--train-lookahead` (2-ply behavior policy
matching play), and `--norm none|batch` / `--features basic` to A/B each
choice against the defaults. Retraining is in progress.
