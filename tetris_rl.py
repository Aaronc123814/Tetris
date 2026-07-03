"""
Tetris reinforcement learning agent (DQN with afterstates).

Approach
--------
Instead of learning frame-by-frame actions (which is brutal for RL — sparse rewards,
huge action sequences per line clear), we use the "afterstate" formulation:

  1. For each falling piece, enumerate every legal final placement
     (rotation x column). That's ~30-40 options per piece.
  2. For each placement, simulate the result: drop the piece, clear lines,
     get the resulting board.
  3. A small neural network maps board-features -> value.
  4. Pick the placement whose afterstate has the highest value.
  5. Train with Q-learning where Q(s, a) collapses to V(afterstate(s, a)),
     because the dynamics from afterstate -> next state are stochastic
     only in which piece arrives next.

This converges in a few hundred episodes on a laptop CPU.

Features (Dellacherie set, per placement)
-----------------------------------------
  - landing_height:      height at which the placed piece came to rest
                         (midpoint of the piece, measured from the floor)
  - eroded_piece_cells:  lines_cleared * (# of the placed piece's own cells
                         that were part of the cleared rows) — rewards
                         clears that actually use the piece efficiently
  - row_transitions:     horizontal filled<->empty flips (walls count as
                         filled); penalizes jagged rows
  - column_transitions:  vertical filled<->empty flips (floor counts as
                         filled); penalizes overhangs
  - holes:               empty cells with at least one filled cell above
  - cumulative_wells:    sum of triangular well depths — this is what lets
                         the agent keep a Tetris well open

This is the classic Dellacherie feature set, famous for near-perfect play
even with a linear model; here a small MLP learns the weighting.

Run
---
  python tetris_rl.py train       # train and save weights to tetris_agent.pt
  python tetris_rl.py play        # watch the trained agent play one game
"""

import sys
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Tetromino shapes. Each piece has 1-4 rotations; each rotation is a list of
# (row, col) offsets relative to the piece's bounding-box top-left.
# ---------------------------------------------------------------------------
TETROMINOES = {
    'I': [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    'O': [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    'T': [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
    ],
    'S': [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    'Z': [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    'J': [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    'L': [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}
PIECES = list(TETROMINOES.keys())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Tetris:
    """Tetris board with afterstate access. Uses a 7-bag randomizer."""

    HEIGHT = 20
    WIDTH = 10
    # [landing_height, eroded_piece_cells, row_transitions,
    #  column_transitions, holes, cumulative_wells]
    FEATURE_DIM = 6

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int8)
        self.bag = []
        self.lines_total = 0
        self.pieces_total = 0
        self.game_over = False
        self.current_piece = self._draw_piece()
        return self.board.copy()

    def _draw_piece(self):
        if not self.bag:
            self.bag = list(PIECES)
            random.shuffle(self.bag)
        return self.bag.pop()

    def peek_next_piece(self):
        """The piece that will spawn after the current one, or None at a
        7-bag boundary where it isn't determined yet. `_draw_piece` pops
        from the end of the bag, so the next piece is bag[-1]."""
        return self.bag[-1] if self.bag else None

    # -- placement enumeration --------------------------------------------
    def get_possible_placements(self, piece=None, board=None):
        """
        Return dict mapping (rotation, x) ->
            (next_board, lines_cleared, landing_height, eroded_piece_cells).
        x is the leftmost column of the piece in the resulting board.
        landing_height and eroded_piece_cells are move-dependent and can't be
        recovered from the afterstate board alone, so we compute them here.
        """
        if piece is None:
            piece = self.current_piece
        if board is None:
            board = self.board

        placements = {}
        rotations = TETROMINOES[piece]
        for rot_idx, shape in enumerate(rotations):
            max_col = max(c for _, c in shape)
            for x in range(self.WIDTH - max_col):
                cells = [(r, c + x) for r, c in shape]
                y = self._drop_y(cells, board)
                if y is None:
                    continue
                placed = [(r + y, c) for r, c in cells]
                # If any cell lands above the board, game would be over —
                # treat as invalid placement.
                if any(r < 0 for r, _ in placed):
                    continue

                pre_board = board.copy()
                for r, c in placed:
                    pre_board[r, c] = 1

                full = np.all(pre_board == 1, axis=1)
                lines = int(full.sum())
                # Eroded cells: only the piece's *own* cells that sat in a
                # row that got cleared, weighted by how many lines cleared.
                eroded = lines * sum(1 for r, _ in placed if full[r])
                # Landing height: piece midpoint measured from the floor
                # (before line clears shift things down).
                rs = [r for r, _ in placed]
                landing_height = self.HEIGHT - (min(rs) + max(rs)) / 2.0

                new_board, _ = self._clear_lines(pre_board)
                placements[(rot_idx, x)] = (new_board, lines,
                                            landing_height, eroded)
        return placements

    def _drop_y(self, cells, board):
        """Return the largest y_offset such that cells + y_offset fits in the board."""
        # Start with the piece just above its highest row; descend.
        y = -min(r for r, _ in cells)  # ensures topmost row >= 0
        if not self._fits(cells, y, board):
            return None
        while self._fits(cells, y + 1, board):
            y += 1
        return y

    @staticmethod
    def _fits(cells, y_offset, board):
        h, w = board.shape
        for r, c in cells:
            ry = r + y_offset
            if ry >= h or c < 0 or c >= w:
                return False
            if ry >= 0 and board[ry, c]:
                return False
        return True

    @staticmethod
    def _clear_lines(board):
        full = np.all(board == 1, axis=1)
        n = int(full.sum())
        if n == 0:
            return board, 0
        kept = board[~full]
        new_board = np.zeros_like(board)
        new_board[-len(kept):] = kept
        return new_board, n

    # -- features ---------------------------------------------------------
    @staticmethod
    def features(board, lines_cleared, landing_height, eroded):
        """Dellacherie 6-feature vector for an afterstate.

        lines_cleared is unused directly (it is folded into `eroded`), kept
        in the signature so callers can pass the full placement tuple.
        """
        h, w = board.shape
        b = board == 1

        # Row transitions: walls count as filled; fully empty rows skipped
        # so an empty board doesn't look "rough". Equivalent to counting
        # adjacent flips in [wall, row..., wall] per non-empty row.
        rpad = np.ones((h, w + 2), dtype=bool)
        rpad[:, 1:-1] = b
        row_flips = (rpad[:, 1:] != rpad[:, :-1]).sum(axis=1)
        row_trans = int(row_flips[b.any(axis=1)].sum())

        # Column transitions: open sky above (empty), floor below (filled).
        # Adjacent flips in [sky, col..., floor] per column.
        cpad = np.empty((h + 2, w), dtype=bool)
        cpad[0] = False
        cpad[1:-1] = b
        cpad[-1] = True
        col_trans = int((cpad[1:] != cpad[:-1]).sum())

        # Holes: empty cells below the topmost filled cell in each column =
        # (cells from first filled down) - (filled cells), per column.
        any_filled = b.any(axis=0)
        first_filled = b.argmax(axis=0)            # 0 when column empty
        holes_per_col = (h - first_filled) - b.sum(axis=0)
        holes = int(holes_per_col[any_filled].sum())

        # Cumulative wells: empty cell whose horizontal neighbours (or
        # walls) are filled; consecutive depth contributes triangularly
        # (1, 1+2, ...). The triangular sum is exactly the per-cell run
        # length summed, computed via the cumsum-reset trick (no Python
        # loop over cells).
        left = np.ones((h, w), dtype=bool)
        left[:, 1:] = b[:, :-1]
        right = np.ones((h, w), dtype=bool)
        right[:, :-1] = b[:, 1:]
        wellmask = (~b) & left & right
        cs = wellmask.cumsum(axis=0)
        reset = np.where(~wellmask, cs, 0)
        run = (cs - np.maximum.accumulate(reset, axis=0)) * wellmask
        wells = int(run.sum())

        return np.array([landing_height, eroded, row_trans,
                         col_trans, holes, wells], dtype=np.float32)

    # -- step -------------------------------------------------------------
    def step(self, action, placements):
        """Apply the chosen placement. Returns (reward, done)."""
        new_board, lines, _, _ = placements[action]
        self.board = new_board
        self.lines_total += lines
        self.pieces_total += 1
        # Reward: survive + quadratic bonus for multi-line clears.
        reward = 1.0 + (lines ** 2) * 10.0

        self.current_piece = self._draw_piece()
        next_placements = self.get_possible_placements()
        if not next_placements:
            self.game_over = True
            reward -= 5.0
        return reward, self.game_over

    # -- rendering --------------------------------------------------------
    def render(self):
        rows = []
        rows.append('+' + '-' * (self.WIDTH * 2) + '+')
        for r in range(self.HEIGHT):
            row = '|'
            for c in range(self.WIDTH):
                row += '[]' if self.board[r, c] else '  '
            row += '|'
            rows.append(row)
        rows.append('+' + '-' * (self.WIDTH * 2) + '+')
        rows.append(f' piece={self.current_piece} lines={self.lines_total} pieces={self.pieces_total}')
        return '\n'.join(rows)


# ---------------------------------------------------------------------------
# Value network and agent
# ---------------------------------------------------------------------------
class ValueNet(nn.Module):
    def __init__(self, in_dim=Tetris.FEATURE_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Agent:
    def __init__(self, lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_decay_steps=300,
                 buffer_size=30000, batch_size=512, device='cpu'):
        self.device = torch.device(device)
        self.q = ValueNet().to(self.device)
        self.target = ValueNet().to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_step = 0
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def _features_for_placements(self, placements):
        items = list(placements.items())
        feats = np.stack([
            Tetris.features(board, lines, lh, eroded)
            for (_, _), (board, lines, lh, eroded) in items
        ])
        return items, feats

    def select(self, placements, training=True, env=None, lookahead=False):
        """Pick (rotation, x) from placements dict.

        With lookahead and a known next piece (7-bag boundaries excepted),
        do deterministic 2-ply search: score each current placement by the
        best value reachable after also placing the next piece, using the
        learned value net as the leaf evaluator. All leaf boards are scored
        in one batched forward pass.
        """
        items, feats = self._features_for_placements(placements)

        if training and random.random() < self.eps:
            idx = random.randrange(len(items))
            return items[idx][0], feats[idx]

        next_piece = env.peek_next_piece() if (lookahead and env) else None
        if next_piece is None:
            # No lookahead (disabled, or unknown next piece at bag edge).
            with torch.no_grad():
                v = self.q(torch.from_numpy(feats).to(self.device)).cpu().numpy()
            idx = int(np.argmax(v))
            return items[idx][0], feats[idx]

        # 2-ply: for each current placement's board, enumerate the next
        # piece's placements and collect every leaf board's features.
        LOSS = -1e6  # any reachable real value dominates a forced game over
        leaf_feats, owner = [], []
        scores = np.full(len(items), LOSS, dtype=np.float64)
        for i, ((_, _), (board1, _, _, _)) in enumerate(items):
            nxt = env.get_possible_placements(piece=next_piece, board=board1)
            if not nxt:
                continue  # placing here forces a game over next piece
            for (_, _), (b2, l2, lh2, er2) in nxt.items():
                owner.append(i)
                leaf_feats.append(Tetris.features(b2, l2, lh2, er2))

        if leaf_feats:
            with torch.no_grad():
                lv = self.q(torch.from_numpy(np.stack(leaf_feats)
                                             ).to(self.device)).cpu().numpy()
            owner = np.asarray(owner)
            for i in np.unique(owner):
                scores[i] = lv[owner == i].max()

        # If every current move forces a loss, fall back to plain V(board1)
        # so we still pick the least-bad option.
        if np.all(scores <= LOSS):
            with torch.no_grad():
                scores = self.q(torch.from_numpy(feats).to(self.device)
                                ).cpu().numpy()

        idx = int(np.argmax(scores))
        return items[idx][0], feats[idx]

    def remember(self, chosen_feats, reward, next_feats, done):
        self.buffer.append((chosen_feats, reward, next_feats, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s = torch.from_numpy(np.stack([b[0] for b in batch])).to(self.device)
        r = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=self.device)
        d = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)

        # next-state values: max over the available afterstates after the next piece
        with torch.no_grad():
            vs = []
            for (_, _, ns_feats, done) in batch:
                if done or ns_feats is None or len(ns_feats) == 0:
                    vs.append(0.0)
                else:
                    nv = self.target(torch.from_numpy(ns_feats).to(self.device))
                    vs.append(float(nv.max().item()))
            target = r + self.gamma * torch.tensor(vs, device=self.device) * (1 - d)

        pred = self.q(s)
        # Huber (smooth L1) instead of MSE: bounded gradient on the large
        # TD errors that long Tetris episodes produce, which is the main
        # source of the late-training value oscillation.
        loss = nn.functional.smooth_l1_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        # Gradient clipping: a second guard against the same blow-up.
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def decay_eps(self):
        self.eps_step += 1
        frac = min(1.0, self.eps_step / self.eps_decay_steps)
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * frac


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(num_episodes=1500, max_pieces=500, target_sync_every=10,
          log_every=25, save_path='tetris_agent.pt', seed=0,
          train_every=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = Tetris()
    agent = Agent()
    lines_hist, reward_hist = [], []
    step_count = 0
    best_avg = -1.0          # best rolling-average lines seen so far
    eval_window = 100        # episodes the rolling average is taken over

    for ep in range(1, num_episodes + 1):
        env.reset()
        ep_reward = 0.0
        for _ in range(max_pieces):
            placements = env.get_possible_placements()
            if not placements:
                env.game_over = True
                break

            (rot, x), chosen_feats = agent.select(placements, training=True)
            reward, done = env.step((rot, x), placements)
            ep_reward += reward

            if done:
                next_feats = None
            else:
                next_placements = env.get_possible_placements()
                if next_placements:
                    _, next_feats = agent._features_for_placements(next_placements)
                else:
                    next_feats = None

            agent.remember(chosen_feats, reward, next_feats, done)
            step_count += 1
            if step_count % train_every == 0:
                agent.train_step()

            if done:
                break

        agent.decay_eps()
        if ep % target_sync_every == 0:
            agent.sync_target()

        lines_hist.append(env.lines_total)
        reward_hist.append(ep_reward)

        if ep % log_every == 0:
            recent_lines = np.mean(lines_hist[-log_every:])
            recent_reward = np.mean(reward_hist[-log_every:])
            # Rolling average over a wider window decides "best" — it is
            # far less noisy than the per-log-interval mean and is what we
            # actually want to maximise.
            roll_avg = float(np.mean(lines_hist[-eval_window:]))
            improved = roll_avg > best_avg
            tag = '  <- best, saved' if improved else ''
            print(f'ep {ep:5d} | avg lines (last {log_every}): {recent_lines:7.1f} '
                  f'| avg reward: {recent_reward:8.1f} | eps {agent.eps:.3f} '
                  f'| roll{eval_window}: {roll_avg:7.1f}{tag}',
                  flush=True)
            # Only overwrite the checkpoint when the rolling average
            # improves, so DQN oscillation can't clobber good weights with
            # a later bad region (the bug that lost the peak last run).
            if improved:
                best_avg = roll_avg
                torch.save({'state_dict': agent.q.state_dict(),
                            'lines_hist': lines_hist,
                            'reward_hist': reward_hist,
                            'best_avg': best_avg,
                            'best_ep': ep}, save_path)

    print(f'\nSaved best weights to {save_path}')
    print(f'Best {eval_window}-episode rolling average lines: {best_avg:.1f}')
    print(f'Final {eval_window}-episode average lines: '
          f'{np.mean(lines_hist[-eval_window:]):.1f}')
    return agent, lines_hist, reward_hist


def finetune(weights='tetris_agent.pt', num_episodes=300, max_pieces=400,
             target_sync_every=10, log_every=20,
             save_path='tetris_agent.pt', seed=1, train_every=4,
             eps_start=0.05, eps_end=0.01, eps_decay_steps=30):
    """Continue training from saved weights with lookahead-in-loop.

    Lookahead is the action-selection policy during rollouts, so V is
    regressed against samples the deployed (lookahead) policy actually
    visits — closing the policy/evaluator gap that limited the earlier
    1-piece-lookahead benchmark gain.

    The best-checkpoint guard is seeded with the loaded checkpoint's
    `best_avg`, so this can only overwrite `save_path` if the fine-tune
    actually exceeds the existing agent's rolling average. A worse
    fine-tune is harmless.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = Tetris()
    agent = Agent(eps_start=eps_start, eps_end=eps_end,
                  eps_decay_steps=eps_decay_steps)
    ckpt = torch.load(weights, map_location='cpu', weights_only=True)
    agent.q.load_state_dict(ckpt['state_dict'])
    agent.target.load_state_dict(ckpt['state_dict'])
    lines_hist = list(ckpt.get('lines_hist', []))
    reward_hist = list(ckpt.get('reward_hist', []))
    best_avg = float(ckpt.get('best_avg', 0.0))
    base_ep = len(lines_hist)
    eval_window = 100
    step_count = 0

    print(f'Fine-tuning from {weights} (loaded best_avg={best_avg:.1f}). '
          f'Will only overwrite if {eval_window}-ep rolling avg exceeds it.',
          flush=True)

    for ep in range(1, num_episodes + 1):
        env.reset()
        ep_reward = 0.0
        for _ in range(max_pieces):
            placements = env.get_possible_placements()
            if not placements:
                env.game_over = True
                break

            # Lookahead at action selection — the whole point of the
            # fine-tune. Random branch (eps) is handled inside select.
            (rot, x), chosen_feats = agent.select(
                placements, training=True, env=env, lookahead=True)
            reward, done = env.step((rot, x), placements)
            ep_reward += reward

            if done:
                next_feats = None
            else:
                next_placements = env.get_possible_placements()
                if next_placements:
                    _, next_feats = agent._features_for_placements(next_placements)
                else:
                    next_feats = None

            agent.remember(chosen_feats, reward, next_feats, done)
            step_count += 1
            if step_count % train_every == 0:
                agent.train_step()
            if done:
                break

        agent.decay_eps()
        if ep % target_sync_every == 0:
            agent.sync_target()

        lines_hist.append(env.lines_total)
        reward_hist.append(ep_reward)

        if ep % log_every == 0:
            recent_lines = np.mean(lines_hist[-log_every:])
            roll = float(np.mean(lines_hist[-eval_window:]))
            improved = roll > best_avg
            tag = '  <- best, saved' if improved else ''
            print(f'ft ep {ep:4d} | avg{log_every}: {recent_lines:7.1f} '
                  f'| roll{eval_window}: {roll:7.1f} | eps {agent.eps:.3f}{tag}',
                  flush=True)
            if improved:
                best_avg = roll
                torch.save({'state_dict': agent.q.state_dict(),
                            'lines_hist': lines_hist,
                            'reward_hist': reward_hist,
                            'best_avg': best_avg,
                            'best_ep': base_ep + ep,
                            'finetuned': True}, save_path)

    print(f'\nFine-tune done. Best {eval_window}-ep rolling avg: {best_avg:.1f}')
    return agent, lines_hist, reward_hist


def play(weights='tetris_agent.pt', render=True, max_pieces=2000, delay=0.08,
         lookahead=True):
    """Watch the trained agent play. Clears screen between pieces for animation."""
    import time, os, shutil
    env = Tetris()
    agent = Agent()
    state = torch.load(weights, map_location='cpu', weights_only=True)
    agent.q.load_state_dict(state['state_dict'])
    agent.eps = 0.0  # greedy

    use_ansi = render and sys.stdout.isatty()

    env.reset()
    for _ in range(max_pieces):
        placements = env.get_possible_placements()
        if not placements:
            break
        (rot, x), _ = agent.select(placements, training=False,
                                   env=env, lookahead=lookahead)
        env.step((rot, x), placements)
        if render:
            if use_ansi:
                # ANSI: cursor home + clear screen below
                sys.stdout.write('\033[H\033[J')
            print(env.render())
            sys.stdout.flush()
            time.sleep(delay)
        if env.game_over:
            break
    print(f'\nGame over: {env.game_over}. Lines cleared: {env.lines_total}, pieces placed: {env.pieces_total}')


def record(weights='tetris_agent.pt', out='tetris_play.gif',
           max_pieces=160, cell=22, fps=12, lookahead=True):
    """Render the trained agent playing to an animated GIF (headless).

    One frame per placement (afterstate formulation has no in-between
    falling animation). Capped at max_pieces so the GIF stays small.
    """
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    env = Tetris()
    agent = Agent()
    state = torch.load(weights, map_location='cpu', weights_only=True)
    agent.q.load_state_dict(state['state_dict'])
    agent.eps = 0.0  # greedy

    pad = 12
    header = 28
    W = Tetris.WIDTH * cell + 2 * pad
    H = Tetris.HEIGHT * cell + 2 * pad + header
    bg = (18, 18, 24)
    grid = (40, 40, 52)
    block = (90, 200, 250)

    def frame():
        img = Image.new('RGB', (W, H), bg)
        d = ImageDraw.Draw(img)
        d.text((pad, 8),
                f'lines {env.lines_total}   pieces {env.pieces_total}',
                fill=(220, 220, 230))
        oy = header + pad
        for r in range(Tetris.HEIGHT):
            for c in range(Tetris.WIDTH):
                x0 = pad + c * cell
                y0 = oy + r * cell
                d.rectangle([x0, y0, x0 + cell, y0 + cell], outline=grid)
                if env.board[r, c]:
                    d.rectangle([x0 + 1, y0 + 1, x0 + cell - 1, y0 + cell - 1],
                                fill=block)
        return img

    frames = [frame()]
    env.reset()
    for _ in range(max_pieces):
        placements = env.get_possible_placements()
        if not placements:
            break
        (rot, x), _ = agent.select(placements, training=False,
                                   env=env, lookahead=lookahead)
        env.step((rot, x), placements)
        frames.append(frame())
        if env.game_over:
            break

    imageio.mimsave(out, [f for f in frames], fps=fps, loop=0)
    print(f'Saved {len(frames)} frames to {out} '
          f'(lines={env.lines_total}, pieces={env.pieces_total})')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if cmd == 'train':
        train()
    elif cmd == 'finetune':
        finetune()
    elif cmd == 'play':
        play()
    elif cmd == 'record':
        record()
    else:
        print('usage: tetris_rl.py [train|finetune|play|record]')
