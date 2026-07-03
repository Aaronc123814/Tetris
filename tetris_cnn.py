"""
Tetris CNN-over-afterstate agent (hybrid).

Keeps the afterstate Q-learning formulation from tetris_rl.py — same
deterministic transition trick, same buffer / target-net machinery — but
swaps the 6 Dellacherie features for a CNN over a stack of afterstate maps:

  channel 0: post-clear afterstate board                       (20x10 binary)
  channel 1: just-placed piece cells in afterstate coordinates (20x10 binary)
             (same row-clear shift applied as channel 0, so the geometry
             matches; the channel-1 ∖ channel-0 difference is implicitly
             the eroded-piece-cells signal Dellacherie used)

The default 'rich' feature set (features='rich', --features rich) adds two
more, the long-range vertical signals a 3x3 conv stack can't compute itself:

  channel 2: column envelope (filled from each column's top cell down) — height
  channel 3: holes (empty cells with a filled cell somewhere above them)

'basic' (channels 0-1 only) reproduces the original 2-channel input.

Saves to `tetris_cnn.pt`, leaving `tetris_agent.pt` (the Dellacherie
agent) untouched for A/B comparison.

Why this design (vs `tetris_deep.py` which uses pre-state Q with action
masking and didn't learn): the afterstate formulation collapses the
"which action?" problem into "which resulting board?", so the network
only has to learn a board evaluator V(s'). That is by far the easiest
form of value function to learn on Tetris and is what the Dellacherie
agent already does, just with hand-crafted features instead of a CNN.

Run:
  python tetris_cnn.py train     # train and save to tetris_cnn.pt
  python tetris_cnn.py play      # watch the trained CNN agent
"""

import os
import sys
import random
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tetris_rl import Tetris, TETROMINOES


# ---------------------------------------------------------------------------
# Env: same as Tetris, but enumeration returns the placed-cell mask too.
# ---------------------------------------------------------------------------
class TetrisCNN(Tetris):
    """Adds the per-placement placed-piece-cell mask (in afterstate
    coordinates) to each placement record so we can build the 2-channel
    CNN input directly. Everything else is inherited."""

    def get_possible_placements(self, piece=None, board=None):
        if piece is None:
            piece = self.current_piece
        if board is None:
            board = self.board

        placements = {}
        for rot_idx, shape in enumerate(TETROMINOES[piece]):
            max_col = max(c for _, c in shape)
            for x in range(self.WIDTH - max_col):
                cells = [(r, c + x) for r, c in shape]
                y = self._drop_y(cells, board)
                if y is None:
                    continue
                placed = [(r + y, c) for r, c in cells]
                if any(r < 0 for r, _ in placed):
                    continue

                pre_board = board.copy()
                placed_mask = np.zeros_like(board)
                for r, c in placed:
                    pre_board[r, c] = 1
                    placed_mask[r, c] = 1

                full = np.all(pre_board == 1, axis=1)
                lines = int(full.sum())
                eroded = lines * sum(1 for r, _ in placed if full[r])
                rs = [r for r, _ in placed]
                landing_height = self.HEIGHT - (min(rs) + max(rs)) / 2.0

                # Apply the same row-clear shift to the placed mask so
                # channel 1 lives in the same coordinate system as channel 0.
                new_board, _ = self._clear_lines(pre_board)
                kept = placed_mask[~full]
                new_mask = np.zeros_like(placed_mask)
                if len(kept):
                    new_mask[-len(kept):] = kept

                placements[(rot_idx, x)] = (new_board, lines, landing_height,
                                            eroded, new_mask)
        return placements

    def step(self, action, placements):
        new_board, lines, _, _, _ = placements[action]
        self.board = new_board
        self.lines_total += lines
        self.pieces_total += 1
        reward = 1.0 + (lines ** 2) * 10.0
        self.current_piece = self._draw_piece()
        next_placements = self.get_possible_placements()
        if not next_placements:
            self.game_over = True
            reward -= 5.0
        return reward, self.game_over


# ---------------------------------------------------------------------------
# Afterstate CNN.
#
# Deeper/wider than v1, which plateaued ~125 lines vs the Dellacherie
# agent's 150. Two changes carry the rework:
#   - 3 conv layers (2->32->64->64) instead of 2, so the net can build the
#     multi-scale structure (holes, wells, column profiles) that Dellacherie
#     hand-codes.
#   - a 256-wide 2-layer head instead of squeezing 6400 conv activations
#     straight into 64 units, which was a severe information bottleneck.
# GroupNorm rather than BatchNorm on purpose: DQN *acts* on tiny per-piece
# placement batches (~34) but *learns* on a 512 batch; BatchNorm's
# batch-dependent statistics are unstable across that mismatch, while
# GroupNorm is batch-independent and behaves identically acting or learning.
#
# head='pool' (default) vs 'wide': the original 'wide' head flattened the full
# 64x20x10=12800 conv activations into the first Linear -> 3.28M weights, ~98%
# of the whole net. A 4.6k-param Dellacherie MLP beats it, so that head was
# overfitting the replay buffer, not under-capacity. 'pool' halves the spatial
# dims (20x10 -> 10x5) with a MaxPool before the flatten: 64x10x5=3200 inputs,
# cutting the head to ~0.82M (-75%). It is a 2x2 *pool*, NOT global average
# pooling on purpose -- vertical position (height) is the dominant Tetris
# signal, so collapsing all spatial structure with GAP would throw away the
# very thing the value depends on; 2x2 pooling keeps coarse position.
# ---------------------------------------------------------------------------
class AfterstateCNN(nn.Module):
    def __init__(self, in_ch=2, h=Tetris.HEIGHT, w=Tetris.WIDTH, hidden=256,
                 norm='group', head='pool'):
        super().__init__()

        def norm_layer(c):
            if norm == 'group':
                return nn.GroupNorm(8, c)
            if norm == 'batch':
                return nn.BatchNorm2d(c)
            if norm in (None, 'none'):
                return nn.Identity()
            raise ValueError(f"unknown norm {norm!r} (use group|batch|none)")

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )
        if head == 'pool':
            self.pool = nn.MaxPool2d(2)
            flat = 64 * (h // 2) * (w // 2)
        elif head == 'wide':
            self.pool = nn.Identity()
            flat = 64 * h * w
        else:
            raise ValueError(f"unknown head {head!r} (use pool|wide)")
        self.head = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.head(self.pool(self.conv(x)).flatten(1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
# Input feature sets. 'basic' is the original 2-channel input; 'rich' adds the
# two long-range vertical signals a 3x3-conv stack (7x7 receptive field) cannot
# compute from the raw board but that dominate Tetris board quality:
#   channel 2: column envelope — 1 from each column's topmost filled cell down
#              to the floor (encodes per-column height directly)
#   channel 3: holes — empty cells with a filled cell somewhere above them
# Both are exact Dellacherie-style signals; handing them to the net removes the
# burden of learning a 20-row vertical aggregation through tiny kernels.
_FEATURE_CHANNELS = {'basic': 2, 'rich': 4}


def _in_ch(features):
    if features not in _FEATURE_CHANNELS:
        raise ValueError(f"unknown features {features!r} (use basic|rich)")
    return _FEATURE_CHANNELS[features]


def _build_inputs(placements, features='basic'):
    """Stack (rot, x) -> placement dict entries into a (N, C, H, W) float32
    array and an aligned (N,) reward array. C is 2 ('basic') or 4 ('rich').
    Pure function — no agent state."""
    items = list(placements.items())
    n = len(items)
    h, w = Tetris.HEIGHT, Tetris.WIDTH
    c = _in_ch(features)
    x = np.zeros((n, c, h, w), dtype=np.float32)
    rewards = np.zeros(n, dtype=np.float32)
    for i, (_, (board, lines, _, _, placed_mask)) in enumerate(items):
        x[i, 0] = board
        x[i, 1] = placed_mask
        if c >= 4:
            filled = board == 1
            envelope = np.maximum.accumulate(filled, axis=0)  # height map
            x[i, 2] = envelope
            x[i, 3] = envelope & (~filled)                    # holes
        # Action score uses immediate reward + gamma*V; mirror env.step's
        # reward formula (without the -5 game-over penalty, since that
        # only applies if the *next* piece has no placements — which we
        # don't know at selection time).
        rewards[i] = 1.0 + (lines ** 2) * 10.0
    return items, x, rewards


def _buffer_to_blob(buffer):
    """Serialize the replay buffer to a dict of tensors + a piece-id list so
    it round-trips through torch.save/load with weights_only=True. numpy
    arrays are *not* weights_only-safe; tensors, str and bool/float lists are.
    next_board is None exactly for terminal items, so we stack zeros there and
    carry a separate `has_nb` mask."""
    buf = list(buffer)
    if not buf:
        return None
    h, w = Tetris.HEIGHT, Tetris.WIDTH
    inp = torch.from_numpy(np.stack([b[0] for b in buf]))          # (N,2,h,w) int8
    rew = torch.tensor([b[1] for b in buf], dtype=torch.float32)
    nb_list = [b[2] for b in buf]
    has_nb = torch.tensor([nb is not None for nb in nb_list])
    nbs = torch.from_numpy(np.stack(
        [nb if nb is not None else np.zeros((h, w), np.int8)
         for nb in nb_list]).astype(np.int8))                       # (N,h,w) int8
    pieces = ['' if b[3] is None else b[3] for b in buf]            # list[str]
    dones = torch.tensor([b[4] for b in buf])
    return {'inp': inp, 'rew': rew, 'nbs': nbs, 'has_nb': has_nb,
            'pieces': pieces, 'dones': dones}


def _blob_to_buffer(blob, maxlen):
    """Inverse of _buffer_to_blob."""
    buf = deque(maxlen=maxlen)
    if not blob:
        return buf
    inp, rew = blob['inp'].cpu().numpy(), blob['rew'].cpu().numpy()
    nbs, has_nb = blob['nbs'].cpu().numpy(), blob['has_nb'].cpu().numpy()
    pieces, dones = blob['pieces'], blob['dones'].cpu().numpy()
    for i in range(len(rew)):
        nb = nbs[i].copy() if has_nb[i] else None
        pc = pieces[i] if pieces[i] != '' else None
        buf.append((inp[i].copy().astype(np.int8), float(rew[i]),
                    nb, pc, bool(dones[i])))
    return buf


class AgentCNN:
    """Mirror of tetris_rl.Agent but with the CNN value head, batched
    per-placement forward at selection time, and a memory-light buffer
    that stores the next env state (board + piece) instead of all next
    placement inputs — we re-enumerate at training time."""

    def __init__(self, lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_decay_steps=300,
                 buffer_size=30000, batch_size=512, device='cpu', norm='group',
                 features='basic', head='pool', weight_decay=1e-4):
        self.device = torch.device(device)
        self.norm = norm
        self.features = features
        self.head_kind = head
        in_ch = _in_ch(features)
        self.q = AfterstateCNN(in_ch=in_ch, norm=norm, head=head).to(self.device)
        self.target = AfterstateCNN(in_ch=in_ch, norm=norm,
                                    head=head).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        # weight_decay regularizes the head Linear, the prime overfitting
        # suspect; pair it with head='pool' or sweep them apart to attribute.
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        self.gamma = gamma
        self.eps = eps_start
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_step = 0
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # Stateless helper for re-enumerating placements at training time
        # without disturbing the live training env. Avoid Tetris.__init__
        # so we don't perturb the global RNG.
        self._helper = TetrisCNN.__new__(TetrisCNN)

    def select(self, placements, training=True, env=None, lookahead=False):
        """Pick (rotation, x) from placements.

        The value head is trained so V(σ) already bakes in σ's own arrival
        reward (V(σ) ≈ r_arrival + γ·max V(next); see train_step), so the
        network value *is* the action value. We therefore rank candidates by
        V directly — the same convention as the Dellacherie agent in
        tetris_rl.py. Scoring `r + γ·V` here (the previous rework's bug)
        double-counts the immediate reward and breaks consistency with the
        training target.

        With `lookahead=True` and a known next piece (7-bag boundaries
        excepted), refine each candidate by enumerating the next piece's
        placements and scoring by the best reachable afterstate value
        max_{a2} V(σ2), all leaves in one batched forward. Falls back to the
        1-ply V(σ) when the next piece is unknown.
        """
        items, x, _ = _build_inputs(placements, self.features)

        if training and random.random() < self.eps:
            idx = random.randrange(len(items))
            return items[idx][0], x[idx]

        next_piece = env.peek_next_piece() if (lookahead and env) else None
        if next_piece is None:
            # 1-ply: rank by V(σ') directly.
            with torch.no_grad():
                v = self.q(torch.from_numpy(x).to(self.device)).cpu().numpy()
            idx = int(np.argmax(v))
            return items[idx][0], x[idx]

        # 2-ply: collect every leaf (board2) across all candidate a1's, score
        # them in one batched forward, then take each candidate's best leaf V.
        LOSS = -1e9
        leaf_x, owner = [], []
        for i, ((_, _), (board1, _, _, _, _)) in enumerate(items):
            nxt = env.get_possible_placements(piece=next_piece, board=board1)
            if not nxt:
                continue
            _, x2, _ = _build_inputs(nxt, self.features)
            for j in range(len(x2)):
                owner.append(i)
                leaf_x.append(x2[j])

        scores = np.full(len(items), LOSS, dtype=np.float64)
        if leaf_x:
            with torch.no_grad():
                big = np.stack(leaf_x)
                v_leaf = self.q(torch.from_numpy(big).to(self.device)
                                ).cpu().numpy()
            owner = np.asarray(owner)
            for i in np.unique(owner):
                scores[i] = v_leaf[owner == i].max()

        # If every candidate forces a game over, fall back to plain 1-ply
        # (rank by V) so we still pick the least-bad option.
        if np.all(scores <= LOSS):
            with torch.no_grad():
                scores = self.q(torch.from_numpy(x).to(self.device)).cpu().numpy()

        idx = int(np.argmax(scores))
        return items[idx][0], x[idx]

    def remember(self, chosen_input, reward, next_board, next_piece, done):
        # Compact storage: 0/1 inputs as int8; everything else trivial.
        self.buffer.append((chosen_input.astype(np.int8), float(reward),
                            None if next_board is None else next_board.copy(),
                            next_piece, bool(done)))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)

        s = torch.from_numpy(
            np.stack([b[0] for b in batch]).astype(np.float32)
        ).to(self.device)
        r = torch.tensor([b[1] for b in batch], dtype=torch.float32,
                         device=self.device)
        d = torch.tensor([float(b[4]) for b in batch], device=self.device)

        # Re-enumerate next placements per item and stack into one big
        # forward through the target net — one pass instead of 128.
        all_next_x = []
        per_item_n = []
        for _, _, nb, np_piece, done in batch:
            if done or nb is None or np_piece is None:
                per_item_n.append(0)
                continue
            placements = self._helper.get_possible_placements(
                piece=np_piece, board=nb)
            if not placements:
                per_item_n.append(0)
                continue
            _, x_arr, _ = _build_inputs(placements, self.features)
            all_next_x.append(x_arr)
            per_item_n.append(len(placements))

        with torch.no_grad():
            next_v = torch.zeros(self.batch_size, device=self.device)
            if all_next_x:
                big = torch.from_numpy(
                    np.concatenate(all_next_x, axis=0)).to(self.device)
                # Double DQN: the ONLINE net picks the best next afterstate,
                # the TARGET net supplies its value. Decoupling selection from
                # evaluation curbs the max-operator overestimation bias that
                # plain target.max() suffers from.
                online_flat = self.q(big).cpu().numpy()
                target_flat = self.target(big).cpu().numpy()
                off = 0
                for i, n in enumerate(per_item_n):
                    if n > 0:
                        a = int(online_flat[off:off + n].argmax())
                        next_v[i] = float(target_flat[off + a])
                        off += n
            target = r + self.gamma * next_v * (1.0 - d)

        pred = self.q(s)
        loss = F.smooth_l1_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        # Mean predicted V is the cheapest divergence canary: in a healthy run
        # it tracks the rolling return; if it runs away while lines stagnate,
        # the value function is diverging.
        return float(loss.item()), float(pred.mean().item())

    def sync_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def decay_eps(self):
        self.eps_step += 1
        frac = min(1.0, self.eps_step / self.eps_decay_steps)
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * frac


# ---------------------------------------------------------------------------
# Training & play
# ---------------------------------------------------------------------------
def pick_device(requested=None):
    """Resolve the compute device: explicit override, else cuda > mps > cpu.

    `mps` is Apple Silicon's Metal backend — on a Mac that's what "GPU"
    means, since `torch.cuda` is NVIDIA-only and stays False there.
    """
    if requested:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return 'mps'
    return 'cpu'


def train(num_episodes=10000, max_pieces=500, target_sync_every=10,
          log_every=25, save_every=50, save_path='tetris_cnn.pt', seed=0,
          train_every=4, resume=None, lr=2.5e-4, batch_size=512, device=None,
          eps_start=1.0, eps_end=0.01, eps_decay_steps=2000,
          norm='group', save_buffer=True, features='rich',
          train_lookahead=False, head='pool', weight_decay=1e-4):
    """Train, with full-state checkpointing so a run can span sessions
    (e.g. Kaggle's 12h limit) without losing progress.

    `num_episodes` is the *total* episode budget. Pass `resume=save_path`
    (or `resume=True`) and re-run the same call each session: training picks
    up at the saved episode with ε, optimizer, history and best weights all
    restored, and stops once the running total reaches `num_episodes`.

    The checkpoint is written every `save_every` episodes (the resume point)
    and whenever the 100-episode rolling average hits a new best. It holds
    both the live weights (`state_dict`, for resuming) and the best weights
    seen (`best_state_dict`, what `play` loads). Saving only on improvement —
    the old behaviour — would lose all progress since the last improvement if
    a session were killed mid-run, which is exactly what spanning sessions
    can't afford.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = pick_device(device)
    env = TetrisCNN()
    agent = AgentCNN(device=device, lr=lr, batch_size=batch_size,
                     eps_start=eps_start, eps_end=eps_end,
                     eps_decay_steps=eps_decay_steps, norm=norm,
                     features=features, head=head, weight_decay=weight_decay)
    print(f'Device: {device} | batch_size: {batch_size} | features: {features} '
          f'({_in_ch(features)}ch) | norm: {norm} | head: {head} | '
          f'wd: {weight_decay:g} | train_lookahead: {train_lookahead}', flush=True)

    eval_window = 100
    lines_hist, reward_hist = [], []
    recent_loss, recent_v = deque(maxlen=1000), deque(maxlen=1000)
    step_count = 0
    best_avg = -1.0
    best_ep = 0
    start_ep = 0

    def cpu_state(module):
        return {k: v.detach().cpu() for k, v in module.state_dict().items()}

    best_state = cpu_state(agent.q)

    def save_ckpt(ep):
        ckpt = {'state_dict': cpu_state(agent.q),
                'best_state_dict': best_state,
                'opt': agent.opt.state_dict(),
                'eps': agent.eps,
                'eps_step': agent.eps_step,
                'step_count': step_count,
                'episode': ep,
                'lines_hist': lines_hist,
                'reward_hist': reward_hist,
                'best_avg': best_avg,
                'best_ep': best_ep,
                'norm': agent.norm,
                'features': agent.features,
                'head': agent.head_kind}
        # Persisting the replay buffer makes resume seamless (no empty-buffer
        # destabilization), at the cost of a larger checkpoint (~20MB for a
        # full 30k buffer). Pass save_buffer=False for lean, portable files.
        if save_buffer:
            ckpt['buffer'] = _buffer_to_blob(agent.buffer)
        torch.save(ckpt, save_path)

    resume_path = save_path if resume is True else resume
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        ckpt_norm = ckpt.get('norm', 'group')
        if ckpt_norm != norm:
            raise ValueError(
                f"checkpoint {resume_path} was trained with norm={ckpt_norm!r} "
                f"but train(norm={norm!r}) was requested; pass norm={ckpt_norm!r} "
                f"to resume it (norm changes the architecture).")
        ckpt_feat = ckpt.get('features', 'basic')
        if ckpt_feat != features:
            raise ValueError(
                f"checkpoint {resume_path} was trained with features={ckpt_feat!r} "
                f"but train(features={features!r}) was requested; pass "
                f"features={ckpt_feat!r} to resume it (the input-channel count, "
                f"hence the first conv, differs).")
        # Older checkpoints predate the pooled head, so they are 'wide'.
        ckpt_head = ckpt.get('head', 'wide')
        if ckpt_head != head:
            raise ValueError(
                f"checkpoint {resume_path} was trained with head={ckpt_head!r} "
                f"but train(head={head!r}) was requested; pass head={ckpt_head!r} "
                f"to resume it (the head changes the first Linear's shape).")
        agent.q.load_state_dict(ckpt['state_dict'])
        agent.target.load_state_dict(ckpt['state_dict'])
        if 'opt' in ckpt:
            agent.opt.load_state_dict(ckpt['opt'])
            # Make the requested lr / weight_decay authoritative on resume —
            # Adam's load_state_dict otherwise restores the checkpoint's
            # values, so a `--lr` / `--weight-decay` change would silently
            # have no effect.
            for g in agent.opt.param_groups:
                g['lr'] = lr
                g['weight_decay'] = weight_decay
        agent.eps = float(ckpt.get('eps', eps_end))
        agent.eps_step = int(ckpt.get('eps_step', eps_decay_steps))
        lines_hist = list(ckpt.get('lines_hist', []))
        reward_hist = list(ckpt.get('reward_hist', []))
        best_avg = float(ckpt.get('best_avg', -1.0))
        best_ep = int(ckpt.get('best_ep', 0))
        best_state = ckpt.get('best_state_dict') or ckpt['state_dict']
        step_count = int(ckpt.get('step_count', 0))
        start_ep = int(ckpt.get('episode', len(lines_hist)))
        if ckpt.get('buffer') is not None:
            agent.buffer = _blob_to_buffer(ckpt['buffer'], agent.buffer.maxlen)
        print(f'Resumed from {resume_path}: episode {start_ep}/{num_episodes}, '
              f'best_avg={best_avg:.1f}, eps={agent.eps:.3f}, '
              f'buffer={len(agent.buffer)}, lr={lr:g}', flush=True)
    elif resume_path:
        print(f'No checkpoint at {resume_path} yet — starting fresh.', flush=True)

    if start_ep >= num_episodes:
        print(f'Already at {start_ep}/{num_episodes} episodes — nothing to do.',
              flush=True)
        return

    for ep in range(start_ep + 1, num_episodes + 1):
        env.reset()
        ep_reward = 0.0
        for _ in range(max_pieces):
            placements = env.get_possible_placements()
            if not placements:
                env.game_over = True
                break

            # Lookahead-in-the-loop (optional): regress V against the same
            # 2-ply policy that `play` deploys, closing the policy/evaluator
            # gap. Costs more per step but compute is the cheap axis here.
            (rot, x), chosen_input = agent.select(
                placements, training=True,
                env=env if train_lookahead else None,
                lookahead=train_lookahead)
            reward, done = env.step((rot, x), placements)
            ep_reward += reward

            # We store the env state right after step() so the buffer item
            # encodes "from this resulting state, with this piece up next,
            # what's V?".
            if done:
                agent.remember(chosen_input, reward, None, None, True)
            else:
                agent.remember(chosen_input, reward,
                               env.board, env.current_piece, False)

            step_count += 1
            if step_count % train_every == 0:
                out = agent.train_step()
                if out is not None:
                    recent_loss.append(out[0])
                    recent_v.append(out[1])

            if done:
                break

        agent.decay_eps()
        if ep % target_sync_every == 0:
            agent.sync_target()

        lines_hist.append(env.lines_total)
        reward_hist.append(ep_reward)

        improved = False
        if ep % log_every == 0:
            recent = float(np.mean(lines_hist[-log_every:]))
            roll = float(np.mean(lines_hist[-eval_window:]))
            improved = roll > best_avg
            if improved:
                best_avg = roll
                best_ep = ep
                best_state = cpu_state(agent.q)
            tag = '  <- new best' if improved else ''
            lo = float(np.mean(recent_loss)) if recent_loss else float('nan')
            vb = float(np.mean(recent_v)) if recent_v else float('nan')
            print(f'ep {ep:5d} | avg{log_every}: {recent:7.1f} '
                  f'| roll{eval_window}: {roll:7.1f} | eps {agent.eps:.3f} '
                  f'| loss {lo:7.3f} | V̄ {vb:6.1f} '
                  f'| best {best_avg:6.1f}{tag}', flush=True)

        # Save on a new best (so the best weights are always on disk) and
        # every save_every episodes (so a killed session loses at most that
        # many episodes of progress on resume).
        if improved or ep % save_every == 0:
            save_ckpt(ep)

    save_ckpt(num_episodes)
    print(f'\nDone: {num_episodes} episodes. Best {eval_window}-ep rolling '
          f'avg: {best_avg:.1f} (ep {best_ep}).', flush=True)


def play(weights='tetris_cnn.pt', render=True, max_pieces=2000, delay=0.08,
         lookahead=True, device=None):
    import time
    env = TetrisCNN()
    device = pick_device(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    # Build the net with the same norm + input features it was trained with,
    # then load weights.
    agent = AgentCNN(device=device, norm=state.get('norm', 'group'),
                     features=state.get('features', 'basic'),
                     head=state.get('head', 'wide'))
    # Prefer the best-ever weights; fall back to live weights for older files.
    agent.q.load_state_dict(state.get('best_state_dict') or state['state_dict'])
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
                sys.stdout.write('\033[H\033[J')
            print(env.render())
            sys.stdout.flush()
            time.sleep(delay)
        if env.game_over:
            break
    print(f'\nGame over: {env.game_over}. '
          f'Lines: {env.lines_total}, pieces: {env.pieces_total}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tetris CNN afterstate agent.')
    sub = parser.add_subparsers(dest='cmd')

    pt = sub.add_parser('train', help='train and checkpoint to tetris_cnn.pt')
    pt.add_argument('--episodes', type=int, default=10000,
                    help='total episode budget (resume counts toward it)')
    pt.add_argument('--batch-size', type=int, default=512,
                    help='larger batches are nearly free on GPU and cut '
                         'gradient noise; try 1024-2048 with plenty of VRAM')
    pt.add_argument('--device', default=None,
                    help='cuda | mps | cpu (default: auto-detect cuda>mps>cpu)')
    pt.add_argument('--lr', type=float, default=2.5e-4,
                    help='Adam learning rate (default 2.5e-4, the canonical '
                         'DQN value; authoritative on resume too)')
    pt.add_argument('--eps-end', type=float, default=0.01,
                    help='exploration floor (kept low: a random drop can end '
                         'a game, so a high floor caps the converged average)')
    pt.add_argument('--eps-decay-steps', type=int, default=2000,
                    help='episodes to linearly anneal ε from 1.0 to --eps-end')
    pt.add_argument('--resume', default=None,
                    help='checkpoint to continue from (full state restored); '
                         'usually the same path as --save-path')
    pt.add_argument('--save-path', default='tetris_cnn.pt')
    pt.add_argument('--save-every', type=int, default=50,
                    help='episodes between resume checkpoints (smaller = less '
                         'lost if a session is killed, more disk I/O)')
    pt.add_argument('--norm', default='group',
                    choices=['group', 'batch', 'none'],
                    help='conv normalization (A/B test the GroupNorm choice); '
                         'changes the architecture, so resume must match')
    pt.add_argument('--features', default='rich',
                    choices=['basic', 'rich'],
                    help="input channels: 'basic' = board + placed mask (2ch); "
                         "'rich' = + column-height + holes maps (4ch, default). "
                         'changes the architecture, so resume must match')
    pt.add_argument('--head', default='pool', choices=['pool', 'wide'],
                    help="value head: 'pool' = 2x2 MaxPool before the flatten "
                         "(64x10x5->256, ~0.82M, default); 'wide' = flatten the "
                         'full 64x20x10 (~3.28M, the original). changes the '
                         "first Linear's shape, so resume must match. use "
                         "--head wide --weight-decay 0 to reproduce the old net")
    pt.add_argument('--weight-decay', type=float, default=1e-4,
                    help='Adam L2 weight decay regularizing the head (default '
                         '1e-4; pass 0 to disable). authoritative on resume')
    pt.add_argument('--train-lookahead', action='store_true',
                    help='use 2-ply lookahead as the behavior policy during '
                         'training (matches play; slower but compute is cheap)')
    pt.add_argument('--no-save-buffer', action='store_true',
                    help='do not store the replay buffer in the checkpoint '
                         '(smaller files, but resume starts empty)')

    pp = sub.add_parser('play', help='watch the trained agent')
    pp.add_argument('--weights', default='tetris_cnn.pt')
    pp.add_argument('--device', default=None)
    pp.add_argument('--no-lookahead', action='store_true')

    args = parser.parse_args()
    if args.cmd == 'play':
        play(weights=args.weights, device=args.device,
             lookahead=not args.no_lookahead)
    elif args.cmd == 'train':
        train(num_episodes=args.episodes, batch_size=args.batch_size,
              device=args.device, lr=args.lr, resume=args.resume,
              save_path=args.save_path, save_every=args.save_every,
              norm=args.norm, save_buffer=not args.no_save_buffer,
              eps_end=args.eps_end, eps_decay_steps=args.eps_decay_steps,
              features=args.features, train_lookahead=args.train_lookahead,
              head=args.head, weight_decay=args.weight_decay)
    else:  # bare `python tetris_cnn.py` -> train with defaults
        train()
