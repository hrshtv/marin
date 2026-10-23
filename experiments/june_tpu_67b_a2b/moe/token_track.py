# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-token loss tracking across epochs (Harshit's rule, 2026-10-07).

Every trained token's loss is recorded at each visit. The Snowball loader samples rows with replacement, so visits
are irregular (a row can return after 1 step or 25, and a "pass" is only a token-equivalent count); the rule counts
visits, not epochs. At a visit, once a token has two earlier observations, its last three losses
(l_{v-2}, l_{v-1}, l_v) are compared; if both consecutive relative changes are below ``tol`` the token is FLAT -
either already learned or not learnable - and it gets no gradient this visit. Observation never stops: a dropped
token whose loss later moves by more than ``tol`` is no longer flat and is trained again. With ``Reentry.RISE`` only
a rise brings it back: a dropped token whose loss falls stays dropped (the history keeps each token's last decision).
With ``drop_weight > 0`` a dropped token is down-weighted instead of removed: its loss keeps that fraction of its weight.

Mechanics. A token is keyed by (cache row, offset within the row), which is stable across epochs because the
cache row IS the document and packing only moves whole documents. A replicated host-side table holds each
token's last two observed losses and visit count. Before a step the host gathers those for the batch's positions
and hands them to the step as extra inputs; inside the step the current loss completes the triple and
``tracked_loss`` applies the mask; after the step the host writes the current losses back.

The table is checkpointed beside the model checkpoints (``save``/``load``) so a resumed job continues the same
histories; a table whose step does not match the restored model is ignored and tracking starts afresh.
"""

from __future__ import annotations

import os
from enum import StrEnum

import jax
import jax.numpy as jnp
import numpy as np

from experiments.june_tpu_67b_a2b.moe import tail_filter

TRACK_STAT_KEYS: tuple[str, ...] = (
    "track/kept_token_frac",
    "track/dropped_frac",
    "track/dropped_low_frac",
    "track/dropped_high_frac",
    "track/eligible_frac",
    "track/loss_unfiltered",
    "track/loss_dropped",
    "track/trained_tokens",
    "track/kept_tokens",
)
ALREADY_LEARNED_LOSS = 0.10536  # -log(0.9): a flat token below this is "learned", above it "not learnable"


class Reentry(StrEnum):
    """When a token dropped at its previous visit is trained again."""

    EITHER = "either"  # its loss moved by more than tol in either direction (it is no longer flat)
    RISE = "rise"  # its loss rose by more than tol; a falling loss keeps it dropped


def token_keys(segment_ids: np.ndarray, weight: np.ndarray, num_docs: int, max_len: int) -> np.ndarray:
    """``(B, S)`` int64 key per trained position (``row * max_len + offset``), ``-1`` elsewhere.

    Documents are contiguous within a packed row, so a token's offset in its document is its index minus the
    index where that run of equal segment ids began. Shared by the tracker's history table and the token-level
    TailSFT reference table, so both address the same (document, offset) cell.
    """
    seg = np.asarray(segment_ids).astype(np.int64)
    idx = np.broadcast_to(np.arange(seg.shape[1])[None, :], seg.shape)
    starts = np.ones_like(seg, dtype=bool)
    starts[:, 1:] = seg[:, 1:] != seg[:, :-1]
    run_start = np.maximum.accumulate(np.where(starts, idx, 0), axis=1)
    offset = idx - run_start
    trained = (np.asarray(weight) > 0) & (seg >= 0)
    if trained.any() and int(offset[trained].max()) >= max_len:
        raise ValueError(f"a document is longer than max_len={max_len}: offset {int(offset[trained].max())}")
    if trained.any() and int(seg[trained].max()) >= num_docs:
        raise ValueError(f"segment id {int(seg[trained].max())} >= num_docs={num_docs}")
    return np.where(trained, seg * max_len + offset, -1)


class TokenHistory:
    """Replicated host-side table: the last two observed losses, the visit count and the last drop decision of every token."""

    def __init__(self, num_docs: int, max_len: int):
        self.num_docs, self.max_len = int(num_docs), int(max_len)
        n = self.num_docs * self.max_len
        self.prev1 = np.full(n, np.nan, dtype=np.float32)  # most recent observed loss
        self.prev2 = np.full(n, np.nan, dtype=np.float32)  # the one before it
        self.count = np.zeros(n, dtype=np.int16)
        self.dropped = np.zeros(n, dtype=bool)  # whether the rule dropped the token at its most recent visit

    def save(self, path: str, step: int) -> None:
        """Write the table for ``step`` atomically. Call from one process: every process holds the same table."""
        tmp = f"{path}.tmp-{os.getpid()}.npz"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # the permanent checkpoint folder may not exist yet
        np.savez(tmp, prev1=self.prev1, prev2=self.prev2, count=self.count, dropped=self.dropped, step=np.int64(step),
                 num_docs=np.int64(self.num_docs), max_len=np.int64(self.max_len))
        os.replace(tmp, path)

    def load(self, path: str, step: int) -> bool:
        """Restore the table saved for ``step``. Returns False (table untouched) if the file is missing, was saved
        at another step, or describes a differently sized table."""
        if not os.path.exists(path):
            return False
        with np.load(path) as z:
            if int(z["step"]) != step or int(z["num_docs"]) != self.num_docs or int(z["max_len"]) != self.max_len:
                return False
            self.prev1[:] = z["prev1"]
            self.prev2[:] = z["prev2"]
            self.count[:] = z["count"]
            self.dropped[:] = z["dropped"]
        return True

    def keys(self, segment_ids: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """``(B, S)`` int64 key per trained position (``row * max_len + offset``), ``-1`` elsewhere (see ``token_keys``)."""
        return token_keys(segment_ids, weight, self.num_docs, self.max_len)

    def gather(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(prev1, prev2, count, dropped)`` for the batch's positions, NaN / 0 where there is no trained token."""
        safe = np.where(keys >= 0, keys, 0)
        valid = keys >= 0
        p1 = np.where(valid, self.prev1[safe], np.nan).astype(np.float32)
        p2 = np.where(valid, self.prev2[safe], np.nan).astype(np.float32)
        c = np.where(valid, self.count[safe], 0).astype(np.float32)
        d = np.where(valid, self.dropped[safe], False).astype(np.float32)
        return p1, p2, c, d

    def update(self, keys: np.ndarray, loss_now: np.ndarray, dropped_now: np.ndarray) -> dict[str, int]:
        """Record this visit's losses and drop decisions (``decide``). Returns duplicate-key diagnostics (see below).

        A key seen twice in one batch means the same (document, offset) landed in two rows of the batch - for
        example the last rows of one epoch and the first rows of the next. The first occurrence is recorded and
        the rest are counted and returned, so the caller can log them; a large count means the keying is wrong.
        """
        valid = keys >= 0
        k = keys[valid]
        l = np.asarray(loss_now, dtype=np.float32)[valid]
        d = np.asarray(dropped_now, dtype=bool)[valid]
        uniq, first = np.unique(k, return_index=True)
        diag = {"keys": int(len(k)), "duplicate_keys": int(len(k) - len(uniq))}
        if diag["duplicate_keys"]:
            docs = k // self.max_len
            dup_mask = np.ones(len(k), dtype=bool); dup_mask[first] = False
            diag["documents_with_duplicates"] = int(len(np.unique(docs[dup_mask])))
            k, l, d = uniq, l[first], d[first]
        self.prev2[k] = self.prev1[k]
        self.prev1[k] = l
        self.count[k] += 1
        self.dropped[k] = d
        return diag

    def update_ema(self, keys: np.ndarray, loss_now: np.ndarray, *, floor: float, alpha: float) -> dict[str, int]:
        """EMA variant: the ``prev2`` slot holds an exponential moving average of the relative visit-to-visit change.

        First difference seeds the average; afterwards ``ema <- (1 - alpha) * ema + alpha * rel``. Same duplicate
        handling as ``update``.
        """
        valid = keys >= 0
        k = keys[valid]
        l = np.asarray(loss_now, dtype=np.float32)[valid]
        uniq, first = np.unique(k, return_index=True)
        diag = {"keys": int(len(k)), "duplicate_keys": int(len(k) - len(uniq))}
        if diag["duplicate_keys"]:
            docs = k // self.max_len
            dup_mask = np.ones(len(k), dtype=bool); dup_mask[first] = False
            diag["documents_with_duplicates"] = int(len(np.unique(docs[dup_mask])))
            k, l = uniq, l[first]
        p1 = self.prev1[k]; c = self.count[k]
        rel = np.abs(l - p1) / np.maximum(p1, floor)
        ema = np.where(c >= 2, (1.0 - alpha) * self.prev2[k] + alpha * rel, rel).astype(np.float32)
        self.prev2[k] = np.where(c >= 1, ema, np.nan)
        self.prev1[k] = l
        self.count[k] += 1
        return diag


class TraceRecorder:
    """Per-visit record of every trained token in a fixed sample of documents (the token-level analysis).

    For each visit it stores the step, document, offset, target token id, the loss at this visit, the visit number
    (1 = first) and whether the token was kept. ``kept`` is recomputed on the host from the same history inputs and
    rule as ``tracked_loss`` (``plateaued``), so it matches the training mask. Rows accumulate in memory and are
    written as numbered npz shards by ``flush``; call it from one process only.
    """

    def __init__(self, docs: np.ndarray, max_len: int, out_dir: str, *, tol: float, floor: float, shard_rows: int = 2_000_000):
        self.docs = np.unique(np.asarray(docs, dtype=np.int64))
        self.max_len, self.out_dir, self.tol, self.floor, self.shard_rows = int(max_len), out_dir, float(tol), float(floor), shard_rows
        self.rows: list[dict[str, np.ndarray]] = []
        self.pending, self.shards = 0, 0

    def record(self, step: int, keys: np.ndarray, target_ids: np.ndarray, loss_now: np.ndarray,
               prev1: np.ndarray, prev2: np.ndarray, count: np.ndarray) -> None:
        k = np.asarray(keys).reshape(-1)
        doc = np.where(k >= 0, k // self.max_len, -1)
        sel = (k >= 0) & np.isin(doc, self.docs)
        if not sel.any():
            return
        p1, p2, c = (np.asarray(a).reshape(-1)[sel] for a in (prev1, prev2, count))
        now = np.asarray(loss_now, dtype=np.float32).reshape(-1)[sel]
        flat = plateaued(p1, p2, c, now, tol=self.tol, floor=self.floor)
        self.rows.append({
            "step": np.full(int(sel.sum()), step, dtype=np.int32), "doc": doc[sel].astype(np.int32),
            "offset": (k[sel] % self.max_len).astype(np.int32), "token": np.asarray(target_ids).reshape(-1)[sel].astype(np.int32),
            "loss": now.astype(np.float32), "visit": (c + 1).astype(np.int16), "kept": ~flat,
        })
        self.pending += int(sel.sum())
        if self.pending >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        merged = {key: np.concatenate([r[key] for r in self.rows]) for key in self.rows[0]}
        np.savez(os.path.join(self.out_dir, f"trace-{self.shards:04d}.npz"), **merged)
        self.rows, self.pending, self.shards = [], 0, self.shards + 1


def plateaued(prev1: np.ndarray, prev2: np.ndarray, count: np.ndarray, loss_now: np.ndarray, *, tol: float, floor: float) -> np.ndarray:
    """Host copy of ``tracked_loss``'s hard rule: from the third visit, both relative changes below ``tol``."""
    p1, p2, now = (np.asarray(a, dtype=np.float32) for a in (prev1, prev2, loss_now))
    rel = lambda a, b: np.abs(b - a) / np.maximum(a, np.float32(floor))
    with np.errstate(invalid="ignore"):
        return (np.asarray(count) >= 2) & np.isfinite(p1) & np.isfinite(p2) & np.isfinite(now) & (rel(p2, p1) < tol) & (rel(p1, now) < tol)


def decide(prev1: np.ndarray, prev2: np.ndarray, count: np.ndarray, prev_dropped: np.ndarray, loss_now: np.ndarray, *,
           tol: float, floor: float, reentry: Reentry) -> np.ndarray:
    """Host copy of ``tracked_loss``'s drop decision for the hard rule, including the re-entry rule."""
    flat = plateaued(prev1, prev2, count, loss_now, tol=tol, floor=floor)
    if reentry is Reentry.EITHER:
        return flat
    p1, now = (np.asarray(a, dtype=np.float32) for a in (prev1, loss_now))
    was = np.asarray(prev_dropped) > 0.5
    with np.errstate(invalid="ignore"):
        stays = was & np.isfinite(p1) & np.isfinite(now) & ((now - p1) / np.maximum(p1, np.float32(floor)) < tol)
    return np.where(was, stays, flat)


def tracked_loss(
    per_pos: jax.Array,
    weight: jax.Array,
    prev1: jax.Array,
    prev2: jax.Array,
    count: jax.Array,
    prev_dropped: jax.Array,
    *,
    tol: float,
    floor: float,
    ema_alpha: float = 0.0,
    reentry: Reentry = Reentry.EITHER,
    drop_weight: float = 0.0,
    psum: tail_filter.PsumFn = tail_filter._identity,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Token-mean loss over the tokens whose loss is still moving, with replicated diagnostics.

    With ``ema_alpha > 0`` the ``prev2`` input is the EMA of past relative changes (see ``update_ema``) and a
    token is flat when the EMA updated with this visit's change is below ``tol``; otherwise the hard rule
    (both of the last two changes below ``tol``) applies.

    ``per_pos`` is the trainer's per-position loss (already multiplied by ``weight``); the per-token loss is
    ``per_pos / weight``. Relative change ``|b - a| / max(a, floor)``; flat means both changes below ``tol``.

    With ``Reentry.RISE`` a token dropped at its previous visit (``prev_dropped``) stays dropped unless its loss rose
    by at least ``tol`` (signed change ``(now - prev1) / max(prev1, floor)``).

    ``drop_weight`` is the weight a dropped token keeps (0 removes it). The loss is the weighted mean over kept tokens
    at weight 1 and dropped tokens at ``drop_weight``.
    """
    w = weight.reshape(-1).astype(jnp.float32)
    pp = per_pos.reshape(-1).astype(jnp.float32)
    trained = w > 0
    now = jnp.where(trained, pp / jnp.where(trained, w, 1.0), 0.0)
    p1 = prev1.reshape(-1).astype(jnp.float32)
    p2 = prev2.reshape(-1).astype(jnp.float32)
    c = count.reshape(-1).astype(jnp.float32)
    eligible = trained & (c >= 2.0) & jnp.isfinite(p1) & jnp.isfinite(p2)

    def rel(a, b):
        return jnp.abs(b - a) / jnp.maximum(a, floor)

    if ema_alpha > 0:
        ema_now = (1.0 - ema_alpha) * p2 + ema_alpha * rel(p1, now)
        flat = eligible & (ema_now < tol)
    else:
        flat = eligible & (rel(p2, p1) < tol) & (rel(p1, now) < tol)
    if reentry is Reentry.RISE:
        was = trained & (prev_dropped.reshape(-1) > 0.5) & jnp.isfinite(p1)
        flat = jnp.where(was, (now - p1) / jnp.maximum(p1, floor) < tol, flat)
    keep = jax.lax.stop_gradient(trained & ~flat)
    drop = trained & flat
    f = lambda m: psum(jnp.sum(m.astype(jnp.float32)))
    n_trained = jnp.maximum(f(trained), 1.0)
    loss_sum = psum(jnp.sum(jnp.where(keep, pp, 0.0)))
    w_kept = psum(jnp.sum(jnp.where(keep, w, 0.0)))
    if drop_weight > 0:
        loss_sum = loss_sum + drop_weight * psum(jnp.sum(jnp.where(drop, pp, 0.0)))
        w_kept = w_kept + drop_weight * psum(jnp.sum(jnp.where(drop, w, 0.0)))
    loss_all = psum(jnp.sum(pp))
    w_all = psum(jnp.sum(w))
    dropped_loss_sum = psum(jnp.sum(jnp.where(drop, pp, 0.0)))
    w_drop = psum(jnp.sum(jnp.where(drop, w, 0.0)))
    loss = jnp.where(w_kept > 0, loss_sum / jnp.where(w_kept > 0, w_kept, 1.0), 0.0)
    stats = {
        "track/kept_token_frac": f(keep) / n_trained,
        "track/dropped_frac": f(drop) / n_trained,
        "track/dropped_low_frac": f(drop & (now < ALREADY_LEARNED_LOSS)) / n_trained,
        "track/dropped_high_frac": f(drop & (now >= ALREADY_LEARNED_LOSS)) / n_trained,
        "track/eligible_frac": f(eligible) / n_trained,
        "track/loss_unfiltered": jnp.where(w_all > 0, loss_all / jnp.where(w_all > 0, w_all, 1.0), 0.0),
        "track/loss_dropped": jnp.where(w_drop > 0, dropped_loss_sum / jnp.where(w_drop > 0, w_drop, 1.0), 0.0),
        "track/trained_tokens": f(trained),
        "track/kept_tokens": f(keep),
    }
    stats = {k: jax.lax.stop_gradient(v) for k, v in stats.items()}
    assert tuple(stats) == TRACK_STAT_KEYS, (tuple(stats), TRACK_STAT_KEYS)
    return loss, stats
