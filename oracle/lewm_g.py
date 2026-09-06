"""Real g: wraps the trained, FROZEN LeWM checkpoint (theta_0 = lewm_droid_s0)
as the step_fn/label_fn pair LatentOracle needs.

Established facts from M0 (repo_audit.json), reused directly:
  - window_N = history_size = 3, causal predictor (jepa.py / module.py).
  - action_encoder.input_dim = 14 = frameskip(2) * raw_action_dim(7): one
    "model step" (one predict() call advancing the window by one position)
    consumes ONE 14-dim vector = two consecutive RAW actions concatenated.
  - The model never re-encodes pixels mid-rollout; JEPA.predict operates
    purely in latent+action-embedding space (jepa.py:47-55).

New for full DROID (M0.5's chosen family): droid_1.0.1's canonical `action`
field is 8-dim (does not match theta_0's 7-dim training data at all).
`action.original` is 7-dim, dimension-matched to droid_100's `action` that
theta_0 was actually trained on -- used here. Their VALUE RANGES do not
obviously match either (droid_100's `action` is clipped to ~[-1,1] with a
[0,1] gripper channel; `action.original`'s raw values run well outside that,
e.g. into the 3.0 range).

Action-convention guardrail (M1_report.md): every raw action from Phase B
onward is passed through `ActionPipeline`, which (1) empirically rescales
+ hard-clips `action.original` into droid_100's OWN observed per-dimension
range -- the convention theta_0 was actually trained on -- with the
post-conversion range asserted, THEN (2) z-scores using droid_100's own
fixed, baked-in statistics (not a per-sample fit). Phase B's codebook/
segments/medoids should use stage (1) only (`pipeline.to_convention`) --
a human-facing, droid_100-matched convention -- and stage (2) is added
only at the action_encoder input boundary (`pipeline(...)`, the full
composition). This is a mitigation, not a proof the two datasets' action
fields are the same physical quantity -- recorded as an open risk; M1's
L_max=36/17 is encouraging evidence, not confirmation.
"""
from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_LEWM_DIR = r"C:\Users\Admin\Projects\le-wm"
if _LEWM_DIR not in sys.path:
    sys.path.insert(0, _LEWM_DIR)
os.environ.setdefault("STABLEWM_HOME", str(Path(_LEWM_DIR) / ".stable-wm"))

HISTORY_SIZE = 3
FRAMESKIP = 2          # raw frames per model step (fixed by theta_0's training)
RAW_ACTION_DIM = 7      # action.original
EMBED_DIM = 192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_D100_ACTIONS_PATH = Path(_LEWM_DIR) / ".frame_cache" / "actions.npy"

_model = None


def get_model():
    global _model
    if _model is None:
        import warnings; warnings.filterwarnings("ignore")
        from stable_worldmodel.wm.utils import load_pretrained
        _model = load_pretrained("lewm_droid_s0/weights_epoch_20.pt").to(DEVICE).eval()
        _model.requires_grad_(False)
    return _model


@dataclass(frozen=True)
class LeWMWindowState:
    """zeta_t = (z_{t-2}, z_{t-1}, z_t) plus the action embeddings that were
    active at each of those 3 positions (needed to call predict() again)."""
    emb: torch.Tensor       # (HISTORY_SIZE, EMBED_DIM)
    act_emb: torch.Tensor   # (HISTORY_SIZE, EMBED_DIM)  (action_encoder output)


def _img_transform():
    from utils import get_img_preprocessor
    return get_img_preprocessor(source="pixels", target="pixels", img_size=224)


@functools.lru_cache(maxsize=1)
def _droid100_action_stats() -> dict[str, np.ndarray]:
    """droid_100's OWN observed per-dimension stats -- the convention (and,
    deliberately, the same un-fixed §2.3-leaky z-score statistics) theta_0
    was actually trained against."""
    d100 = np.load(_D100_ACTIONS_PATH).astype(np.float64)  # (32212, 7)
    return {
        "min": d100.min(0), "max": d100.max(0),
        "mean": d100.mean(0), "std": d100.std(0),
    }


class ActionConventionConverter:
    """Empirically rescales + hard-clips raw `action.original` values into
    droid_100's OWN observed per-dimension range. Fit once on a
    representative sample of the TARGET data's raw actions; destination
    range is always droid_100's fixed, measured range. Gripper channel
    (last dim) is separately clipped to exactly [0,1]. Range is asserted
    after conversion -- a silent range violation here would surface
    downstream as "a bad alphabet", not as what it actually is."""

    def __init__(self, src_lo: np.ndarray, src_hi: np.ndarray):
        self.src_lo = src_lo
        self.src_hi = src_hi
        stats = _droid100_action_stats()
        self.dst_lo = stats["min"]
        self.dst_hi = stats["max"]

    @classmethod
    def fit(cls, raw_action_sample: np.ndarray, pct: tuple[float, float] = (1.0, 99.0)) -> "ActionConventionConverter":
        lo = np.percentile(raw_action_sample, pct[0], axis=0)
        hi = np.percentile(raw_action_sample, pct[1], axis=0)
        return cls(lo, hi)

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        span_src = np.clip(self.src_hi - self.src_lo, 1e-6, None)
        span_dst = self.dst_hi - self.dst_lo
        out = self.dst_lo + (raw - self.src_lo) * (span_dst / span_src)
        out = np.clip(out, self.dst_lo, self.dst_hi)
        out[..., -1] = np.clip(out[..., -1], 0.0, 1.0)  # gripper channel, exact [0,1]
        eps = 1e-3
        assert out.min() >= self.dst_lo.min() - eps and out.max() <= self.dst_hi.max() + eps, (
            f"converted action escaped droid_100's asserted range: "
            f"min={out.min():.4f} max={out.max():.4f} "
            f"(expected within [{self.dst_lo.min():.4f}, {self.dst_hi.max():.4f}])"
        )
        return out


class ActionNormalizer:
    """z-score using droid_100's OWN fixed statistics (matching what
    theta_0 was actually trained against) -- NOT a per-sample fit."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = np.where(std < 1e-6, 1.0, std)

    @classmethod
    def from_droid100(cls) -> "ActionNormalizer":
        stats = _droid100_action_stats()
        return cls(stats["mean"], stats["std"])

    def __call__(self, actions_in_droid100_convention: np.ndarray) -> np.ndarray:
        return (actions_in_droid100_convention - self.mean) / self.std


class ActionPipeline:
    """Composes ActionConventionConverter -> ActionNormalizer. Use
    `.to_convention(raw)` (stage 1 only) for anything human-facing --
    segments, codebook, medoids (Phase B). Use the full call `pipeline(raw)`
    only at the action_encoder input boundary (oracle rollouts, coverage
    validation) -- that is the only place z-scoring belongs."""

    def __init__(self, converter: ActionConventionConverter, normalizer: ActionNormalizer | None = None):
        self.converter = converter
        self.normalizer = normalizer or ActionNormalizer.from_droid100()

    @classmethod
    def fit(cls, raw_action_sample: np.ndarray) -> "ActionPipeline":
        return cls(ActionConventionConverter.fit(raw_action_sample))

    def to_convention(self, raw: np.ndarray) -> np.ndarray:
        return self.converter(raw)

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return self.normalizer(self.converter(raw))


def encode_initial_window(
    pixel_frames: np.ndarray,        # (HISTORY_SIZE, H, W, 3) uint8, subsampled by FRAMESKIP
    raw_actions: np.ndarray,         # (HISTORY_SIZE, FRAMESKIP, RAW_ACTION_DIM) float, per-position action pairs
    pipeline: "ActionPipeline",
) -> LeWMWindowState:
    """Build a WindowState from real episode data (used for reset symbols,
    A.2, and for seeding rollouts during L_max measurement)."""
    model = get_model()
    tfm = _img_transform()

    px = torch.from_numpy(pixel_frames).float() / 255.0          # (H,h,w,3)
    px = px.permute(0, 3, 1, 2)                                    # (H,3,h,w)
    px = tfm({"pixels": px})["pixels"].unsqueeze(0)                # (1,H,3,224,224)

    # convert+normalize each raw 7-dim action BEFORE concatenating pairs --
    # the pipeline is fit per physical action dimension, not per 14-dim chunk
    act_norm = pipeline(raw_actions.reshape(-1, RAW_ACTION_DIM))  # (H*FRAMESKIP, 7)
    act_flat = act_norm.reshape(HISTORY_SIZE, FRAMESKIP * RAW_ACTION_DIM)
    act = torch.from_numpy(act_flat).float().unsqueeze(0)           # (1,H,14)

    with torch.no_grad():
        info = model.encode({"pixels": px.to(DEVICE), "action": act.to(DEVICE)})
    return LeWMWindowState(emb=info["emb"][0].cpu(), act_emb=info["act_emb"][0].cpu())


def encode_pixel_windows_batch(pixel_windows: np.ndarray, batch_size: int = 32) -> torch.Tensor:
    """pixel_windows: (B, H, h, w, 3) uint8 -> (B, H, EMBED_DIM). Pixels-only
    (no action_encoder call) -- used for the k-medoids visual-diversity
    diagnostic (Phase A.2), which is about scene/visual structure, not
    action semantics, and for L_max's ground-truth encoding of real frames."""
    model = get_model()
    tfm = _img_transform()
    B, H = pixel_windows.shape[:2]
    out = []
    with torch.no_grad():
        for i in range(0, B, batch_size):
            chunk = pixel_windows[i:i + batch_size]
            b = chunk.shape[0]
            px = torch.from_numpy(chunk).float() / 255.0          # (b,H,h,w,3)
            px = px.permute(0, 1, 4, 2, 3).reshape(-1, 3, chunk.shape[2], chunk.shape[3])
            px = tfm({"pixels": px})["pixels"]
            # Resize is aspect-preserving, NOT a fixed 224x224 -- read the
            # actual output height/width back rather than assuming square
            # (this exact assumption broke masking earlier in the session).
            _, out_h, out_w = px.shape[-3:]
            px = px.reshape(b, H, 3, out_h, out_w).to(DEVICE)
            emb = model.encode({"pixels": px})["emb"]              # (b,H,D)
            out.append(emb.cpu())
    return torch.cat(out, dim=0)


def rollout_batch(
    init_emb: torch.Tensor,        # (B, HISTORY_SIZE, D)
    init_act_emb: torch.Tensor,    # (B, HISTORY_SIZE, D)
    raw_action_steps: np.ndarray,  # (B, n_model_steps, FRAMESKIP, RAW_ACTION_DIM)
    pipeline: "ActionPipeline",
) -> torch.Tensor:
    """Vectorised autoregressive rollout across a whole batch of episodes at
    once, at the model's OWN native step granularity (one model step =
    FRAMESKIP raw frames). Returns (B, n_model_steps, D): the predicted
    latent after each model step. Computed ONCE at the finest granularity;
    the k-sweep (A.5) re-samples this same trace at stride k/FRAMESKIP for
    each k, rather than re-running the rollout per k -- the trajectory
    itself (using the REAL recorded actions, not an alphabet) does not
    depend on k at all; only which positions we score agreement at does.
    """
    model = get_model()
    B, n_steps = raw_action_steps.shape[:2]
    emb = init_emb.clone().to(DEVICE)
    act_emb = init_act_emb.clone().to(DEVICE)
    trace = []
    with torch.no_grad():
        for s in range(n_steps):
            chunk = raw_action_steps[:, s]  # (B,FRAMESKIP,7)
            norm = pipeline(chunk.reshape(-1, RAW_ACTION_DIM)).reshape(B, FRAMESKIP * RAW_ACTION_DIM)
            a = torch.from_numpy(norm).float().unsqueeze(1).to(DEVICE)  # (B,1,14)
            new_act_emb = model.action_encoder(a)[:, 0]                  # (B,D)
            pred = model.predict(emb, act_emb)[:, -1]                     # (B,D)
            trace.append(pred.cpu())
            emb = torch.cat([emb[:, 1:], pred.unsqueeze(1)], dim=1)
            act_emb = torch.cat([act_emb[:, 1:], new_act_emb.unsqueeze(1)], dim=1)
    return torch.stack(trace, dim=1)  # (B, n_steps, D)


def encode_action_embeddings_batch(
    raw_action_windows: np.ndarray,  # (B, positions, FRAMESKIP, RAW_ACTION_DIM)
    pipeline: "ActionPipeline",
    batch_size: int = 64,
) -> torch.Tensor:
    model = get_model()
    B, P = raw_action_windows.shape[:2]
    out = []
    with torch.no_grad():
        for i in range(0, B, batch_size):
            chunk = raw_action_windows[i:i + batch_size]
            b = chunk.shape[0]
            norm = pipeline(chunk.reshape(-1, RAW_ACTION_DIM)).reshape(b, P, FRAMESKIP * RAW_ACTION_DIM)
            a = torch.from_numpy(norm).float().to(DEVICE)
            emb = model.action_encoder(a)
            out.append(emb.cpu())
    return torch.cat(out, dim=0)


def rollout_batch_from_convention(
    init_emb: torch.Tensor,        # (B, HISTORY_SIZE, D)
    init_act_emb: torch.Tensor,    # (B, HISTORY_SIZE, D)
    raw_action_steps_converted: np.ndarray,  # (B, n_model_steps, FRAMESKIP, RAW_ACTION_DIM), ALREADY droid_100 convention
    pipeline: "ActionPipeline",
    model=None,
) -> torch.Tensor:
    """Sibling of `rollout_batch`, for when the action steps are ALREADY in
    droid_100 convention (e.g. alphabet medoid segments) rather than raw
    `action.original` -- only the z-score normalizer applies, matching
    `oracle/production_oracle.py`'s `_step_fn_from_convention` (which is
    per-window; this is the batched analogue, needed for e.g. E0's 500-word
    rollout-collapse probe). Passing `raw_action_steps_converted` through the
    FULL pipeline here would double-convert and silently corrupt the action
    scale -- this function exists specifically so that mistake isn't made at
    each new call site."""
    model = model or get_model()
    B, n_steps = raw_action_steps_converted.shape[:2]
    emb = init_emb.clone().to(DEVICE)
    act_emb = init_act_emb.clone().to(DEVICE)
    trace = []
    with torch.no_grad():
        for s in range(n_steps):
            chunk = raw_action_steps_converted[:, s]
            normed = pipeline.normalizer(chunk.reshape(-1, RAW_ACTION_DIM)).reshape(B, FRAMESKIP * RAW_ACTION_DIM)
            a = torch.from_numpy(normed).float().unsqueeze(1).to(DEVICE)
            new_act_emb = model.action_encoder(a)[:, 0]
            pred = model.predict(emb, act_emb)[:, -1]
            trace.append(pred.cpu())
            emb = torch.cat([emb[:, 1:], pred.unsqueeze(1)], dim=1)
            act_emb = torch.cat([act_emb[:, 1:], new_act_emb.unsqueeze(1)], dim=1)
    return torch.stack(trace, dim=1)  # (B, n_steps, D)


def step_fn_factory(pipeline: "ActionPipeline"):
    """Returns a step_fn(window, raw_action_segment) -> window for
    LatentOracle, closing over the fitted action pipeline. One call
    advances the window by len(raw_action_segment) raw frames, i.e.
    len(raw_action_segment) // FRAMESKIP internal model steps."""
    model = get_model()

    def step_fn(window: LeWMWindowState, raw_action_segment: np.ndarray) -> LeWMWindowState:
        n_raw = raw_action_segment.shape[0]
        assert n_raw % FRAMESKIP == 0, f"segment length {n_raw} not a multiple of FRAMESKIP={FRAMESKIP}"
        n_model_steps = n_raw // FRAMESKIP

        emb = window.emb.unsqueeze(0).to(DEVICE)          # (1,H,D)
        act_emb = window.act_emb.unsqueeze(0).to(DEVICE)  # (1,H,D)

        with torch.no_grad():
            for s in range(n_model_steps):
                chunk_raw = raw_action_segment[s * FRAMESKIP:(s + 1) * FRAMESKIP]  # (FRAMESKIP,7)
                chunk = pipeline(chunk_raw).reshape(1, -1)                          # (1,14)
                a = torch.from_numpy(chunk).float().unsqueeze(0).to(DEVICE)  # (1,1,14)
                new_act_emb = model.action_encoder(a)[:, 0]                   # (1,D)

                pred = model.predict(emb, act_emb)[:, -1:]                    # (1,1,D)
                emb = torch.cat([emb[:, 1:], pred], dim=1)
                act_emb = torch.cat([act_emb[:, 1:], new_act_emb.unsqueeze(1)], dim=1)

        return LeWMWindowState(emb=emb[0].cpu(), act_emb=act_emb[0].cpu())

    return step_fn
