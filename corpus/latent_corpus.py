"""Shared latent+symbol corpus builder, used by Phases T / S / T2 / R / G / K.

Every one of those phases needs the same thing: for a set of real DROID
episodes, the per-position current-frame latent z_t (NOT the 3-window) and
the k*=2 alphabet symbol s_t driving z_t -> z_{t+1}. Streaming is the
expensive part (HTTP-range video decode), so this builds once and caches
to .npz; all later phases hit the cache.

Chunked deliberately: a full (600 episodes x 80 positions) pixel stack is
~17 GB, so episodes are streamed in small batches, encoded to latents
immediately, and the pixels dropped. Only latents (N,T,192 float32) and
symbol indices (N,T-1 int8) are retained/persisted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate.acceptance import K_STAR, load_alphabet_feats, quantize_to_symbols
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_many_windows
from oracle.lewm_g import ActionPipeline, FRAMESKIP, RAW_ACTION_DIM, encode_pixel_windows_batch

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"


class Corpus:
    """latents: (N,T,D) float32 | symbols: (N,T-1) int8 index into symbol_names
    | episode_ids: (N,) int64 | raw_actions kept only if requested."""

    def __init__(self, latents, symbols, episode_ids, symbol_names, raw_actions=None):
        self.latents = latents
        self.symbols = symbols
        self.episode_ids = episode_ids
        self.symbol_names = list(symbol_names)
        self.raw_actions = raw_actions

    @property
    def n_episodes(self):
        return self.latents.shape[0]

    @property
    def n_positions(self):
        return self.latents.shape[1]

    @property
    def dim(self):
        return self.latents.shape[2]

    def flat_latents(self) -> np.ndarray:
        return self.latents.reshape(-1, self.dim)

    def symbol_seq(self, i) -> list[str]:
        return [self.symbol_names[k] for k in self.symbols[i]]


def build_corpus(
    episode_ids: list[int], n_positions: int, cache_name: str,
    chunk: int = 16, max_workers: int = 16, encode_bs: int = 4,
    keep_raw_actions: bool = False, force: bool = False, model=None,
) -> Corpus:
    """`model` defaults to None (encode_pixel_windows_batch's own epoch-20
    fallback), matching every existing caller's historical behavior. Add
    this explicitly (and pass a distinct `cache_name` + `force=True`) to
    build a corpus in a DIFFERENT checkpoint's coordinates -- e.g. the
    two-arm v2 pilot's epoch-7 Lever-0 recomputation
    (need_two_arm_pilot.build_v2_lever0_and_directions)."""
    cache = OUT_DIR / f"corpus_{cache_name}.npz"
    if cache.exists() and not force:
        blob = np.load(cache, allow_pickle=False)
        names = json.loads((OUT_DIR / f"corpus_{cache_name}_symbols.json").read_text())
        raw = blob["raw_actions"] if "raw_actions" in blob.files else None
        print(f"corpus[{cache_name}]: loaded {blob['latents'].shape} from cache", flush=True)
        return Corpus(blob["latents"], blob["symbols"], blob["episode_ids"], names, raw)

    n_raw = n_positions * FRAMESKIP
    print(f"corpus[{cache_name}]: loading actions for {len(episode_ids)} episodes...", flush=True)
    actions_all = load_actions_for_episodes(episode_ids, max_frames=n_raw)
    usable = [e for e in episode_ids if e in actions_all and actions_all[e].shape[0] >= n_raw]
    print(f"corpus[{cache_name}]: {len(usable)} episodes have >= {n_raw} raw actions", flush=True)
    if not usable:
        raise RuntimeError(f"corpus[{cache_name}]: no usable episodes")

    # ONE pipeline fit over the whole corpus -- per-chunk fits would give
    # chunk-dependent action conventions and therefore chunk-dependent symbols
    all_raw = np.concatenate([actions_all[e][:n_raw] for e in usable], axis=0)
    pipeline = ActionPipeline.fit(all_raw)
    feats, names = load_alphabet_feats(ALPHABET_PATH)

    lat, sym, kept, raws = [], [], [], []
    for i in range(0, len(usable), chunk):
        batch = usable[i:i + chunk]
        windows = stream_many_windows(batch, num_frames=n_positions, frameskip=FRAMESKIP,
                                        max_workers=max_workers, per_episode_timeout_s=90.0)
        ok = sorted(set(batch) & set(windows.keys()))
        ok = [e for e in ok if windows[e].shape[0] == n_positions]
        if not ok:
            print(f"  chunk {i // chunk}: 0/{len(batch)} usable, skipping", flush=True)
            continue
        px = np.stack([windows[e] for e in ok])
        emb = encode_pixel_windows_batch(px, batch_size=encode_bs, model=model).numpy().astype(np.float32)
        for j, e in enumerate(ok):
            conv = pipeline.to_convention(actions_all[e][:n_raw].astype(np.float64))
            s = quantize_to_symbols(conv, K_STAR, feats, names)  # length n_positions
            lat.append(emb[j])
            sym.append(np.array([names.index(x) for x in s[:n_positions - 1]], dtype=np.int8))
            kept.append(e)
            if keep_raw_actions:
                raws.append(actions_all[e][:n_raw].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM).astype(np.float32))
        del windows, px, emb
        print(f"  chunk {i // chunk}: +{len(ok)} episodes (total {len(kept)}/{len(usable)})", flush=True)

    if not lat:
        # Every episode failed to stream (observed: transient HF I/O errors
        # after sustained parallel fetching). Return an EMPTY corpus rather
        # than crashing in np.stack -- the caller decides whether an empty
        # scan is fatal, and a starved retrieval is a reportable result.
        print(f"corpus[{cache_name}]: 0 episodes usable -- returning empty corpus", flush=True)
        d = 192
        return Corpus(np.zeros((0, n_positions, d), np.float32),
                        np.zeros((0, n_positions - 1), np.int8),
                        np.zeros(0, np.int64), names, None)

    latents = np.stack(lat)
    symbols = np.stack(sym)
    eids = np.array(kept, dtype=np.int64)
    raw_arr = np.stack(raws) if keep_raw_actions and raws else None

    save = {"latents": latents, "symbols": symbols, "episode_ids": eids}
    if raw_arr is not None:
        save["raw_actions"] = raw_arr
    np.savez(cache, **save)
    (OUT_DIR / f"corpus_{cache_name}_symbols.json").write_text(json.dumps(names))
    print(f"corpus[{cache_name}]: built {latents.shape} -> {cache.name}", flush=True)
    return Corpus(latents, symbols, eids, names, raw_arr)


def train_ids() -> list[int]:
    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    return sorted(set(split["train_episode_ids"]) - reset_ids)


def test_ids() -> list[int]:
    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    return sorted(split["test_episode_ids"])
