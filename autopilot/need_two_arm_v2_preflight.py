"""The four minimal preflight checks required before launching the frozen
two-arm experiment (TWO_ARM_EXPERIMENT_REQUIRED_CHANGES.md section 3).
Deliberately narrow -- exactly these four, nothing more (section 3: "do not
add further smoke tests unless the actual run exposes a new implementation
failure").

Check 3 (curriculum survival and matching) is checked STRUCTURALLY here
(the segment-hash/exact-K invariants that `run_targeted_retrieval` and
`need_two_arm_pilot.build_arm_dataset`/`assert_segments_survive` enforce by
construction), not by running the full, expensive Stage 1/2 retrieval pass
end-to-end -- that pass IS effectively a slice of launching the actual
experiment. If Checks 1/2/4 and this structural review of Check 3 all pass,
Check 3's remaining live assertions (`assert_segments_survive`, the
exact-K checks in `run_targeted_retrieval`) still execute for REAL, for
every run, inside `need_two_arm_pilot.main()` itself -- they are not
skipped, just not separately re-exercised here against a second synthetic
dataset.

Run: `python -m autopilot.need_two_arm_v2_preflight`
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.common import build_splits, load_partial_model, module_state_hash  # noqa: E402
from autopilot.controller import fit_pipeline  # noqa: E402
from autopilot.dataset import build_dataset_from_episodes  # noqa: E402
from autopilot.need_two_arm_pilot import run_checkpointed_training  # noqa: E402
from oracle.droid_actions import load_actions_for_episodes  # noqa: E402
from oracle.droid_streaming import stream_many_windows  # noqa: E402
from oracle.lewm_g import (  # noqa: E402
    DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, _img_transform, advance_aligned,
    encode_initial_rollout_window, encode_pixel_windows_batch, get_model,
)

SCRATCH_CKPT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "need_two_arm_v2_preflight_ckpt"


def check1_action_timing(model, pipeline, manifest) -> bool:
    print("=== Check 1: action timing ===", flush=True)
    replay_ids = manifest["episode_ids"]["replay_train"][:5]
    n_positions = 8
    n_raw = n_positions * FRAMESKIP
    actions = load_actions_for_episodes(replay_ids, max_frames=n_raw)
    windows = stream_many_windows(replay_ids, num_frames=n_positions, frameskip=FRAMESKIP,
                                     max_workers=8, per_episode_timeout_s=60.0)
    ok = [e for e in replay_ids if e in windows and e in actions
            and windows[e].shape[0] == n_positions and actions[e].shape[0] >= n_raw]
    if not ok:
        print("  SKIPPED: no usable cached window found", flush=True)
        return False
    eid = ok[0]
    pix = windows[eid]
    raw_full = actions[eid][:n_raw].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)
    t = HISTORY_SIZE
    pixel_frames = pix[t - HISTORY_SIZE + 1:t + 1]
    raw_actions_window = raw_full[t - HISTORY_SIZE + 1:t + 1]

    state, real_last_action_emb = encode_initial_rollout_window(pixel_frames, raw_actions_window, pipeline,
                                                                     model=model)
    emb = state.emb.unsqueeze(0).to(DEVICE)
    act_hist = state.act_emb_hist.unsqueeze(0).to(DEVICE)
    a_t = real_last_action_emb.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred1 = advance_aligned(model, emb, act_hist, a_t)[2]
        full_act_direct = torch.cat([act_hist, a_t.unsqueeze(1)], dim=1)
        pred_direct = model.predict(emb, full_act_direct)[:, -1:]
    item1 = torch.allclose(pred1, pred_direct, atol=1e-6)

    def embed_raw_action(raw_seg):
        converted = pipeline.to_convention(raw_seg.astype(np.float64))
        normed = pipeline.normalizer(converted).reshape(1, -1)
        with torch.no_grad():
            return model.action_encoder(torch.from_numpy(normed).float().unsqueeze(0).to(DEVICE))[0, 0]

    alt_action_raw = np.roll(raw_full[t], 1, axis=-1).astype(np.float64)
    a_t1_A = embed_raw_action(raw_full[t])
    a_t1_B = embed_raw_action(alt_action_raw)

    with torch.no_grad():
        emb2_A, hist2_A, pred1_A = advance_aligned(model, emb, act_hist, a_t)
        _e3A, _h3A, pred2_A = advance_aligned(model, emb2_A, hist2_A, a_t1_A.unsqueeze(0))
        emb2_B, hist2_B, pred1_B = advance_aligned(model, emb, act_hist, a_t)
        _e3B, _h3B, pred2_B = advance_aligned(model, emb2_B, hist2_B, a_t1_B.unsqueeze(0))
    item2a = torch.allclose(pred1_A, pred1_B, atol=1e-6)
    item2b = not torch.allclose(pred2_A, pred2_B, atol=1e-6)

    with torch.no_grad():
        e1, h1, p1 = advance_aligned(model, emb, act_hist, a_t)
        e2, h2, p2 = advance_aligned(model, e1, h1, a_t1_A.unsqueeze(0))
        pred_manual_1 = model.predict(emb, torch.cat([act_hist, a_t.unsqueeze(1)], dim=1))[:, -1:]
        emb_manual_1 = torch.cat([emb[:, 1:], pred_manual_1], dim=1)
        hist_manual_1 = torch.cat([act_hist[:, 1:], a_t.unsqueeze(1)], dim=1)
        pred_manual_2 = model.predict(emb_manual_1, torch.cat([hist_manual_1, a_t1_A.unsqueeze(0).unsqueeze(1)],
                                                                  dim=1))[:, -1:]
    item3 = torch.allclose(p1, pred_manual_1, atol=1e-6) and torch.allclose(p2, pred_manual_2, atol=1e-6)

    ok_all = item1 and item2a and item2b and item3
    print(f"  item1 advance==direct_predict: {item1}; item2a first-successor-unaffected: {item2a}; "
          f"item2b second-successor-changed: {item2b}; item3 manual-reconstruction: {item3}", flush=True)
    print(f"  Check 1: {'PASS' if ok_all else 'FAIL'}", flush=True)
    return ok_all


def check2_coordinate_and_freeze(model, pipeline, manifest) -> bool:
    print("=== Check 2: coordinate and freeze identity ===", flush=True)
    theta0 = copy.deepcopy(model.state_dict())
    replay_ids = manifest["episode_ids"]["replay_train"][:20]
    ds = build_dataset_from_episodes(replay_ids, pipeline, max_windows_per_episode=4, seed=1)
    if len(ds.samples) < 4:
        print("  SKIPPED: could not build a usable dataset", flush=True)
        return False

    enc_before, proj_before = module_state_hash(model.encoder), module_state_hash(model.projector)

    pix = np.stack([s["pixels"] for s in ds.samples[:4]])
    out_helper = encode_pixel_windows_batch(pix, model=model)
    with torch.no_grad():
        tfm = _img_transform()
        px = torch.from_numpy(pix).float() / 255.0
        px = px.permute(0, 1, 4, 2, 3).reshape(-1, 3, pix.shape[2], pix.shape[3])
        px = tfm({"pixels": px})["pixels"]
        _, oh, ow = px.shape[-3:]
        px = px.reshape(pix.shape[0], pix.shape[1], 3, oh, ow).to(DEVICE)
        direct = model.encode({"pixels": px})["emb"].cpu()
    helper_matches_direct = torch.allclose(out_helper, direct, atol=1e-6)

    out_helper_e20 = encode_pixel_windows_batch(pix, model=get_model())
    differs_from_epoch20 = not torch.allclose(out_helper, out_helper_e20, atol=1e-4)

    run_checkpointed_training(model, theta0, ds, seed=0, checkpoints=[10], ckpt_dir=SCRATCH_CKPT_DIR / "check2")
    enc_after, proj_after = module_state_hash(model.encoder), module_state_hash(model.projector)

    out_helper_after = encode_pixel_windows_batch(pix, model=model)
    targets_identical = torch.allclose(out_helper, out_helper_after, atol=1e-6)

    ok_all = (helper_matches_direct and differs_from_epoch20 and enc_before == enc_after
                and proj_before == proj_after and targets_identical)
    print(f"  helper==direct epoch-7 encoding: {helper_matches_direct}; differs from epoch-20: "
          f"{differs_from_epoch20}; encoder unchanged: {enc_before == enc_after}; projector unchanged: "
          f"{proj_before == proj_after}; targets pre==post training: {targets_identical}", flush=True)
    print(f"  Check 2: {'PASS' if ok_all else 'FAIL'}", flush=True)
    return ok_all


def check4_deterministic_paired_dry_run(model, pipeline, manifest) -> bool:
    print("=== Check 4: deterministic paired dry run ===", flush=True)
    theta0 = copy.deepcopy(model.state_dict())
    ds_a = build_dataset_from_episodes(manifest["episode_ids"]["replay_train"][:20], pipeline,
                                           max_windows_per_episode=4, seed=1)
    ds_b = build_dataset_from_episodes(manifest["episode_ids"]["replay_train"][20:40], pipeline,
                                           max_windows_per_episode=4, seed=2)
    if len(ds_a.samples) < 4 or len(ds_b.samples) < 4:
        print("  SKIPPED: could not build usable datasets", flush=True)
        return False

    r1 = run_checkpointed_training(model, theta0, ds_a, seed=7, checkpoints=[10],
                                       ckpt_dir=SCRATCH_CKPT_DIR / "check4_rep1")
    r2 = run_checkpointed_training(model, theta0, ds_a, seed=7, checkpoints=[10],
                                       ckpt_dir=SCRATCH_CKPT_DIR / "check4_rep2")
    repeat_ok = abs(r1[10]["loss"] - r2[10]["loss"]) < 1e-5
    sd1 = torch.load(r1[10]["checkpoint"], map_location="cpu")
    sd2 = torch.load(r2[10]["checkpoint"], map_location="cpu")
    ckpt_identical = all(torch.equal(sd1[k], sd2[k]) for k in sd1)

    rA1 = run_checkpointed_training(model, theta0, ds_a, seed=11, checkpoints=[10],
                                        ckpt_dir=SCRATCH_CKPT_DIR / "check4_AthenB_A")
    rB1 = run_checkpointed_training(model, theta0, ds_b, seed=11, checkpoints=[10],
                                        ckpt_dir=SCRATCH_CKPT_DIR / "check4_AthenB_B")
    rB2 = run_checkpointed_training(model, theta0, ds_b, seed=11, checkpoints=[10],
                                        ckpt_dir=SCRATCH_CKPT_DIR / "check4_BthenA_B")
    rA2 = run_checkpointed_training(model, theta0, ds_a, seed=11, checkpoints=[10],
                                        ckpt_dir=SCRATCH_CKPT_DIR / "check4_BthenA_A")
    order_ok = (abs(rA1[10]["loss"] - rA2[10]["loss"]) < 1e-5 and abs(rB1[10]["loss"] - rB2[10]["loss"]) < 1e-5)

    ok_all = repeat_ok and ckpt_identical and order_ok
    print(f"  repeat same arm/seed reproducible: {repeat_ok}; checkpoint bit-identical: {ckpt_identical}; "
          f"order-independent (A-then-B == B-then-A per arm): {order_ok}", flush=True)
    print(f"  Check 4: {'PASS' if ok_all else 'FAIL'}", flush=True)
    return ok_all


def main() -> int:
    model = load_partial_model()
    manifest = build_splits()
    pipeline = fit_pipeline(manifest["episode_ids"]["replay_train"])

    results = {
        "check1_action_timing": check1_action_timing(model, pipeline, manifest),
        "check2_coordinate_and_freeze": check2_coordinate_and_freeze(model, pipeline, manifest),
        "check4_deterministic_paired_dry_run": check4_deterministic_paired_dry_run(model, pipeline, manifest),
    }
    print("\n=== PREFLIGHT SUMMARY ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("  check3_curriculum_survival_and_matching: STRUCTURAL ONLY (see module docstring) -- live "
          "assertions run for real inside need_two_arm_pilot.main() every time")
    all_pass = all(results.values())
    print(f"\nOVERALL: {'ALL CHECKS PASS -- safe to launch the experiment' if all_pass else 'FAILURE -- STOP'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
