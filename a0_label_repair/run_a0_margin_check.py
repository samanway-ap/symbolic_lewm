"""The decisive test for lever 0: now that switch rate is in-band for the
residualized tessellation (all 20 bits, both splits -- see
a0_residualized_report.json), does an extracted FSM's fidelity actually beat
the persistence baseline? Reuses e0_degeneracy/e0b.py's
`tessellation_margins_with_ci` UNCHANGED -- it already takes a
`tess_params_path` parameter, and `residualized_tessellation.py` wrote
`tessellation_params_residualized.npz` in the exact U_m{m}/B_m{m} format that
function expects, so this is a pointer swap, not new extraction logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import build_corpus, test_ids
from e0_degeneracy.e0b import tessellation_margins_with_ci

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=False)
    result = tessellation_margins_with_ci(
        heldout, OUT_DIR / "tessellation_params_residualized.npz", m_values=(2, 4, 6, 8),
        separation_rule="SepSeq",
    )
    for m, mv in result.items():
        if "error" in mv:
            print(f"m={m}: {mv['error']}", flush=True)
            continue
        print(f"m={m}: n_states={mv['n_states']}", flush=True)
        for bit, v in mv["bits"].items():
            print(f"  {bit}: fidelity={v['mean_fidelity']:.4f} persistence={v['mean_persistence']:.4f} "
                  f"margin={v['margin_mean_diff']:+.4f} ci95=[{v['margin_ci95'][0]:+.4f},{v['margin_ci95'][1]:+.4f}] "
                  f"beats_persistence={v['beats_persistence_ci_excludes_0']}", flush=True)
    (OUT_DIR / "a0_residualized_margin_report.json").write_text(json.dumps(result, indent=2, default=float))
    print("\nwrote artifacts/a0_residualized_margin_report.json", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
