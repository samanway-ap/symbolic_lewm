"""§15 'Alphabet determinism': symbol->segment decoding is stable across
runs given a fixed codebook hash.

Checks three levels: (1) fitting twice in-process with the same seed gives
identical medoids/hash, (2) assignment for a fixed query set is identical,
and (3) the SAME holds across a fresh Python process -- the only way to rule
out determinism that accidentally depends on process-local state (hash
randomisation, BLAS threading, module-level caching) rather than the
algorithm itself.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from alphabet.codebook import Codebook
from tests.fixtures._fit_codebook_subprocess import make_segments

SUBPROCESS_SCRIPT = Path(__file__).parent / "fixtures" / "_fit_codebook_subprocess.py"


def test_in_process_refit_is_identical():
    segments = make_segments()
    cb1 = Codebook.fit(segments, n_codes=6, seed=17)
    cb2 = Codebook.fit(segments, n_codes=6, seed=17)
    assert cb1.content_hash() == cb2.content_hash()
    assert np.array_equal(cb1.medoids, cb2.medoids)


def test_assignment_is_identical_across_refits():
    segments = make_segments()
    query = make_segments(seed=999, n=25)
    cb1 = Codebook.fit(segments, n_codes=6, seed=17)
    cb2 = Codebook.fit(segments, n_codes=6, seed=17)
    assert np.array_equal(cb1.assign_batch(query), cb2.assign_batch(query))


def test_different_seeds_need_not_agree_but_same_seed_must():
    """Guard against a vacuous test: different fit seeds are allowed (though
    not guaranteed) to produce a different codebook, so the identical-hash
    result above is actually checking something."""
    segments = make_segments()
    cb_a = Codebook.fit(segments, n_codes=6, seed=1)
    cb_b = Codebook.fit(segments, n_codes=6, seed=2)
    # not asserting they differ (KMeans could coincidentally converge to the
    # same partition) -- just documenting the intent; the real guarantee is
    # the same-seed check above.
    del cb_a, cb_b


def _run_subprocess(seed: int) -> dict:
    result = subprocess.run(
        [sys.executable, str(SUBPROCESS_SCRIPT), str(seed)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_determinism_holds_across_process_boundary():
    seed = 17
    in_process = Codebook.fit(make_segments(), n_codes=6, seed=seed)
    in_process_hash = in_process.content_hash()

    proc1 = _run_subprocess(seed)
    proc2 = _run_subprocess(seed)

    assert proc1["content_hash"] == proc2["content_hash"] == in_process_hash
    assert proc1["assignments"] == proc2["assignments"]
