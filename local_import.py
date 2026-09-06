"""Load a module from THIS repo by explicit file path.

Why this exists: `oracle/lewm_g.py` puts `C:\\Users\\Admin\\Projects\\le-wm`
on sys.path[0] so the trained model's own code (`utils`, `module`, `jepa`)
can be imported. le-wm ALSO contains a top-level `eval.py`, which then
shadows this repo's `eval/` PACKAGE for any module imported after that
point -- `import eval.stats` raises "'eval' is not a package".

Earlier phases only avoided this by accident of import order (they touched
`eval.*` before anything pulled in le-wm). Rather than depend on that, or
reorder sys.path and risk le-wm's own `utils`/`module` resolving to
something else, load our modules by path so the name never has to be
resolved through sys.path at all.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def load_local(module_name: str, relpath: str):
    """load_local("symbolic_eval_stats", "eval/stats.py") -> module"""
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _ROOT / relpath
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {relpath} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod
