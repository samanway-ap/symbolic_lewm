"""Prefix trie cache for LatentOracle (SYMBOLIC_ABSTRACTION_PLAN.md, Phase A.3).

TTT/L#-style learners issue prefix-closed query sets; a query for `u` should
reuse the rollout state of its longest cached prefix instead of replaying
from the reset. This trie stores, at each node, the state reached by the
word spelled out by the path from the root to that node.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Hashable

_UNSET = object()  # sentinel: a cached state could legitimately be None-ish


class _Node:
    __slots__ = ("state", "children")

    def __init__(self) -> None:
        self.state: Any = _UNSET
        self.children: dict[Hashable, "_Node"] = {}


class PrefixTrie:
    def __init__(self) -> None:
        self._root = _Node()
        self._hits = 0
        self._misses = 0

    def insert(self, word: tuple, state: Any) -> None:
        node = self._root
        for sym in word:
            node = node.children.setdefault(sym, _Node())
        node.state = state

    def longest_cached_prefix(self, word: tuple) -> tuple[Any, int]:
        """Returns (state, depth) for the deepest cached ancestor of `word`
        (depth = number of symbols consumed to reach it). depth=0, state=None
        if not even the empty prefix is cached (which is always true only
        before the very first insert)."""
        node = self._root
        best_state, best_depth = None, 0
        for depth, sym in enumerate(word, start=1):
            child = node.children.get(sym)
            if child is None:
                break
            node = child
            if node.state is not _UNSET:
                best_state, best_depth = node.state, depth
        if best_depth > 0:
            self._hits += 1
        else:
            self._misses += 1
        return best_state, best_depth

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def __len__(self) -> int:
        count = 0
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.state is not _UNSET:
                count += 1
            stack.extend(node.children.values())
        return count

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._root, f)

    @classmethod
    def load(cls, path: str | Path) -> "PrefixTrie":
        trie = cls()
        with open(path, "rb") as f:
            trie._root = pickle.load(f)
        return trie
