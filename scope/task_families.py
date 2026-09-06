"""Phase M0.5 -- scope selection (SYMBOLIC_ABSTRACTION_PLAN v2, §3).

Group episodes by (normalised task string[, scene id]), rank groups, and
select a single family satisfying:
  - >= min_episodes (200 by default)
  - a consistent scene (single camera viewpoint / workspace layout)
  - visible discrete structure (reach/grasp/lift/place/release), preferring
    families that exercise the gripper

This module is dataset-agnostic: it operates on a list of `EpisodeRecord`s,
not on lerobot/DROID directly, so the same grouping/ranking/report logic can
be reused if the data source changes (e.g. full DROID instead of droid_100).
The lerobot-specific extraction lives in
`scope/_extract_droid_episode_records.py`, run from whatever environment has
`lerobot` installed, producing the JSON this module consumes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EpisodeRecord:
    episode_id: int
    task: str            # normalised task string; "" means no annotation
    scene_key: str        # best available proxy for "consistent scene"; "" if unknown
    length: int           # frames
    action_spread: float  # e.g. mean per-dim std of the episode's raw actions
    involves_gripper: bool


@dataclass
class FamilyStats:
    key: tuple           # (task, scene_key)
    episode_ids: list
    n_episodes: int
    mean_length: float
    mean_action_spread: float
    involves_gripper_frac: float


def normalise_task(task: str) -> str:
    return " ".join(task.strip().lower().split())


def group_into_families(records: list[EpisodeRecord]) -> list[FamilyStats]:
    groups: dict[tuple, list[EpisodeRecord]] = {}
    for r in records:
        key = (normalise_task(r.task), r.scene_key)
        groups.setdefault(key, []).append(r)

    stats = []
    for key, members in groups.items():
        n = len(members)
        stats.append(
            FamilyStats(
                key=key,
                episode_ids=[m.episode_id for m in members],
                n_episodes=n,
                mean_length=sum(m.length for m in members) / n,
                mean_action_spread=sum(m.action_spread for m in members) / n,
                involves_gripper_frac=sum(m.involves_gripper for m in members) / n,
            )
        )
    return sorted(stats, key=lambda s: s.n_episodes, reverse=True)


def select_family(
    families: list[FamilyStats], min_episodes: int = 200
) -> FamilyStats | None:
    """§3.1 selection: >= min_episodes, non-empty task label (an empty task
    string is 'no annotation available', not a real task family), preferring
    gripper-involving families among qualifiers."""
    qualifying = [
        f for f in families if f.n_episodes >= min_episodes and f.key[0] != ""
    ]
    if not qualifying:
        return None
    qualifying.sort(key=lambda f: (-f.involves_gripper_frac, -f.n_episodes))
    return qualifying[0]


def write_full_droid_interim_report(out_dir: Path) -> None:
    """Interim status for the full lerobot/droid_1.0.1 investigation -- not
    yet a final decision, since scene/building consistency (§3.1's
    'consistent scene' requirement) cannot be checked without a further,
    materially larger download (the `building` column lives in the
    ~15.6GB `data/*.parquet` files, not the ~600MB `meta/episodes/*.parquet`
    already pulled). Written so the interim finding survives even if this
    session ends before the building-column check happens."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = """# Phase M0.5 -- full lerobot/droid_1.0.1 interim finding

**Status: PROMISING, NOT YET DECIDED.** Superseded the droid_100-only kill
result below once full DROID's metadata was checked.

## What was checked (free / cheap: ~600MB, meta/episodes/*.parquet only)

Exact-string task grouping on all 95,658 episodes: only 2 non-degenerate
task strings clear 200 episodes ("put the marker in the cup": 221, "put the
marker in the mug": 211) -- exact string match alone barely clears the bar,
same failure mode as droid_100 just at a different scale, because DROID's
crowdsourced instructions are highly free-form (49,505 distinct strings /
95,658 episodes).

Keyword-merged pseudo-families (grouping near-duplicate phrasings by
shared object/verb) tell a very different story:

| candidate family | n_episodes | mean_length (frames) | distinct phrasings merged |
|---|---|---|---|
| marker_in_container (marker + cup/mug/bowl) | 3893 | 217.7 | 1105 |
| drawer_open_close | 3184 | 228.2 | 1518 |
| towel | 4113 | 327.4 | 2593 |
| tap_open_close | 759 | 222.0 | 323 |
| microwave_open_close | 421 | 285.8 | 136 |

All five clear the 200-episode threshold with 2x-20x headroom. This is a
real, substantive candidate pool -- full DROID is NOT dataset-poor for
M0.5's purposes; droid_100 (the 100-episode curated subset actually used
for M0 and the earlier LeWM training) was simply too small and too
deliberately-diversified to ever contain a repeated task family.

## What's NOT yet checked

**Scene/workspace consistency** (§3.1: "a single camera viewpoint and
workspace layout"). `building` and `task_category` are real columns in
full DROID's schema but live in the per-frame `data/*.parquet` files
(~156 files, ~15.6GB total for all episodes), not in the lightweight
`meta/episodes/*.parquet` already downloaded. None of the candidates above
are confirmed single-scene; DROID is explicitly collected across 564 scenes
and 86 buildings, so a large keyword family could still fragment into many
small per-building cells once checked.

## Decision point

Confirming scene consistency requires downloading ~15.6GB (`data/` only,
still no `videos/` -- those alone would be ~400GB and are out of the
question at this stage). This is ~25x the metadata pull already done
autonomously in this session. Given the genuine uncertainty in the outcome
(a family with thousands of episodes could still fail the single-scene
requirement) and the size of the commitment, this was deliberately NOT
started without checking in first.
"""
    (out_dir / "scope_report_full_droid_interim.md").write_text(text, encoding="utf-8")


def write_scope_report(
    families: list[FamilyStats],
    chosen: FamilyStats | None,
    min_episodes: int,
    dataset_name: str,
    out_dir: Path,
    extra_notes: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase M0.5 -- scope selection report",
        "",
        f"Dataset: `{dataset_name}`  |  min_episodes threshold: {min_episodes}",
        "",
    ]
    if extra_notes:
        lines += [extra_notes, ""]

    lines += [
        "## Ranked candidate families (top 15 by episode count)",
        "",
        "| rank | task | scene_key | n_episodes | mean_length | mean_action_spread | gripper_frac |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, f in enumerate(families[:15], start=1):
        task_disp = f.key[0] if f.key[0] else "*(no annotation)*"
        scene_disp = f.key[1] if f.key[1] else "*(unknown)*"
        lines.append(
            f"| {i} | {task_disp} | {scene_disp} | {f.n_episodes} | "
            f"{f.mean_length:.1f} | {f.mean_action_spread:.4f} | {f.involves_gripper_frac:.2f} |"
        )

    lines += ["", "## Decision", ""]
    if chosen is None:
        lines += [
            f"**KILL CRITERION M0.5 TRIGGERED.** No family reaches {min_episodes} "
            "episodes in a consistent scene. Largest qualifying family: "
            f"{max((f.n_episodes for f in families if f.key[0] != ''), default=0)} episodes "
            "(excluding the unlabeled bucket, which is not a real task family).",
            "",
            "Per the plan: do not weaken the threshold. Report the distribution and halt.",
        ]
    else:
        lines += [
            f"**Selected:** task=`{chosen.key[0]}`, scene=`{chosen.key[1]}`, "
            f"{chosen.n_episodes} episodes.",
            "",
            f"Episode ids written to `scope_episode_ids.json`.",
        ]

    (out_dir / "scope_report.md").write_text("\n".join(lines), encoding="utf-8")

    if chosen is not None:
        (out_dir / "scope_episode_ids.json").write_text(
            json.dumps({"task": chosen.key[0], "scene": chosen.key[1],
                        "episode_ids": chosen.episode_ids}, indent=2),
            encoding="utf-8",
        )


def run_from_records_json(records_path: Path, out_dir: Path, min_episodes: int, dataset_name: str):
    raw = json.loads(Path(records_path).read_text(encoding="utf-8"))
    records = [EpisodeRecord(**r) for r in raw]
    families = group_into_families(records)
    chosen = select_family(families, min_episodes=min_episodes)
    write_scope_report(families, chosen, min_episodes, dataset_name, out_dir)
    return families, chosen
