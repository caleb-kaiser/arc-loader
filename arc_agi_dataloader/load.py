from pathlib import Path
import os
import json
from typing import List
import numpy as np
from .classes import GridSample, EpisodicGridSample
from .augment import dihedral_transform, get_permutation_grid

def load_filenames(data_dir: Path) -> list[Path]:
    training_files = list(data_dir.glob("*.json"))
    return training_files

def load_data(file: Path) -> dict:
    with open(file, "r") as f:
        data = json.load(f)
        return EpisodicGridSample(
            train=[GridSample(input=sample["input"], output=sample["output"]) for sample in data["train"]],
            test=[GridSample(input=sample["input"], output=sample["output"]) for sample in data["test"]]
        )

def _default_data_root() -> Path:
    env_override = os.getenv("ARC_AGI_DATA_DIR")
    if env_override:
        return Path(env_override)
    # When installed: include externals at the distribution root next to the package
    package_dir = Path(__file__).resolve().parent
    installed_candidate = package_dir.parent / "externals" / "ARC-AGI" / "data"
    if installed_candidate.exists():
        return installed_candidate
    # Dev fallback: project root externals directory
    dev_candidate = package_dir.parent / "externals" / "ARC-AGI" / "data"
    return dev_candidate

def _apply_dihedral_to_grid(grid: List[List[int]], transform_id: int) -> List[List[int]]:
    arr = np.asarray(grid, dtype=np.uint8)
    transformed = dihedral_transform(arr, transform_id)
    return transformed.tolist()

def _augment_episode_by_tid(episode: EpisodicGridSample, transform_id: int) -> EpisodicGridSample:
    augmented_train = [
        GridSample(
            input=_apply_dihedral_to_grid(sample.input, transform_id),
            output=_apply_dihedral_to_grid(sample.output, transform_id),
        )
        for sample in episode.train
    ]
    augmented_test = [
        GridSample(
            input=_apply_dihedral_to_grid(sample.input, transform_id),
            output=_apply_dihedral_to_grid(sample.output, transform_id),
        )
        for sample in episode.test
    ]
    return EpisodicGridSample(train=augmented_train, test=augmented_test)

def _augment_episode_all(episode: EpisodicGridSample) -> list[EpisodicGridSample]:
    # Generate seven additional augmented puzzles (exclude identity transform 0)
    return [_augment_episode_by_tid(episode, tid) for tid in range(1, 8)]

def _apply_color_permutation_to_grid(grid: List[List[int]], mapping: np.ndarray) -> List[List[int]]:
    arr = np.asarray(grid, dtype=np.uint8)
    result = arr.copy()
    max_idx = mapping.shape[0]
    mask = arr < max_idx
    result[mask] = mapping[arr[mask]]
    return result.tolist()

def _augment_episode_with_color_permutation(episode: EpisodicGridSample, mapping: np.ndarray) -> EpisodicGridSample:
    augmented_train = [
        GridSample(
            input=_apply_color_permutation_to_grid(sample.input, mapping),
            output=_apply_color_permutation_to_grid(sample.output, mapping),
        )
        for sample in episode.train
    ]
    augmented_test = [
        GridSample(
            input=_apply_color_permutation_to_grid(sample.input, mapping),
            output=_apply_color_permutation_to_grid(sample.output, mapping),
        )
        for sample in episode.test
    ]
    return EpisodicGridSample(train=augmented_train, test=augmented_test)

def _augment_episode_colors(episode: EpisodicGridSample, num_permutations: int) -> list[EpisodicGridSample]:
    augmented: list[EpisodicGridSample] = []
    for _ in range(num_permutations):
        mapping = get_permutation_grid()
        augmented.append(_augment_episode_with_color_permutation(episode, mapping))
    return augmented

def _augment_episode_dihedral_and_colors(episode: EpisodicGridSample, color_permutations: int, include_dihedral_plain: bool) -> list[EpisodicGridSample]:
    augmented: list[EpisodicGridSample] = []
    for tid in range(0, 8):
        # Apply dihedral transform (identity for tid=0)
        dihedral_episode = episode if tid == 0 else _augment_episode_by_tid(episode, tid)
        # Optionally include the dihedral-only episode (avoid duplicating identity)
        if include_dihedral_plain and tid != 0:
            augmented.append(dihedral_episode)
        # Apply N color permutations to this dihedral orientation
        if color_permutations and color_permutations > 0:
            for _ in range(color_permutations):
                mapping = get_permutation_grid()
                augmented.append(_augment_episode_with_color_permutation(dihedral_episode, mapping))
    return augmented

def load_dataset(data_dir: Path | None = None, split: str = "training", augment: bool = True, color_permutations: int = 1) -> list[EpisodicGridSample]:
    if type(data_dir) == str:
        data_dir = Path(data_dir)
    base_dir = data_dir if data_dir is not None else _default_data_root()
    if split == "training":
        files = load_filenames(base_dir / "training")
    elif split == "evaluation":
        files = load_filenames(base_dir / "evaluation")
    else:
        raise ValueError(f"Invalid split: {split}")
    episodes: list[EpisodicGridSample] = []
    for file in files:
        episode = load_data(file)
        episodes.append(episode)
        if augment:
            # For maximum coverage: include dihedral-only (tid 1..7) and color permutations for all tids (0..7)
            episodes.extend(_augment_episode_dihedral_and_colors(episode, color_permutations, include_dihedral_plain=True))
        else:
            # No dihedral transforms; only color permutations on the original orientation
            if color_permutations and color_permutations > 0:
                episodes.extend(_augment_episode_colors(episode, color_permutations))
    return episodes


