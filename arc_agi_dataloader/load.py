from pathlib import Path
import os
import json
from .classes import GridSample, EpisodicGridSample

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

def load_dataset(data_dir: Path | None = None, split: str = "training") -> list[EpisodicGridSample]:
    if type(data_dir) == str:
        data_dir = Path(data_dir)
    base_dir = data_dir if data_dir is not None else _default_data_root()
    if split == "training":
        files = load_filenames(base_dir / "training")
    elif split == "evaluation":
        files = load_filenames(base_dir / "evaluation")
    else:
        raise ValueError(f"Invalid split: {split}")
    return [load_data(file) for file in files]


