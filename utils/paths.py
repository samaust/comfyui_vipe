import hashlib
from dataclasses import replace
from pathlib import Path

import folder_paths

from vipe import VipeAnnotationData, save_annotation_data
from vipe.utils.io import ArtifactPath


def _contained_path(root: str | Path, relative: str) -> Path:
    root_path = Path(root).resolve()
    candidate_text = Path(relative)
    if candidate_text.is_absolute() or any(part == ".." for part in candidate_text.parts):
        raise ValueError("dataset path must be input-relative and may not traverse directories")
    candidate = (root_path / candidate_text).resolve(strict=True)
    if not candidate.is_dir() or not candidate.is_relative_to(root_path):
        raise ValueError("dataset path escapes the ComfyUI input directory")
    return candidate


def _artifact_paths(dataset: Path) -> tuple[Path, ...]:
    rgb = sorted((dataset / "rgb").glob("*.mp4"))
    if len(rgb) != 1:
        raise ValueError(f"expected exactly one ViPE RGB sequence, found {len(rgb)}")
    artifact = ArtifactPath(dataset, rgb[0].stem)
    return (
        artifact.rgb_path,
        artifact.mask_path,
        artifact.mask_phrase_path,
        artifact.depth_path,
        artifact.pose_path,
        artifact.intrinsics_path,
        artifact.camera_type_path,
        artifact.meta_info_path,
    )


def _validate_artifact_containment(dataset: Path, input_root: Path) -> None:
    for path in _artifact_paths(dataset):
        if not path.exists():
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"cannot resolve ViPE artifact {path.name}") from exc
        if not resolved.is_relative_to(input_root):
            raise ValueError(f"ViPE artifact {path.name} escapes the ComfyUI input directory")


def discover_input_datasets(input_root: str | Path | None = None) -> list[str]:
    root = Path(input_root or folder_paths.get_input_directory()).resolve()
    datasets: set[str] = set()
    if not root.is_dir():
        return []
    for rgb_dir in root.rglob("rgb"):
        rgb_files = list(rgb_dir.glob("*.mp4")) if rgb_dir.is_dir() else []
        if len(rgb_files) != 1:
            continue
        dataset = rgb_dir.parent
        try:
            resolved = dataset.resolve(strict=True)
        except OSError:
            continue
        artifact_paths = _artifact_paths(resolved)
        required = (*artifact_paths[:2], *artifact_paths[3:])
        try:
            _validate_artifact_containment(resolved, root)
        except ValueError:
            continue
        if resolved.is_relative_to(root) and all(path.is_file() for path in required):
            datasets.add(dataset.relative_to(root).as_posix())
    return sorted(datasets)


def resolve_input_dataset(relative: str, input_root: str | Path | None = None) -> Path:
    root = Path(input_root or folder_paths.get_input_directory()).resolve()
    dataset = _contained_path(root, relative)
    _validate_artifact_containment(dataset, root)
    output = Path(folder_paths.get_output_directory()).resolve()
    if dataset.is_relative_to(output):
        raise ValueError("VipeLoadData does not load datasets from the ComfyUI output directory")
    return dataset


def dataset_signature(relative: str, input_root: str | Path | None = None) -> str:
    try:
        dataset = resolve_input_dataset(relative, input_root)
        paths = _artifact_paths(dataset)
    except ValueError:
        return "invalid"
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(dataset).as_posix().encode())
        try:
            stat = path.stat()
        except OSError:
            digest.update(b"missing")
        else:
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def save_dataset(data: VipeAnnotationData, filename_prefix: str) -> str:
    output_root = Path(folder_paths.get_output_directory()).resolve()
    folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, str(output_root))
    folder_path = Path(folder).resolve()
    if not folder_path.is_relative_to(output_root):
        raise ValueError("filename_prefix escapes the ComfyUI output directory")
    dataset_name = f"{filename}_{counter:05}_"
    dataset = folder_path / dataset_name
    while dataset.exists():
        counter += 1
        dataset_name = f"{filename}_{counter:05}_"
        dataset = folder_path / dataset_name
    saved = replace(data, sequence_name=f"{filename}_{counter:05}")
    save_annotation_data(saved, dataset)
    relative = dataset.relative_to(output_root).as_posix()
    expected_subfolder = Path(subfolder).as_posix().strip(".")
    if expected_subfolder and not relative.startswith(expected_subfolder):
        raise RuntimeError("ComfyUI save resolver returned an inconsistent subfolder")
    return relative
