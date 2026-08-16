import json
from fractions import Fraction
from typing import Any

import torch

from vipe import VipeAnnotationData


def build_metadata_json(data: VipeAnnotationData) -> str:
    height, width = data.frames[0].size()
    rate = Fraction(data.fps).limit_denominator(100_000)
    payload = {
        "schema": "comfyui_vipe_metadata",
        "schema_version": 1,
        "sequence_name": data.sequence_name,
        "timing": {
            "fps": data.fps,
            "frame_count": len(data.frames),
            "frame_rate_denominator": rate.denominator,
            "frame_rate_numerator": rate.numerator,
        },
        "dimensions": {"height": height, "width": width},
        "camera_to_world": {
            "convention": "opencv_x_right_y_down_z_forward",
            "frame_indices": list(range(len(data.frames))),
            "matrix_layout": "row_major",
            "matrices": [matrix.reshape(-1).tolist() for matrix in data.poses],
        },
        "intrinsics": {
            "camera_models": list(data.camera_models),
            "frame_indices": list(range(len(data.frames))),
            "layout": ["fx", "fy", "cx", "cy"],
            "units": "pixels",
            "values": data.intrinsics.tolist(),
        },
        "mask_labels": {str(key): value for key, value in data.mask_labels.items()},
        "ba_residual": data.ba_residual,
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def extract_annotation(data: VipeAnnotationData) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, dict[str, Any], str, str]:
    masks = {"values": data.instance_ids, "labels": data.mask_labels}
    intrinsics = {
        "values": data.intrinsics,
        "frame_indices": torch.arange(len(data.frames), dtype=torch.int64),
        "camera_models": data.camera_models,
    }
    available = sorted(int(value) for value in torch.unique(data.instance_ids).tolist())
    return (
        data.rgb,
        masks,
        data.metric_depth.unsqueeze(-1),
        intrinsics,
        build_metadata_json(data),
        ",".join(str(value) for value in available),
    )


def select_masks(masks: dict[str, Any], selected_ids: str) -> torch.Tensor:
    values = masks.get("values")
    if not isinstance(values, torch.Tensor) or values.ndim != 3 or values.dtype != torch.uint8:
        raise ValueError("VIPE_MASKS values must be uint8 [N,H,W]")
    available = sorted(int(value) for value in torch.unique(values).tolist())
    text = selected_ids.strip()
    if text:
        parts = [part.strip() for part in text.split(",")]
        if any(not part for part in parts):
            raise ValueError("selected_ids must be a comma-separated list of integers")
        try:
            selected = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError("selected_ids must contain integers") from exc
        if len(selected) != len(set(selected)):
            raise ValueError("selected_ids must not contain duplicates")
        if any(value < 0 or value > 255 for value in selected):
            raise ValueError("selected_ids must be in 0..255")
        unavailable = sorted(set(selected) - set(available))
        if unavailable:
            raise ValueError(f"selected mask IDs are unavailable: {unavailable}")
    else:
        selected = available
    result = torch.zeros(values.shape, dtype=torch.bool, device=values.device)
    for value in selected:
        result |= values == value
    return result.to(torch.float32)
