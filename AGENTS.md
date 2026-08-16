# Development Guide

## Scope and structure

Keep ComfyUI node classes thin. Put reusable inference, annotation, metadata, and filesystem logic in `utils/`; node registration belongs in `nodes/nodes.py`. Public node names use the `Vipe` prefix and every class/display mapping must remain symmetric.

## ViPE data contracts

- `VIPE_DATA` is `vipe.VipeAnnotationData` and remains CPU resident between nodes.
- `VIPE_MASKS` is a dictionary containing exact `torch.uint8 [N,H,W]` instance IDs and integer-to-text labels.
- `VIPE_INTRINSICS` is a dictionary containing `float32 [N,4]` pinhole intrinsics, frame indices, and camera model names.
- Never replace intrinsics with field-of-view approximations. Only pinhole ViPE data is supported.

## ComfyUI integration

Use ComfyUI `IMAGE` tensors as float RGB `[N,H,W,3]` in `[0,1]` and `MASK` tensors as float `[N,H,W]`. Derive inference device selection from ComfyUI. Do not write files from estimation or extraction nodes.

All loading must be contained beneath the resolved ComfyUI input directory. Reject absolute paths, traversal, escaping symlinks, malformed datasets, and direct output-directory loading. Saving must use ComfyUI's save-path resolver and create a uniquely numbered dataset beneath the output root.

## Tests and documentation

Add unit coverage under `tests/` for contracts, registry symmetry, filesystem containment, deterministic metadata, and node shapes/dtypes. Stub optional ComfyUI or inference dependencies where practical. Update `README.md` whenever a node contract or manual workflow changes. Run compile checks, the unit suite, and `git diff --check` before handoff.
