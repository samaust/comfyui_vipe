# ComfyUI ViPE

ComfyUI nodes for running ViPE entirely in memory, saving its standard synchronized artifacts, loading manually imported datasets, and extracting RGB, categorical masks, metric depth, pinhole intrinsics, poses, and deterministic JSON metadata.

Requires Python 3.14 and the `tridi` branch of [`samaust/vipe`](https://github.com/samaust/vipe). For sibling-repository development, install ViPE editable into the ComfyUI environment:

```bash
python -m pip install -e ../vipe
```

## Nodes

- `VipeEstimate` runs a selected pinhole ViPE pipeline without writing files.
- `VipeSaveData` writes a uniquely numbered standard ViPE result beneath ComfyUI output.
- `VipeLoadData` reads a complete result only from ComfyUI input.
- `VipeExtractData` exposes RGB, exact categorical masks, metric depth, intrinsics, metadata JSON, and available mask IDs.
- `VipeSelectMasks` converts selected categorical IDs into a ComfyUI `MASK`; an empty selection includes every available ID.

## Manual save/load workflow

Run `VipeSaveData`, then copy the complete numbered result directory (for example `output/vipe/run_00001_/`) into the ComfyUI input directory. Refresh the browser or restart ComfyUI if needed, select that input-relative directory in `VipeLoadData`, and load it. Direct loading from output is intentionally unsupported.
