import comfy.model_management
import torch

from vipe import VipeAnnotationData, annotation_from_stream, load_annotation_data, make_pipeline
from vipe.config import parse_typed_config
from vipe.streams import TensorVideoStream
from vipe.utils.device import configure_device

from ..utils.data import extract_annotation, select_masks
from ..utils.paths import dataset_signature, discover_input_datasets, resolve_input_dataset, save_dataset


class VipeEstimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 30.0, "min": 0.001, "max": 1000.0, "step": 0.001}),
            "pipeline_preset": (["default", "dav3", "lyra", "no_vda", "static_vda"],),
        }}

    RETURN_TYPES = ("VIPE_DATA",)
    RETURN_NAMES = ("vipe_data",)
    FUNCTION = "estimate"
    CATEGORY = "ViPE"

    def estimate(self, images: torch.Tensor, fps: float, pipeline_preset: str):
        device = comfy.model_management.get_torch_device()
        if device.type not in {"cpu", "cuda"}:
            raise ValueError(f"ViPE requires a CPU or CUDA device, got {device}")
        configure_device(device)
        config = parse_typed_config("default", [
            f"pipeline={pipeline_preset}",
            "pipeline.output.save_artifacts=false",
            "pipeline.output.save_viz=false",
            "pipeline.output.save_slam_map=false",
            "pipeline.output.skip_exists=false",
        ])
        pipeline = make_pipeline(config.pipeline)
        pipeline.return_output_streams = True
        result = pipeline.run(TensorVideoStream(images, fps, "comfyui"))
        if result.output_streams is None or len(result.output_streams) != 1:
            raise RuntimeError("ViPE did not return exactly one annotated stream")
        return (annotation_from_stream(result.output_streams[0], result.ba_residual),)


class VipeSaveData:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vipe_data": ("VIPE_DATA",),
            "filename_prefix": ("STRING", {"default": "vipe/run"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_relative_path",)
    FUNCTION = "save"
    CATEGORY = "ViPE"
    OUTPUT_NODE = True

    def save(self, vipe_data: VipeAnnotationData, filename_prefix: str):
        return (save_dataset(vipe_data, filename_prefix),)


class VipeLoadData:
    @classmethod
    def INPUT_TYPES(cls):
        choices = discover_input_datasets()
        return {"required": {"input_folder": (choices or [""],)}}

    RETURN_TYPES = ("VIPE_DATA",)
    RETURN_NAMES = ("vipe_data",)
    FUNCTION = "load"
    CATEGORY = "ViPE"

    @classmethod
    def IS_CHANGED(cls, input_folder: str):
        return dataset_signature(input_folder)

    @classmethod
    def VALIDATE_INPUTS(cls, input_folder: str):
        if input_folder not in discover_input_datasets():
            return "Select a valid ViPE dataset beneath the ComfyUI input directory"
        return True

    def load(self, input_folder: str):
        return (load_annotation_data(resolve_input_dataset(input_folder)),)


class VipeExtractData:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vipe_data": ("VIPE_DATA",)}}

    RETURN_TYPES = ("IMAGE", "VIPE_MASKS", "IMAGE", "VIPE_INTRINSICS", "STRING", "STRING")
    RETURN_NAMES = ("images", "masks", "depths", "intrinsics", "json_data", "available_ids")
    FUNCTION = "extract"
    CATEGORY = "ViPE"

    def extract(self, vipe_data: VipeAnnotationData):
        return extract_annotation(vipe_data)


class VipeSelectMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "masks": ("VIPE_MASKS",),
            "selected_ids": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "select"
    CATEGORY = "ViPE"

    def select(self, masks, selected_ids: str):
        return (select_masks(masks, selected_ids),)


NODE_CLASS_MAPPINGS = {
    "VipeEstimate": VipeEstimate,
    "VipeSaveData": VipeSaveData,
    "VipeLoadData": VipeLoadData,
    "VipeExtractData": VipeExtractData,
    "VipeSelectMasks": VipeSelectMasks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VipeEstimate": "ViPE - Estimate",
    "VipeSaveData": "ViPE - Save Data",
    "VipeLoadData": "ViPE - Load Data from Input",
    "VipeExtractData": "ViPE - Extract Data",
    "VipeSelectMasks": "ViPE - Select Masks",
}
