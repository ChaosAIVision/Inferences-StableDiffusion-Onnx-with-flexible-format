import argparse
import os
import shutil
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from packaging import version
from polygraphy.backend.onnx.loader import fold_constants
from torch.onnx import export

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from typing import Union, Optional, Tuple
import controlnet_hinter


is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")
is_torch_2_0_1 = version.parse(version.parse(torch.__version__).base_version) == version.parse("2.0.1")

class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


def optimize(onnx_graph, name, verbose):
    opt = Optimizer(onnx_graph, verbose=verbose)
    opt.info(name + ": original")
    opt.cleanup()
    opt.info(name + ": cleanup")
    opt.fold_constants()
    opt.info(name + ": fold constants")
    # opt.infer_shapes()
    # opt.info(name + ': shape inference')
    onnx_opt_graph = opt.cleanup(return_onnx=True)
    opt.info(name + ": finished")
    return onnx_opt_graph


class ControlNetModelCustom(torch.nn.Module):
    def __init__(self, controlnet ):
        super().__init__()
        self.controlnet = controlnet

    def forward(self, sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                controlnet_cond: torch.FloatTensor ):
        output_controlnet = self.controlnet(
            sample = sample,
            timestep = timestep,
            encoder_hidden_states = encoder_hidden_states,
            controlnet_cond = controlnet_cond,
            return_dict = False
        )

        return output_controlnet
        

        
def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    with torch.inference_mode(), torch.autocast("cuda"):
        if is_torch_less_than_1_11:
            export(
                model,
                model_args,
                f=output_path.as_posix(),
                input_names=ordered_input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                use_external_data_format=use_external_data_format,
                enable_onnx_checker=True,
                opset_version=opset,
            )
        else:
            export(
                model,
                model_args,
                f=output_path.as_posix(),
                input_names=ordered_input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                opset_version=opset,
            )

@torch.no_grad()
def convert_models(
    model_path:str,
    controlnet_path:str,
    control_net_hinter_type:str,
    output_path:str,
    opset:int,
    fp16: bool = False   
):
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = "cuda"
    elif fp16 and not torch.cuda.is_available():
        raise ValueError("`float16` model export is only supported on GPUs with CUDA")
    else:
        device = "cpu"

    #Get attribute controlnet
    hinter_controlnet = controlnet_hinter
    if hasattr(hinter_controlnet, control_net_hinter_type):
        hinter = getattr(hinter_controlnet, control_net_hinter_type)

    CONTROLNET_MAPPING = {

    "model_type": {
        "model_id": f"{controlnet_path}",
        "hinter": hinter
    },

}
    controlnet_type =  control_net_hinter_type
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MAPPING[controlnet_type]['model_id'], torch_dtype=dtype).to(device)
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_path, controlnet=controlnet, torch_dtype=dtype
        ).to(device)
     
     # # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    controlnet_model  = ControlNetModelCustom(controlnet)
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    img_size = 8 * unet_sample_size
    output_path = Path(output_path)
    controlnet_output_path = output_path / "controlnet" / "model.onnx"
    
    onnx_export(
        controlnet_model,
        model_args=(
        torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
        torch.tensor([1.0]).to(device=device, dtype=dtype),
        torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
        torch.randn(2,3, img_size, img_size ).to(device=device, dtype=dtype),
        ),
        output_path = controlnet_output_path,
        ordered_input_names=[
            "sample",
            "timestep",
            "encoder_hidden_states",
            'controlnet_cond'],
        output_names= ['output_controlnet'],
        dynamic_axes={
                "sample": {0: "2B", 2: "H", 3: "W"},
                "encoder_hidden_states": {0: "2B"},
                "encoder_hidden_states": {0: "2B"},  # Tensor encoder hidden states
                "controlnet_cond": {0:"2B", 2: "H", 3: "W"},
                },
         opset= opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
)
    controlnet_model_path = str(controlnet_output_path.absolute().as_posix())
    unet_dir = os.path.dirname(controlnet_model_path)
    
    # optimize onnx
    shape_inference.infer_shapes_path(controlnet_model_path, controlnet_model_path)
    unet_opt_graph = optimize(onnx.load(controlnet_model_path), name="controlnet", verbose=True)
    # clean up existing tensor files
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)
    # collate external tensor files into one
    onnx.save_model(
            unet_opt_graph,
            controlnet_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,)

    del pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_xl", action="store_true", default=False, help="SD XL pipeline")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument(
        "--controlnet_path",
        nargs="+",
        required=True,
        help="Path to the `controlnet` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")
    parser.add_argument("--control_net_hinter_type", type=str, required=True, help="controlnet hinter name.")


    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    args = parser.parse_args()

    convert_models(args.model_path, args.controlnet_path,args.control_net_hinter_type, args.output_path, args.opset, args.fp16)



