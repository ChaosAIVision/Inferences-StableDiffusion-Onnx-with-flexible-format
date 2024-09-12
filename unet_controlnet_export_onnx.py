import warnings
import argparse
import os 
import shutil
from pathlib import Path
import torch
from torch.onnx import export 
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from diffusers import StableDiffusionPipeline
from unet_2d_condition_cnet import UNet2DConditionModel_Cnet
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline
from onnxruntime.transformers.onnx_model_unet import UnetOnnxModel


warnings.filterwarnings('ignore', '.*will be truncated.*')
warnings.filterwarnings('ignore', '.*The shape inference of prim::Constant type is missing.*')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the `diffusers` model to convert (either a local directory or on the Hub)."
)

parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to the output model."
)

parser.add_argument(
    "--opset",
    default=15,
    type=int,
    help="The version of the ONNX operator set to use.",
)

parser.add_argument(
    "--fp16",
    action="store_true",
    help="Export UNET in mixed `float16` mode"
)


class UnetOnnxModelDML(UnetOnnxModel):
    def __init__(self, model: onnx.ModelProto):
        self.model = model

    def optimize(self, enable_shape_inference=False):
            if not enable_shape_inference:
                self.disable_shape_inference()
                self.fuse_layer_norm()
                self.preprocess()
                self.postprocess() 

def onnx_export(
          model, 
          model_args: tuple,
          output_path: Path,
          ordered_input_names,
          output_names,
          dynamic_axes,
          opset,
):
    output_path.parent.mkdir(parents=True, exist_ok= True) 
    export(
         model,
         model_args,
         f= output_path.as_posix(),
         input_name =ordered_input_names,
         output_names= output_names,
         dynamic_axes = dynamic_axes,
         do_constant_folding= True,
         opset_version = opset,)
    
@torch.no_grad()
def convert_to_fp16(model_path):
     model_dir = os.path.dirname(model_path)
     onnx.shape_inference.infer_shapes_path(model_path)
     fp16_model = onnx.load(model_path)
     fp16_model = convert_float_to_float16(fp16_model, keep_io_types=True, disable_shape_infer= True)
     shutil.rmtree(model_dir)
     os.mkdir(model_dir)
     onnx.save(fp16_model, model_path)

@torch.no_grad()
def convert_unet(pipeline: StableDiffusionPipeline, output_path: str, opset: int,fp16: bool, device , dtype):
         # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    output_path =Path(output_path)
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / "model.onnx"
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
            torch.randn(2).to(device=device, dtype=dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
            torch.randn(2, 320, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
            torch.randn(2, 320, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
            torch.randn(2, 320, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
            torch.randn(2, 320, unet_sample_size//2,unet_sample_size//2).to(device=device, dtype=dtype),
            torch.randn(2, 640, unet_sample_size//2,unet_sample_size//2).to(device=device, dtype=dtype),
            torch.randn(2, 640, unet_sample_size//2,unet_sample_size//2).to(device=device, dtype=dtype),
            torch.randn(2, 640, unet_sample_size//4,unet_sample_size//4).to(device=device, dtype=dtype),
            torch.randn(2, 1280, unet_sample_size//4,unet_sample_size//4).to(device=device, dtype=dtype),
            torch.randn(2, 1280, unet_sample_size//4,unet_sample_size//4).to(device=device, dtype=dtype),
            torch.randn(2, 1280, unet_sample_size//8,unet_sample_size//8).to(device=device, dtype=dtype),
            torch.randn(2, 1280, unet_sample_size//8,unet_sample_size//8).to(device=device, dtype=dtype),
            torch.randn(2, 1280, unet_sample_size//8,unet_sample_size//8).to(device=device, dtype=dtype),
            torch.randn(2, 1280, unet_sample_size//8,unet_sample_size//8).to(device=device, dtype=dtype)),
            output_path= unet_path,
            ordered_input_names=[
            "sample", 
            "timestep", 
            "encoder_hidden_states", 
            "down_block_0",
            "down_block_1",
            "down_block_2",
            "down_block_3",
            "down_block_4",
            "down_block_5",
            "down_block_6",
            "down_block_7",
            "down_block_8",
            "down_block_9",
            "down_block_10",
            "down_block_11",
            "mid_block_additional_residual"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "down_block_0": {0: "batch", 2: "height", 3: "width"},
            "down_block_1": {0: "batch", 2: "height", 3: "width"},
            "down_block_2": {0: "batch", 2: "height", 3: "width"},
            "down_block_3": {0: "batch", 2: "height2", 3: "width2"},
            "down_block_4": {0: "batch", 2: "height2", 3: "width2"},
            "down_block_5": {0: "batch", 2: "height2", 3: "width2"},
            "down_block_6": {0: "batch", 2: "height4", 3: "width4"},
            "down_block_7": {0: "batch", 2: "height4", 3: "width4"},
            "down_block_8": {0: "batch", 2: "height4", 3: "width4"},
            "down_block_9": {0: "batch", 2: "height8", 3: "width8"},
            "down_block_10": {0: "batch", 2: "height8", 3: "width8"},
            "down_block_11": {0: "batch", 2: "height8", 3: "width8"},
            "mid_block_additional_residual": {0: "batch", 2: "height8", 3: "width8"},
        },
        opset=opset,)
    del pipeline.unet
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = os.path.dirname(unet_model_path)
    unet = onnx.load(unet_model_path)
    # clean up existing tensor files
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)

    optimizer = UnetOnnxModelDML(unet, 0, 0)
    if not notune:
        optimizer.optimize()
        optimizer.topological_sort()

    # collate external tensor files into one
    onnx.save_model(
        optimizer.model,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    if fp16:
        convert_to_fp16(unet_model_path)
    del unet, optimizer

    print(f"UNET model has been exported to {unet_path}")

args = parser.parse_args()
dtype = torch.float32
device = 'cuda'
unet = UNet2DConditionModel_Cnet.from_pretrained(args.model_path, subfolder="unet", use_safetensors=True)
pl = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype, unet=unet).to(device)
convert_unet(pl, args.output_path, args.opset, args.fp16)


        