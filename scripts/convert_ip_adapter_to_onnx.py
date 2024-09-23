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
from typing import Union, Optional, Tuple
from diffusers import AutoPipelineForText2Image
from repo.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from transformers import CLIPVisionModelWithProjection



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
        if onnx_graph.ByteSize() > 4147483648:
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




class ImageProjModel(torch.nn.Module):
    def __init__(self, proj_model):
        super().__init__()
        self.proj_model = proj_model

    def forward(self, image_embedding: torch.tensor, clip_embedding: torch.tensor) -> torch.tensor:
        output_proj = self.proj_model(image_embedding, clip_embedding)
        return output_proj

class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self,unet):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        down_block_add_res00: Optional[torch.Tensor] = None,
        down_block_add_res01: Optional[torch.Tensor] = None,
        down_block_add_res02: Optional[torch.Tensor] = None,
        down_block_add_res03: Optional[torch.Tensor] = None,
        down_block_add_res04: Optional[torch.Tensor] = None,
        down_block_add_res05: Optional[torch.Tensor] = None,
        down_block_add_res06: Optional[torch.Tensor] = None,
        down_block_add_res07: Optional[torch.Tensor] = None,
        down_block_add_res08: Optional[torch.Tensor] = None,
        down_block_add_res09: Optional[torch.Tensor] = None,
        down_block_add_res10: Optional[torch.Tensor] = None,
        down_block_add_res11: Optional[torch.Tensor] = None,
        mid_block_res_sample: Optional[torch.Tensor] = None
    ):
        down_block_res_samples = (
            down_block_add_res00, down_block_add_res01, down_block_add_res02,
            down_block_add_res03, down_block_add_res04, down_block_add_res05,
            down_block_add_res06, down_block_add_res07, down_block_add_res08,
            down_block_add_res09, down_block_add_res10, down_block_add_res11)
        
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        return noise_pred

def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset:int ,
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
    image_model_path:str,
    ip_adapter_weight_path:str,
    output_path:str,
    opset:int=16,
    fp16: bool = False,
    lora_weight_path:str =None 
):
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = "cuda"
    elif fp16 and not torch.cuda.is_available():
        raise ValueError("`float16` model export is only supported on GPUs with CUDA")
    else:
        device = "cpu"



    #init controlnet
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")


    #init sd pipeline
    sd_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_path, controlnet = controlnet, torch_dtype=dtype
        ).to(device)
    
    #fuselora
    if lora_weight_path is not None:
        sd_pipe.load_lora_weights('/content/ip-adapter-faceid-plus_sd15_lora.safetensors')
        sd_pipe.fuse_lora()


    #init ip-adapter pipeline
    pipeline = IPAdapterFaceIDPlus(sd_pipe, image_encoder_path=  '/content/clip_model', ip_ckpt= '/content/ip-adapter-faceid-plus_sd15.bin' , device= device, torch_dtype= dtype )

     
     # # TEXT ENCODER
    num_tokens = sd_pipe.text_encoder.config.max_position_embeddings
    text_hidden_size = sd_pipe.text_encoder.config.hidden_size
    text_input = sd_pipe.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=sd_pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    
    #export UNET

    unet_controlnet = UNet2DConditionControlNetModel(pipeline.pipe.unet)
    unet_in_channels = pipeline.pipe.unet.in_channels
    unet_sample_size = pipeline.pipe.unet.config.sample_size
    img_size = 8 * unet_sample_size
    output_path = Path(output_path)

    unet_path = output_path / "unet" / "model.onnx"

    onnx_export(
            unet_controlnet,
            model_args=(
                torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.tensor([1.0]).to(device=device, dtype=dtype),
                torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
                torch.randn(1, 320, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.randn(1, 320, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.randn(1, 320, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.randn(1, 320, unet_sample_size//2, unet_sample_size//2).to(device=device, dtype=dtype),
                torch.randn(1, 640, unet_sample_size//2, unet_sample_size//2).to(device=device, dtype=dtype),
                torch.randn(1, 640, unet_sample_size//2, unet_sample_size//2).to(device=device, dtype=dtype),
                torch.randn(1, 640, unet_sample_size//4, unet_sample_size//4).to(device=device, dtype=dtype),
                torch.randn(1, 1280, unet_sample_size//4, unet_sample_size//4).to(device=device, dtype=dtype),
                torch.randn(1, 1280, unet_sample_size//4, unet_sample_size//4).to(device=device, dtype=dtype),
                torch.randn(1, 1280, unet_sample_size//8, unet_sample_size//8).to(device=device, dtype=dtype),
                torch.randn(1, 1280, unet_sample_size//8, unet_sample_size//8).to(device=device, dtype=dtype),
                torch.randn(1, 1280,unet_sample_size//8, unet_sample_size//8).to(device=device, dtype=dtype),
                torch.randn(1, 1280, unet_sample_size//8, unet_sample_size//8).to(device=device, dtype=dtype),
            ),
            output_path=unet_path,
            ordered_input_names=[
                "sample",
                "timestep",
                "encoder_hidden_states",
                'down_block_add_res00',
                'down_block_add_res01',
                'down_block_add_res02',
                'down_block_add_res03',
                'down_block_add_res04', 
                'down_block_add_res05',
                'down_block_add_res06', 
                'down_block_add_res07', 
                'down_block_add_res08',
                'down_block_add_res09',
                'down_block_add_res10',
                'down_block_add_res11',
                'mid_block_res_sample'],


           
            output_names=["noise_pred"], 
            dynamic_axes={
                "sample": {0: "2B", 2: "H", 3: "W"},
                "encoder_hidden_states": {0: "B", 1:"2B", 2: '2B'},  # Tensor encoder hidden states
                "down_block_add_res00": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res01": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res02": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res03": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res04": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res05": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res06": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res07": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res08": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res09": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res10": {0: "B", 2: "H", 3: "W"},
                "down_block_add_res11": {0: "B", 2: "H", 3: "W"},
                "mid_block_res_sample": {0: "B", 2: "H", 3: "W"},
            },
            opset=opset,
            use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
        )
    unet_opt_graph =  onnx.load(unet_path)
    onnx.save_model(
        unet_opt_graph,
        '/content/output_onnx/unet_optimize/model.onnx',  # Đường dẫn lưu mô hình ONNX (chứa graph)
        save_as_external_data=True,  # Lưu dữ liệu tensor ra file ngoài
        all_tensors_to_one_file=True,  # Lưu tất cả tensor vào một file duy nhất
        location="weights.pb",  # Đặt tên file chứa các trọng số
        convert_attribute=False  # Không chuyển thuộc tính thành dữ liệu ngoài
    )

    


#convert proj model to onnx
    proj = (pipeline.image_proj_model)

    image_proj_model = ImageProjModel(proj)
    proj_path = output_path / "proj" / "model.onnx"

    onnx_export(image_proj_model,
                model_args=(torch.rand(1,512).to(device=device, dtype=dtype),
                            torch.rand(1,77,1280).to(device= device, dtype= dtype)),
                output_path = proj_path,
                ordered_input_names=[
                    'image_embedding',
                    'clip_embedding'],
                output_names= ['image_encoder'],
                dynamic_axes={
                        'image_embedding': {0: "batch_size", 1: "height"},
                        'clip_embedding': {0: "batch_size", 1: "seq_length", 2: "feature_dim"},
                        'image_encoder': {0: 'batch_size', 1: 'channels', 2: 'feature_dim'}

                    } ,
                opset=opset,
                use_external_data_format=True,              
               )
    
    proj_model_path = str(proj_path.absolute().as_posix())
    proj_dir = os.path.dirname(proj_model_path)
     # optimize onnx
    shape_inference.infer_shapes_path(proj_model_path, proj_model_path)
    proj_opt_graph = optimize(onnx.load(proj_model_path), name="proj", verbose=True)
    # clean up existing tensor files
    shutil.rmtree(proj_dir)
    os.mkdir(proj_dir)
    # collate external tensor files into one
    onnx.save_model(
            proj_opt_graph,
            proj_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--model_path",
    #     type=str,
    #     required=True,
    #     help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    # )

    # parser.add_argument(
    #     "--controlnet_path",
    #     type=str,
    #     nargs="+",
    #     required=True,
    #     help="Path to the `controlnet` checkpoint to convert (either a local directory or on the Hub).",
    # )

    # parser.add_argument(
    #         "--image_model_path",
    #         type=str,
    #         nargs="+",
    #         required=True,
    #         help="Path to the `model image ` checkpoint to convert (either a local directory or on the Hub).",
    #     )
    
    # parser.add_argument(
    #         "--ip_adapter_weight_path",
    #         type=str,
    #         nargs="+",
    #         required=True,
    #         help="Path to the `model image ` checkpoint to convert (either a local directory or on the Hub).",
    #     )
    # parser.add_argument(
    #     "--lora_weight_path",
    #     type=str,
    #     nargs="+",
    #     required=True,
    #     help="Path to the `model image ` checkpoint to convert (either a local directory or on the Hub).",
    # )

    # parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")


    # parser.add_argument(
    #     "--opset",
    #     default=14,
    #     type=int,
    #     help="The version of the ONNX operator set to use.",
    # )
    # parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    # args = parser.parse_args()






    convert_models(model_path= 'rupeshs/LCM-runwayml-stable-diffusion-v1-5',
                    controlnet_path= 'lllyasviel/sd-controlnet-openpose', 
                    image_model_path= '/content/clip_model',
                    ip_adapter_weight_path='/content/ip-adapter-faceid-plus_sd15.bin',
                    output_path= '/content/output_onnx',
                    opset= 16, 
                    fp16= False,
                    lora_weight_path= '/content/ip-adapter-faceid-plus_sd15_lora.safetensors')

