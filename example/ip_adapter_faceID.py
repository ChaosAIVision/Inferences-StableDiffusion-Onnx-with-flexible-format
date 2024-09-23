from ip_adapter import IPAdapter
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

from ip_adapter.ip_adapter_faceid import MLPProjModel
from diffusers.pipelines import StableDiffusionControlNetImg2ImgPipeline
from diffusers import  ControlNetModel
from repo.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
import torch
from rich.pretty import Pretty
from rich import print as rprint
from transformers import BertModel
from scripts.get_input_class import PyClassExtractor
from convert_ip_adapter_to_onnx import UNet2DConditionControlNetModel


model_path = 'digiplay/Juggernaut_final'
device = 'cuda'
image_model_path = '/home/chaos/Documents/Chaos_project/project/model_sd_weight/clip_image/'
ip_adapter_weight_path = '/home/chaos/Documents/Chaos_project/project/model_sd_weight/ip-adapter-faceid-plus_sd15.bin'
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_path, controlnet=controlnet, torch_dtype=torch.float16
        ).to(device)

pipe =  (IPAdapterFaceIDPlus(sd_pipe= pipeline, image_encoder_path=  image_model_path,ip_ckpt= ip_adapter_weight_path,device= device))

tensor_image = torch.randn(1,512, dtype = torch.float16).to(device)
tensor_clip = torch.randn(1, 77, 1280, dtype = torch.float16).to(device)
model = (pipe.pipe.unet)
model = UNet2DConditionControlNetModel(model)
# proj = (pipe.image_proj_model(tensor_image,tensor_clip ))

# output của nó sẽ là (1,4,768))
model.eval()

# Define the input tensors for ONNX export
input_sample = torch.randn(1, 4, 64, 64, dtype=torch.float16).to(device)  # Input sample tensor
step_input =       torch.tensor([1.0]).to(device=device, dtype=torch.float16).to(device)
encoder_hidden = torch.randn(1, 77, 768, dtype=torch.float16).to(device)  # Encoder hidden state

# Define the file path for saving the ONNX model
onnx_file_path = "unet_model.onnx"
dtype = torch.float16
num_tokens = pipeline.text_encoder.config.max_position_embeddings
text_hidden_size = pipeline.text_encoder.config.hidden_size
unet_in_channels = pipe.pipe.unet.in_channels
unet_sample_size = pipe.pipe.unet.config.sample_size
# Export the model to ONNX
torch.onnx.export(
    model,  # Model being exported
    (
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
            ),  # Input tensors
            'model.onnx',  # Output file path
    input_names=["input_sample", "step_input", "encoder_hidden","encoder_hidden_states",
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
                'mid_block_res_sample'],  # Input names for the ONNX model
    output_names=["output"],  # Output names for the ONNX model
    opset_version=16,  # ONNX opset version
    do_constant_folding=True,  # Whether to perform constant folding optimization
    dynamic_axes={
        "input_sample": {0: "batch_size"},  # Dynamic batch size for input sample
        "encoder_hidden": {0: "batch_size"}  # Dynamic batch size for encoder hidden state
    }
)

print(f"Model has been successfully converted to ONNX and saved to {onnx_file_path}.")