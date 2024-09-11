from stable_diffusion_manager import (
    StableDiffusionManager,
    StableDiffusionOnnxLoader, 
    StableDiffusionWeightPath)
from transformers import CLIPFeatureExtractor, CLIPImageProcessor, CLIPTokenizer
from diffusers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
from typing import Callable, List, Optional, Union
from utils import load_scheduler_config


path = '/home/chaos/Documents/Chaos_project/project/stable_diffusion_pipe_line/model/sd15_onnx'
sd_model_path = StableDiffusionWeightPath(path)
scheduler_config = load_scheduler_config('/home/chaos/Documents/Chaos_project/project/stable_diffusion_pipe_line/model/sd15_onnx/scheduler/scheduler_config.json')
onnx_model = StableDiffusionOnnxLoader(sd_model_path)
scheduler =  LMSDiscreteScheduler()
vocab_file = '/home/chaos/Documents/Chaos_project/project/stable_diffusion_pipe_line/model/sd15_onnx/tokenizer/vocab.json'
merge_file = '/home/chaos/Documents/Chaos_project/project/stable_diffusion_pipe_line/model/sd15_onnx/tokenizer/merges.txt'
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

pipeline = StableDiffusionManager(onnx_model,tokenizer ,scheduler, None ,scheduler_config )
prompt = ' a photo of an astronaut riding a horse on mars'
negative_promt = 'not generate a green horse '
# promt_encoder = pipeline.encoder_prompt(prompt, 1, True, negative_promt, None, None )
# print(promt_encoder.shape)

result = pipeline(prompt=prompt,negative_prompt= negative_promt ,num_inference_steps= 30)
# print(result)

from PIL import Image
import numpy as np

# Example NumPy array with shape (1, 512, 512, 3)
image_array = result * 255
image_array = image_array.astype(np.uint8)  # Convert to uint8 type for Pillow compatibility

# Remove the batch dimension
image_array = image_array[0]  # Now shape is (512, 512, 3)

# Convert the NumPy array to a Pillow Image
image = Image.fromarray(image_array)

# Display the image
image.show()