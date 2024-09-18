#%%
import onnxruntime as ort
import numpy as np

#%%
# onnx_model_path = '/home/chaos/Documents/Chaos_project/project/stable_diffusion_pipe_line/model/sd15_onnx/unet_cnet/model.onnx'

# session =ort.InferenceSession(onnx_model_path)

# Input_meta = session.get_inputs()

# print('input meta:')
# for i, input in enumerate(Input_meta):
#      print(f'input {i}')
#      print(f'name:', input.name)
#      print(f'shape:', input.shape)

# output_meta = session.get_outputs()
# print("\nOutput metadata:")
# for i, output in enumerate(output_meta):
#     print(f"Output {i}:")
#     print(f"  Name: {output.name}")
#     print(f"  Shape: {output.shape}")

#%%



# Define dummy data shapes based on model metadata
# Replace these values with appropriate integers if you know them
import numpy as np
import onnxruntime as ort


batch_size = 1
unet_in_channels = 4  # Update this value based on your model's expected input channels
unet_sample_size = 64
num_tokens = 10
text_hidden_size = 768
dtype = np.float32

# Define dummy numpy arrays for inputs
dummy_inputs = {
    'sample': np.random.randn(batch_size, unet_in_channels, unet_sample_size, unet_sample_size).astype(dtype),
    'timestep': np.random.randn(1).astype(dtype),  # Assuming timestep is an integer
    'encoder_hidden_states': np.random.randn(batch_size, num_tokens, text_hidden_size).astype(dtype),
    'down_block_add_res00': np.random.randn(batch_size, 320, unet_sample_size, unet_sample_size).astype(dtype),
    'down_block_add_res01': np.random.randn(batch_size, 320, unet_sample_size, unet_sample_size).astype(dtype),
    'down_block_add_res02': np.random.randn(batch_size, 320, unet_sample_size, unet_sample_size).astype(dtype),
    'down_block_add_res03': np.random.randn(batch_size, 320, unet_sample_size//2, unet_sample_size//2).astype(dtype),
    'down_block_add_res04': np.random.randn(batch_size, 640, unet_sample_size//2, unet_sample_size//2).astype(dtype),
    'down_block_add_res05': np.random.randn(batch_size, 640, unet_sample_size//2, unet_sample_size//2).astype(dtype),
    'down_block_add_res06': np.random.randn(batch_size, 640, unet_sample_size//4, unet_sample_size//4).astype(dtype),
    'down_block_add_res07': np.random.randn(batch_size, 1280, unet_sample_size//4, unet_sample_size//4).astype(dtype),
    'down_block_add_res08': np.random.randn(batch_size, 1280, unet_sample_size//4, unet_sample_size//4).astype(dtype),
    'down_block_add_res09': np.random.randn(batch_size, 1280, unet_sample_size//8, unet_sample_size//8).astype(dtype),
    'down_block_add_res10': np.random.randn(batch_size, 1280, unet_sample_size//8, unet_sample_size//8).astype(dtype),
    'down_block_add_res11': np.random.randn(batch_size, 1280, unet_sample_size//8, unet_sample_size//8).astype(dtype),
    'mid_block_res_sample': np.random.randn(batch_size, 1280, unet_sample_size//8, unet_sample_size//8).astype(dtype)
}

# Path to the ONNX model
providers = ['CUDAExecutionProvider']

onnx_model_path = '/home/chaos/Documents/Chaos_project/project/Inferences-StableDiffusion-Onnx-with-flexible-format/output_onnx/unet_optimize/model.onnx'
session = ort.InferenceSession(onnx_model_path, provider =providers)



# Run inference
try:
    outputs = session.run(None, dummy_inputs)
    # Print the shapes of the output
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
except Exception as e:
    print("Error during inference:", e)

# %%
