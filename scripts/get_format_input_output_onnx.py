import onnxruntime as ort

# Define the path to the ONNX model
onnx_model_path = "/home/chaos/Documents/Chaos_project/model/controlnet_output/controlnet_seg/controlnet/model.onnx"

# Load the ONNX model
session = ort.InferenceSession(onnx_model_path)

# Get the model's input metadata
input_meta = session.get_inputs()

# Print the input metadata
print("Input metadata:")
for i, input in enumerate(input_meta):
    print(f"Input {i}:")
    print(f"  Name: {input.name}")
    print(f"  Shape: {input.shape}")

# Optionally, get and print the model's output metadata
output_meta = session.get_outputs()
print("\nOutput metadata:")
for i, output in enumerate(output_meta):
    print(f"Output {i}:")
    print(f"  Name: {output.name}")
    print(f"  Shape: {output.shape}")
