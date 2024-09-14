# Inferences-StableDiffusion-Onnx-with-flexible-format
## 📝 **Script Convert Onnx Model**

'''python 
Convert model StableDiffuserControlNet to ONNX
python3 /home/chaos/Documents/temp_folder/diffusers/scripts/convert_stable_diffusion_controlnet_to_onnx.py   --model_path "digiplay/Juggernaut_final" --controlnet_path "lllyasviel/sd-controlnet-canny" --output_path '/home/chaos/Documents/temp_folder/output_sd/' --fp16 

Convert model ControlNet to ONNX
!python /content/convert_controlnet_onnx.py \
--model_path "rupeshs/LCM-runwayml-stable-diffusion-v1-5" \
 --controlnet_path "lllyasviel/sd-controlnet-canny" \
 --fp16  \
 --opset 16 \
 --output_path /content \
 --control_net_hinter_type hint_canny


'''