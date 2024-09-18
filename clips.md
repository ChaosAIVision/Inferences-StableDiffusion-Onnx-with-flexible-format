# Inferences-StableDiffusion-Onnx-with-flexible-format
## 📝 **Script Convert Onnx Model**

Intallation






```shell
!pip install transformers
!pip install onnx==1.15.0
!pip install onnxruntime==1.17.0
!pip install onnxruntime-gpu==1.17.0
!pip install diffusers==0.28.0
!pip install onnx_graphsurgeon
!pip install polygraphy
!pip install peft
!pip install colored
```

```python

# Convert model UnetControlNet to ONNX
#Scriptconvert: https://colab.research.google.com/drive/1mCYjPcjk24rYE20kiylqQZ5ALHazzrvJ#scrollTo=WuiAbbaOdpYj

# Convert model ControlNet to ONNX
!python /content/convert_controlnet_onnx.py \
--model_path "rupeshs/LCM-runwayml-stable-diffusion-v1-5" \
--controlnet_path "lllyasviel/sd-controlnet-openpose" \
--opset 16 \
--output_path /content \


```
