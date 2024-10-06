# ⚙️ **Guide convert Stable Diffusion Controlnet to onnx**
## 📡 **API Documents**



## 📝 **change script**

```bash
python -m scripts.convert_controlnet_to_onnx \           
--model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \  #model stablediffusion version
--controlnet_path lllyasviel/sd-controlnet-depth \   #model controlnet version
--output_path /home/tiennv/hungnq/server/projects/duypc/output_onnx/dept \   # output path for save file onnx. example/onnx_output 
```

## 🚀 **Run code convert **

```text
cd Inferences-StableDiffusion-Onnx-with-flexible-format/scripts
```

```bash
bash script.sh
```