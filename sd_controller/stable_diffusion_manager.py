import warnings
warnings.filterwarnings("ignore")

#%% 
from dataclasses import dataclass
import os
os.environ['ONNX_LOG_SEVERITY_LEVEL'] = '4'
from dataclasses import dataclass
import onnxruntime as ort
from PIL import  Image
import torch
import numpy as np
from transformers import CLIPFeatureExtractor, CLIPImageProcessor, CLIPTokenizer
from diffusers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
from typing import Callable, List, Optional, Union
import inspect
from utils import load_scheduler_config , ORT_TO_NP_TYPE
import logging
from tqdm import tqdm
import logging
logger = logging



@dataclass
class StableDiffusionWeightPath:
    sd_folder_path: str

    def __post_init__(self):
        self.text_encoder = os.path.join(self.sd_folder_path, 'text_encoder/model.onnx')
        self.vae_encoder = os.path.join(self.sd_folder_path, 'vae_encoder/model.onnx')
        self.vae_decoder = os.path.join(self.sd_folder_path, 'vae_decoder/model.onnx')
        self.unet = os.path.join(self.sd_folder_path, 'unet/model.onnx')



#%% 


class StableDiffusionOnnxLoader:
    def __init__(self, sd_path:StableDiffusionWeightPath): 
        self.sd_path =  sd_path
        self.text_encoder = None
        self.vae_encoder = None
        self.vae_decoder= None
        self.unet = None

    def init_onnx_session(self, model_path):
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4
        session = ort.InferenceSession(model_path,sess_options=session_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        return session
    
    def vae_encoder_execute(self, input_data, image_width, image_height):
        image = Image.open(input_data)
        image = image.resize((image_height, image_width))  # Resize image if needed
        image = np.array(image).astype(np.float32)
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, 256, 256]
        image_np = image.cpu().numpy()
        session = self.init_onnx_session(self.sd_path.vae_encoder)
        input_name = session.get_inputs()[0].name  
        inputs = {input_name: image_np }   
        outputs_encoder = self.session.run(None, inputs)
        return outputs_encoder
    
    def vae_decoder_execute(self, latent ):
        image_height = 512
        image_width= 512
        session = self.init_onnx_session(self.sd_path.vae_decoder)

        scheduler_result = latent.reshape(1, 4, image_height // 8, image_width // 8 )
        scheduler_result = np.array(scheduler_result)
        input_decoder = {'latent_sample' :scheduler_result }

        outputs_encoder = session.run(None, input_decoder)
        return outputs_encoder
    
    def unet_execute(self,sample, time_step, text_encoder):
        self.inputs = {
            'sample': sample,
            'timestep':time_step,
            'encoder_hidden_states': text_encoder
        }
        session = self.init_onnx_session(self.sd_path.unet)
        outputs_unet = session.run(None, self.inputs)
        return outputs_unet
    
    def text_encoder_execute(self, input_ids):
        session = self.init_onnx_session(self.sd_path.text_encoder)
        inputs = { 'input_ids': input_ids}
        outputs_text_encoder = session.run(None, inputs)
        return outputs_text_encoder
    

class StableDiffusionManager:
    def __init__(self, 
                onnx_model_execute:StableDiffusionOnnxLoader, 
                tokenizer:CLIPTokenizer,
                scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
                feature_extractor: CLIPImageProcessor,
                scheduler_config: load_scheduler_config ):

    
        self.onnx_model_execute = onnx_model_execute
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config
        self.feature_extractor = feature_extractor

    def encoder_prompt(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: Optional[int],
            do_classifier_free_guidance: bool,
            negative_prompt: Optional[str],
            prompt_embeds: Optional[np.ndarray] = None,
            negative_prompt_embeds: Optional[np.ndarray] = None):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(prompt,
                                         padding= "max_length", 
                                         max_length= self.tokenizer.model_max_length,
                                         truncation = True,
                                         return_tensors= 'np'
                                         )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

            if not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.onnx_model_execute.text_encoder_execute(input_ids=text_input_ids.astype(np.int32))[0]

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
                    uncond_tokens: List[str]
                    if negative_prompt is None:
                        uncond_tokens = [""] * batch_size
                    elif type(prompt) is not type(negative_prompt):
                        raise TypeError(
                            f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                            f" {type(prompt)}."
                        )
                    elif isinstance(negative_prompt, str):
                        uncond_tokens = [negative_prompt] * batch_size
                    elif batch_size != len(negative_prompt):
                        raise ValueError(
                            f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                            f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                            " the batch size of `prompt`."
                        )
                    else:
                        uncond_tokens = negative_prompt

                    max_length = prompt_embeds.shape[1]
                    uncond_input = self.tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="np",
                    )
                    negative_prompt_embeds = self.onnx_model_execute.text_encoder_execute(input_ids=uncond_input.input_ids.astype(np.int32))[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
    
    def check_inputs(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int],
            width: Optional[int],
            callback_steps: int,
            negative_prompt: Optional[str] = None,
            prompt_embeds: Optional[np.ndarray] = None,
            negative_prompt_embeds: Optional[np.ndarray] = None,
        ):
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

            if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
            ):
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )

            if prompt is not None and prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            elif prompt is None and prompt_embeds is None:
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

            if negative_prompt is not None and negative_prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )

            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 0.0,
            generator: Optional[np.random.RandomState] = None,
            latents: Optional[np.ndarray] = None,
            prompt_embeds: Optional[np.ndarray] = None,
            negative_prompt_embeds: Optional[np.ndarray] = None,
            callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
            callback_steps: int = 1,):
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self.encoder_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

         # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        latents = latents * np.float64(self.scheduler.init_noise_sigma)
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        progress_bar = tqdm(self.scheduler.timesteps)
        for i, t in enumerate(progress_bar):
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype='int64')
            noise_pred = self.onnx_model_execute.unet_execute(sample= latent_model_input, time_step= timestep, text_encoder=prompt_embeds)
            noise_pred = noise_pred[0]

             # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)
        latents = 1 / 0.18215 * latents

        image = np.concatenate([self.onnx_model_execute.vae_decoder_execute(latent= latents[i : i+1])[0] for i in range(latents.shape[0])])
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0,2,3,1))

        return image


  

