import numpy as np
import os
import torch
import einops
import traceback
import cv2

import modules.async_worker as worker
from modules.util import generate_temp_filename
from PIL import Image

import os
from comfy.model_base import BaseModel, SDXL, SD3, Flux, Lumina2, HunyuanVideo
from shared import path_manager, settings
import shared

from pathlib import Path
import random
from modules.pipeline_utils import (
    clean_prompt_cond_caches,
)

import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config
from tqdm import tqdm

#from calcuis_gguf.pig import load_gguf_sd, GGMLOps, GGUFModelPatcher
#from comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, GGUFModelPatcher
#from comfyui_gguf.ops import GGMLOps
from molbal_comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, GGUFModelPatcher
from molbal_comfyui_gguf.ops import GGMLOps


from nodes import (
    CLIPTextEncode,
    VAEDecodeTiled,
)

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, RandomNoise, BasicScheduler, KSamplerSelect, BasicGuider
from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo, HunyuanImageToVideo 
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_flux import FluxGuidance


class pipeline:
    pipeline_type = ["hunyuan_video"]

    class StableDiffusionModel:
        def __init__(self, unet, vae, clip, clip_vision):
            self.unet = unet
            self.vae = vae
            self.clip = clip
            self.clip_vision = clip_vision

        def to_meta(self):
            if self.unet is not None:
                self.unet.model.to("meta")
            if self.clip is not None:
                self.clip.cond_stage_model.to("meta")
            if self.vae is not None:
                self.vae.first_stage_model.to("meta")

    model_hash = ""
    model_base = None
    model_hash_patched = ""
    model_base_patched = None
    conditions = None

    ggml_ops = GGMLOps()

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = 1 + ((int(gen_data["image_number"] / 4.0) + 1) * 4)
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, unet_only=True, input_unet=None, hash=None): # Hunyuan_Video never has the clip and vae models?

        # Check if model is already loaded
        if self.model_hash == name:
            return

        self.model_base = None
        self.model_hash = ""
        self.model_base_patched = None
        self.model_hash_patched = ""
        self.conditions = None

# FIXME? Add default model for video
#        default_name = path_manager.get_folder_file_path(
#            "checkpoints",
#            settings.default_settings.get("base_model", "sd_xl_base_1.0_0.9vae.safetensors"),
#        )
#        default = shared.models.get_file("checkpoints", default_name)
        default = None

        filename = str(
            shared.models.get_model_path(
                "checkpoints",
                name,
                hash=hash,
                default=default,
            )
        )

        print(f"Loading Hunyuan video {'unet' if unet_only else 'model'}: {name}")

        if filename.endswith(".gguf") or unet_only:
            with torch.torch.inference_mode():
                try:
                    if filename.endswith(".gguf"):
                        sd = load_gguf_sd(filename)
                        unet = comfy.sd.load_diffusion_model_state_dict(
                            sd, model_options={"custom_operations": self.ggml_ops}
                        )
                        unet = GGUFModelPatcher.clone(unet)
                        unet.patch_on_device = True
                    else:
                        model_options = {}
                        model_options["dtype"] = torch.float8_e4m3fn # FIXME should be a setting
                        unet = comfy.sd.load_diffusion_model(filename, model_options=model_options)

                    clip_paths = []
                    clip_names = []

                    if isinstance(unet.model, HunyuanVideo):
                        clip_name = settings.default_settings.get("clip_l", "clip_l.safetensors")
                        clip_names.append(str(clip_name))
                        clip_path = path_manager.get_folder_file_path(
                            "clip",
                            clip_name,
                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                        )
                        clip_paths.append(str(clip_path))
                        # https://huggingface.co/calcuis/hunyuan-gguf/tree/main
                        clip_name = settings.default_settings.get("clip_llava", "llava_llama3_fp8_scaled.safetensors")
                        clip_names.append(str(clip_name))
                        clip_path = path_manager.get_folder_file_path(
                            "clip",
                            clip_name,
                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                        )
                        clip_paths.append(str(clip_path))
                        clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO
                        # https://huggingface.co/calcuis/hunyuan-gguf/tree/main
                        vae_name = settings.default_settings.get("vae_hunyuan_video", "hunyuan_video_vae_bf16.safetensors")

                    else:
                        print(f"ERROR: Not a Hunyuan Video model?")
                        unet = None
                        return

                    print(f"Loading CLIP: {clip_names}")
                    clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO
                    clip = comfy.sd.load_clip(ckpt_paths=clip_paths, clip_type=clip_type, model_options={})

                    vae_path = path_manager.get_folder_file_path(
                        "vae",
                        vae_name,
                        default = os.path.join(path_manager.model_paths["vae_path"], vae_name)
                    )
                    print(f"Loading VAE: {vae_name}")
                    sd = comfy.utils.load_torch_file(str(vae_path))
                    vae = comfy.sd.VAE(sd=sd)

                    clip_vision = None
                except Exception as e:
                    unet = None
                    traceback.print_exc() 

        else:
            try:
                with torch.torch.inference_mode():
                    unet, clip, vae, clip_vision = load_checkpoint_guess_config(filename)

                if clip == None or vae == None:
                    raise
            except:
                print(f"Failed. Trying to load as unet.")
                self.load_base_model(
                    filename,
                    unet_only=True
                )
                return

        if unet == None:
            print(f"Failed to load {name}")
            self.model_base = None
            self.model_hash = ""
        else:
            self.model_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
            )
            if not (
                isinstance(self.model_base.unet.model, HunyuanVideo)
            ):
                print(
                    f"Model {type(self.model_base.unet.model)} not supported. Expected Hunyuan Video model."
                )
                self.model_base = None

            if self.model_base is not None:
                self.model_hash = name
                print(f"Base model loaded: {self.model_hash}")
        return

    def load_keywords(self, lora):
        filename = lora.replace(".safetensors", ".txt")
        try:
            with open(filename, "r") as file:
                data = file.read()
            return data
        except FileNotFoundError:
            return " "

    def load_loras(self, loras):
        loaded_loras = []

        model = self.model_base
#        for name, weight in loras:
#           if name == "None" or weight == 0:
#               continue
#           filename = str(shared.models.get_file("loras", name))

        for lora in loras:
            name = lora.get("name", "None")
            weight = lora.get("weight", 0)
            hash = lora.get("hash", None)
            if name == "None" or weight == 0:
                continue

            filename = shared.models.get_model_path(
                "loras",
                name,
                hash=hash,
            )

            if filename is None:
                continue

            print(f"Loading LoRAs: {name}")
            try:
                lora = comfy.utils.load_torch_file(filename, safe_load=True)
                unet, clip = comfy.sd.load_lora_for_models(
                    model.unet, model.clip, lora, weight, weight
                )
                model = self.StableDiffusionModel(
                    unet=unet,
                    clip=clip,
                    vae=model.vae,
                    clip_vision=model.clip_vision,
                )
                loaded_loras += [(name, weight)]
            except:
                pass
        self.model_base_patched = model
        self.model_hash_patched = str(loras)

        print(f"LoRAs loaded: {loaded_loras}")

        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    conditions = None

    def textencode(self, id, text, clip_skip):
        update = False
        hash = f"{text} {clip_skip}"
        if hash != self.conditions[id]["text"]:
            self.conditions[id]["cache"] = CLIPTextEncode().encode(
                clip=self.model_base_patched.clip, text=text
            )[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    # From https://github.com/lllyasviel/FramePack/blob/main/diffusers_helper/hunyuan.py#L61C1
    @torch.no_grad()
    def vae_decode_fake(self, latents):
        latent_rgb_factors = [
            [-0.0395, -0.0331, 0.0445],
            [0.0696, 0.0795, 0.0518],
            [0.0135, -0.0945, -0.0282],
            [0.0108, -0.0250, -0.0765],
            [-0.0209, 0.0032, 0.0224],
            [-0.0804, -0.0254, -0.0639],
            [-0.0991, 0.0271, -0.0669],
            [-0.0646, -0.0422, -0.0400],
            [-0.0696, -0.0595, -0.0894],
            [-0.0799, -0.0208, -0.0375],
            [0.1166, 0.1627, 0.0962],
            [0.1165, 0.0432, 0.0407],
            [-0.2315, -0.1920, -0.1355],
            [-0.0270, 0.0401, -0.0821],
            [-0.0616, -0.0997, -0.0727],
            [0.0249, -0.0469, -0.1703]
        ]  # From comfyui

        latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]

        weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
        bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

        images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
        images = images.clamp(0.0, 1.0)

        return images

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Processing text encoding ...", "html/generate_video.jpeg")
            )

        if self.conditions is None:
            self.conditions = clean_prompt_cond_caches()

        positive_prompt = gen_data["positive_prompt"]
        negative_prompt = gen_data["negative_prompt"]
        clip_skip = 1

        self.textencode("+", positive_prompt, clip_skip)
        self.textencode("-", negative_prompt, clip_skip)

        pbar = comfy.utils.ProgressBar(gen_data["steps"])

        def callback_function(step, x0, x, total_steps):
            y = self.vae_decode_fake(x0)
            y = (y * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            y = einops.rearrange(y, 'b c t h w -> (b h) (t w) c')
            # Skip callback() since we'll just confuse the preview grid and push updates outselves
            status = "Generating video"

            maxw = 1920
            maxh = 1080
            image = Image.fromarray(y)
            ow, oh = image.size
            scale = min(maxh / oh, maxw / ow)
            image = image.resize((int(ow * scale), int(oh * scale)), Image.LANCZOS)

            worker.add_result(
                gen_data["task_id"],
                "preview",
                (
                    int(100 * (step / total_steps)),
                    f"{status} - {step}/{total_steps}",
                    image
                )
            )
            pbar.update_absolute(step + 1, total_steps, None)

        # Noise
        noise = RandomNoise().get_noise(noise_seed=seed)[0]

        # latent_image
        # t2v or i2v?
        if gen_data["input_image"]:
            image = np.array(gen_data["input_image"]).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            (positive, latent_image) = HunyuanImageToVideo().encode(
                positive = self.conditions["+"]["cache"],
                vae = self.model_base_patched.vae,
                width = gen_data["width"],
                height = gen_data["height"],
                length = gen_data["original_image_number"],
                batch_size = 1,
                #guidance_type = "v1 (concat)", # "v2 (replace)"
                guidance_type = "v2 (replace)",
                start_image = image,
            )
        else:
            # latent_image
            latent_image = EmptyHunyuanLatentVideo().generate(
                width = gen_data["width"],
                height = gen_data["height"],
                length = gen_data["original_image_number"],
                batch_size = 1,
            )[0]
            positive = self.conditions["+"]["cache"]

        negative = self.conditions["-"]["cache"]

        # Guider
        model_sampling = ModelSamplingSD3().patch(
            model = self.model_base_patched.unet,
            shift = 7.0,
        )[0]
        flux_guideance = FluxGuidance().append(
            conditioning = positive,
            guidance = gen_data["cfg"],
        )[0]

        guider = BasicGuider().get_guider(
            model = model_sampling,
            conditioning = flux_guideance,
        )[0]

        # Sampler
        ksampler = KSamplerSelect().get_sampler(
            sampler_name = gen_data["sampler_name"],
        )[0]

        # Sigmas
        sigmas = BasicScheduler().get_sigmas(
            model = self.model_base_patched.unet,
            scheduler = gen_data["scheduler"],
            steps = gen_data["steps"],
            denoise = 1,
        )[0]

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Generating ...", None)
        )

        # From https://github.com/comfyanonymous/ComfyUI/blob/880c205df1fca4491c78523eb52d1a388f89ef92/comfy_extras/nodes_custom_sampler.py#L623
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        samples = guider.sample(
            noise.generate_noise(latent),
            latent_image,
            ksampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback_function,
            disable_pbar=False,
            seed=noise.seed
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        sampled = latent.copy()
        sampled["samples"] = samples


        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE Decoding ...", None)
            )

        decoded_latent = VAEDecodeTiled().decode(
            samples=sampled,
            tile_size=128,
            overlap=64,
            vae=self.model_base_patched.vae,
        )[0]

        pil_images = []
        for image in decoded_latent:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Saving ...", None)
            )

        file = generate_temp_filename(
            folder=path_manager.model_paths["temp_outputs_path"], extension="gif"
        )
        os.makedirs(os.path.dirname(file), exist_ok=True)

        fps=12.0
        compress_level=9 # Min = 0, Max = 9

        # Save GIF
        pil_images[0].save(
            file,
            compress_level=compress_level,
            save_all=True,
            duration=int(1000.0/fps),
            append_images=pil_images[1:],
            optimize=True,
            loop=0,
        )

        # Save mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mp4_file = file.with_suffix(".mp4")
        out = cv2.VideoWriter(mp4_file, fourcc, fps, (gen_data["width"], gen_data["height"]))
        for frame in pil_images:
            out.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB))
        out.release()

        return [file]
