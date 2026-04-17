import gc
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
from comfy.model_base import LTXAV
from shared import path_manager, settings
import shared

from pathlib import Path
import random
from modules.pipeline_utils import (
    clean_prompt_cond_caches,
)

import comfy.utils
from comfy.sd import load_checkpoint_guess_config
from tqdm import tqdm

from comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, DualCLIPLoaderGGUF, GGUFModelPatcher, UnetLoaderGGUF
from comfyui_gguf.ops import GGMLOps
#from calcuis_gguf.pig import load_gguf_sd, GGMLOps, GGUFModelPatcher, load_gguf_clip
#from calcuis_gguf.pig import DualClipLoaderGGUF as DualCLIPLoaderGGUF


from nodes import (
    CLIPTextEncode,
    DualCLIPLoader,
    VAEDecodeTiled,
    VAEDecode,
)

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, RandomNoise, BasicScheduler, KSamplerSelect, BasicGuider, CFGGuider, RandomNoise
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVImgToVideo, LTXVConditioning, LTXVScheduler, LTXVConcatAVLatent
from comfy_extras.nodes_lt import ModelSamplingLTXV
from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio
from comfy_extras.nodes_flux import FluxGuidance


class pipeline:
    pipeline_type = ["ltx2_video"]

    class StableDiffusionModel:
        def __init__(self, unet, vae, clip, clip_vision, audio_vae=None):
            self.clip = clip
            self.unet = unet
            self.clip_vision = clip_vision
            self.vae = vae
            self.audio_vae = audio_vae

        def to_meta(self):
            if self.unet is not None:
                self.unet.model.to("meta")
            if self.clip is not None:
                self.clip.cond_stage_model.to("meta")
            if self.vae is not None:
                self.vae.first_stage_model.to("meta")

    clip = None
    vae = None
    audio_vae = None
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

    def load_base_model(self, name, unet_only=True, input_unet=None, hash=None):

        # Check if model is already loaded
        # FIXME: should not load all models?`(to avoid oom)
        if self.model_hash == name:
            return

        self.model_base = None
        self.model_hash = ""
        self.model_base_patched = None
        self.model_hash_patched = ""
        self.conditions = None

        default = None

        filename = str(
            shared.models.get_model_path(
                "checkpoints",
                name,
                hash=hash,
                default=default,
            )
        )

        # Try to clean up some ram
        gc.collect(generation=2)
        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()

        unet = None

        print(f"Loading LTX-2 video {'unet' if unet_only else 'model'}: {name}")

        if filename.endswith(".gguf") or unet_only:
            with torch.torch.inference_mode():
                try:
                    if filename.endswith(".gguf"):
                        extra = {}
                        try:
                            sd, extra = load_gguf_sd(filename)
                        except:
                            sd = load_gguf_sd(filename)

                        self.ggml_ops.Linear.dequant_dtype = None
                        self.ggml_ops.Linear.patch_dtype = None
                        unet = comfy.sd.load_diffusion_model_state_dict(
                            sd, model_options={"custom_operations": self.ggml_ops}, metadata=extra.get("metadata", {})
                        )
                        unet = GGUFModelPatcher.clone(unet)
                        unet.patch_on_device = True
                    else:
                        model_options = {}
                        model_options["dtype"] = torch.float8_e4m3fn # FIXME should be a setting
                        unet = comfy.sd.load_diffusion_model(filename, model_options=model_options)

                    clip_paths = []
                    clip_names = []

                    if isinstance(unet.model, LTXAV):
                        clip_name = settings.default_settings.get("clip_gemma3_12b", "gemma-3-12b-it-Q4_0.gguf")
                        clip_names.append(str(clip_name))
                        clip_path = path_manager.get_folder_file_path(
                            "clip",
                            clip_name,
                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                        )
                        clip_paths.append(str(clip_path))

#                        clip_name = settings.default_settings.get("clip_ltx23_mmproj", "mmproj-F16.gguf")
#                        clip_names.append(str(clip_name))
#                        clip_path = path_manager.get_folder_file_path(
#                            "clip",
#                            clip_name,
#                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
#                        )
#                        clip_paths.append(str(clip_path))

                        clip_name = settings.default_settings.get("clip_ltx23_text_proj", "ltx-2.3_text_projection_bf16.safetensors")
                        clip_names.append(str(clip_name))
                        clip_path = path_manager.get_folder_file_path(
                            "clip",
                            clip_name,
                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                        )
                        clip_paths.append(str(clip_path))

#                        clip_name = settings.default_settings.get("clip_ltx23_22b_connwctors", "ltx-2.3-22b-distilled_embeddings_connectors.safetensors")
#                        clip_names.append(str(clip_name))
#                        clip_path = path_manager.get_folder_file_path(
#                            "clip",
#                            clip_name,
#                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
#                        )
#                        clip_paths.append(str(clip_path))

                        vae_name = settings.default_settings.get("vae_ltxv23_video", "LTX23_video_vae_bf16.safetensors")
                        audio_vae_name = settings.default_settings.get("vae_ltxv23_audio", "LTX23_audio_vae_bf16.safetensors")
                    else:
                        print(f"ERROR: Not a LTX2 Video model?")
                        unet = None
                        return

                    print(f"Loading CLIP: {clip_names}")
#                    if all(name.endswith(".safetensors") for name in clip_paths):
#                        model_options = {}
#                        device = comfy.model_management.get_torch_device()
#                        if device == "cpu":
#                            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
#                        clip = comfy.sd.load_clip(ckpt_paths=clip_paths, clip_type=clip_type, model_options=model_options)
#                    else:
                    if True:
                        clip_loader = DualCLIPLoaderGGUF()
                        print(f"DEBUG: load_data")
                        clip = clip_loader.load_patcher(
                            clip_paths,
                            comfy.sd.CLIPType.LTXV,
                            clip_loader.load_data(clip_paths)
                        )
                        print(f"DEBUG: clip done: {type(clip)}")

                    vae_path = path_manager.get_folder_file_path(
                        "vae",
                        vae_name,
                        default = os.path.join(path_manager.model_paths["vae_path"], vae_name)
                    )
                    audio_vae_path = path_manager.get_folder_file_path(
                        "vae",
                        audio_vae_name,
                        default = os.path.join(path_manager.model_paths["vae_path"], audio_vae_name)
                    )

                    print(f"Loading VAE: {vae_name}")
                    if str(vae_path).endswith(".gguf"):
                        sd = load_gguf_sd(str(vae_path))
                        metadata = None
                    else:
                        sd, metadata = comfy.utils.load_torch_file(str(vae_path), return_metadata=True)
                    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
                    print(f"DEBUG: video vae: {type(vae)}")


                    print(f"Loading Audio VAE: {audio_vae_name}")
                    if str(vae_path).endswith(".gguf"):
                        sd = load_gguf_sd(str(audio_vae_path))
                        metadata = None
                    else:
                        sd, metadata = comfy.utils.load_torch_file(str(audio_vae_path), return_metadata=True)
                    audio_vae = comfy.sd.VAE(sd=sd, metadata=metadata)
                    print(f"DEBUG: audio vae: {type(audio_vae)}")

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
                print(f"Trying to load as unet.")
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
                unet=unet, clip=None, vae=vae, clip_vision=clip_vision
            )
            self.clip = clip
            self.vae = vae
            self.audio_vae = audio_vae
            if not (
                isinstance(self.model_base.unet.model, LTXAV)
            ):
                print(
                    f"Model {type(self.model_base.unet.model)} not supported. Expected LTX Video model."
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
#            if name == "None" or weight == 0:
#                continue
#            filename = str(shared.models.get_file("loras", name))

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
                    clip=None,
                    vae=None,
                    clip_vision=None,
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
                clip=self.clip, text=text
            )[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    # From comfyui
    @torch.no_grad()
    def vae_decode_fake(self, latents):
        # FIXME get this from the comfy code
        
        self.latent_rgb_factors = [
            [ 1.1202e-02, -6.3815e-04, -1.0021e-02],
            [ 8.6031e-02,  6.5813e-02,  9.5409e-04],
            [-1.2576e-02, -7.5734e-03, -4.0528e-03],
            [ 9.4063e-03, -2.1688e-03,  2.6093e-03],
            [ 3.7636e-03,  1.2765e-02,  9.1548e-03],
            [ 2.1024e-02, -5.2973e-03,  3.4373e-03],
            [-8.8896e-03, -1.9703e-02, -1.8761e-02],
            [-1.3160e-02, -1.0523e-02,  1.9709e-03],
            [-1.5152e-03, -6.9891e-03, -7.5810e-03],
            [-1.7247e-03,  4.6560e-04, -3.3839e-03],
            [ 1.3617e-02,  4.7077e-03, -2.0045e-03],
            [ 1.0256e-02,  7.7318e-03,  1.3948e-02],
            [-1.6108e-02, -6.2151e-03,  1.1561e-03],
            [ 7.3407e-03,  1.5628e-02,  4.4865e-04],
            [ 9.5357e-04, -2.9518e-03, -1.4760e-02],
            [ 1.9143e-02,  1.0868e-02,  1.2264e-02],
            [ 4.4575e-03,  3.6682e-05, -6.8508e-03],
            [-4.5681e-04,  3.2570e-03,  7.7929e-03],
            [ 3.3902e-02,  3.3405e-02,  3.7454e-02],
            [-2.3001e-02, -2.4877e-03, -3.1033e-03],
            [ 5.0265e-02,  3.8841e-02,  3.3539e-02],
            [-4.1018e-03, -1.1095e-03,  1.5859e-03],
            [-1.2689e-01, -1.3107e-01, -2.1005e-01],
            [ 2.6276e-02,  1.4189e-02, -3.5963e-03],
            [-4.8679e-03,  8.8486e-03,  7.8029e-03],
            [-1.6610e-03, -4.8597e-03, -5.2060e-03],
            [-2.1010e-03,  2.3610e-03,  9.3796e-03],
            [-2.2482e-02, -2.1305e-02, -1.5087e-02],
            [-1.5753e-02, -1.0646e-02, -6.5083e-03],
            [-4.6975e-03,  5.0288e-03, -6.7390e-03],
            [ 1.1951e-02,  2.0712e-02,  1.6191e-02],
            [-6.3704e-03, -8.4827e-03, -9.5483e-03],
            [ 7.2610e-03, -9.9326e-03, -2.2978e-02],
            [-9.1904e-04,  6.2882e-03,  9.5720e-03],
            [-3.7178e-02, -3.7123e-02, -5.6713e-02],
            [-1.3373e-01, -1.0720e-01, -5.3801e-02],
            [-5.3702e-03,  8.1256e-03,  8.8397e-03],
            [-1.5247e-01, -2.1437e-01, -2.1843e-01],
            [ 3.1441e-02,  7.0335e-03, -9.7541e-03],
            [ 2.1528e-03, -8.9817e-03, -2.1023e-02],
            [ 3.8461e-03, -5.8957e-03, -1.5014e-02],
            [-4.3470e-03, -1.2940e-02, -1.5972e-02],
            [-5.4781e-03, -1.0842e-02, -3.0204e-03],
            [-6.5347e-03,  3.0806e-03, -1.0163e-02],
            [-5.0414e-03, -7.1503e-03, -8.9686e-04],
            [-8.5851e-03, -2.4351e-03,  1.0674e-03],
            [-9.0016e-03, -9.6493e-03,  1.5692e-03],
            [ 5.0914e-03,  1.2099e-02,  1.9968e-02],
            [ 1.3758e-02,  1.1669e-02,  8.1958e-03],
            [-1.0518e-02, -1.1575e-02, -4.1307e-03],
            [-2.8410e-02, -3.1266e-02, -2.2149e-02],
            [ 2.9336e-03,  3.6511e-02,  1.8717e-02],
            [-1.6703e-02, -1.6696e-02, -4.4529e-03],
            [ 4.8818e-02,  4.0063e-02,  8.7410e-03],
            [-1.5066e-02, -5.7328e-04,  2.9785e-03],
            [-1.7613e-02, -8.1034e-03,  1.3086e-02],
            [-9.2633e-03,  1.0803e-02, -6.3489e-03],
            [ 3.0851e-03,  4.7750e-04,  1.2347e-02],
            [-2.2785e-02, -2.3043e-02, -2.6005e-02],
            [-2.4787e-02, -1.5389e-02, -2.2104e-02],
            [-2.3572e-02,  1.0544e-03,  1.2361e-02],
            [-7.8915e-03, -1.2271e-03, -6.0968e-03],
            [-1.1478e-02, -1.2543e-03,  6.2679e-03],
            [-5.4229e-02,  2.6644e-02,  6.3394e-03],
            [ 4.4216e-03, -7.3338e-03, -1.0464e-02],
            [-4.5013e-03,  1.6082e-03,  1.4420e-02],
            [ 1.3673e-02,  8.8877e-03,  4.1253e-03],
            [-1.0145e-02,  9.0072e-03,  1.5695e-02],
            [-5.6234e-03,  1.1847e-03,  8.1261e-03],
            [-3.7171e-03, -5.3538e-03,  1.2590e-03],
            [ 2.9476e-02,  2.1424e-02,  3.0424e-02],
            [-3.4925e-02, -2.4340e-02, -2.5316e-02],
            [-3.4127e-02, -2.2406e-02, -1.0589e-02],
            [-1.7342e-02, -1.3249e-02, -1.0719e-02],
            [-2.1478e-03, -8.6051e-03, -2.9878e-03],
            [ 1.2089e-03, -4.2391e-03, -6.8569e-03],
            [ 9.0411e-04, -6.6886e-03, -6.7547e-05],
            [ 1.6048e-02, -1.0057e-02, -2.8929e-02],
            [ 1.2290e-03,  1.0163e-02,  1.8861e-02],
            [ 1.7264e-02,  2.7257e-04,  1.3785e-02],
            [-1.3482e-02, -3.6427e-03,  6.7481e-04],
            [ 4.6782e-03, -5.2423e-03,  2.4467e-03],
            [-5.9113e-03, -6.2244e-03, -1.8162e-03],
            [ 1.5496e-02,  1.4582e-02,  1.9514e-03],
            [ 7.4958e-03,  1.5886e-03, -8.2305e-03],
            [ 1.9086e-02,  1.6360e-03, -3.9674e-03],
            [-5.7021e-03, -2.7307e-03, -4.1066e-03],
            [ 1.7450e-03,  1.4602e-02,  2.5794e-02],
            [-8.2788e-04,  2.2902e-03,  4.5161e-03],
            [ 1.1632e-02,  8.9193e-03, -7.2813e-03],
            [ 7.5721e-03,  2.6784e-03,  1.1393e-02],
            [ 5.1939e-03,  3.6903e-03,  1.4049e-02],
            [-1.8383e-02, -2.2529e-02, -2.4477e-02],
            [ 5.8842e-04, -5.7874e-03, -1.4770e-02],
            [-1.6125e-02, -8.6101e-03, -1.4533e-02],
            [ 2.0540e-02,  2.0729e-02,  6.4338e-03],
            [ 3.3587e-03, -1.1226e-02, -1.6444e-02],
            [-1.4742e-03, -1.0489e-02,  1.7097e-03],
            [ 2.8130e-02,  2.3546e-02,  3.2791e-02],
            [-1.8532e-02, -1.2842e-02, -8.7756e-03],
            [-8.0533e-03, -1.0771e-02, -1.7536e-02],
            [-3.9009e-03,  1.6150e-02,  3.3359e-02],
            [-7.4554e-03, -1.4154e-02, -6.1910e-03],
            [ 3.4734e-03, -1.1370e-02, -1.0581e-02],
            [ 1.1476e-02,  3.9281e-03,  2.8231e-03],
            [ 7.1639e-03, -1.4741e-03, -3.8066e-03],
            [ 2.2250e-03, -8.7552e-03, -9.5719e-03],
            [ 2.4146e-02,  2.1696e-02,  2.8056e-02],
            [-5.4365e-03, -2.4291e-02, -1.7802e-02],
            [ 7.4263e-03,  1.0510e-02,  1.2705e-02],
            [ 6.2669e-03,  6.2658e-03,  1.9211e-02],
            [ 1.6378e-02,  9.4933e-03,  6.6971e-03],
            [ 1.7173e-02,  2.3601e-02,  2.3296e-02],
            [-1.4568e-02, -9.8279e-03, -1.1556e-02],
            [ 1.4431e-02,  1.4430e-02,  6.6362e-03],
            [-6.8230e-03,  1.8863e-02,  1.4555e-02],
            [ 6.1156e-03,  3.4700e-03, -2.6662e-03],
            [-2.6983e-03, -5.9402e-03, -9.2276e-03],
            [ 1.0235e-02,  7.4173e-03, -7.6243e-03],
            [-1.3255e-02,  1.9322e-02, -9.2153e-04],
            [ 2.4222e-03, -4.8039e-03, -1.5759e-02],
            [ 2.6244e-02,  2.5951e-02,  2.0249e-02],
            [ 1.5711e-02,  1.8498e-02,  2.7407e-03],
            [-2.1714e-03,  4.7214e-03, -2.2443e-02],
            [-7.4747e-03,  7.4166e-03,  1.4430e-02],
            [-8.3906e-03, -7.9776e-03,  9.7927e-03],
            [ 3.8321e-02,  9.6622e-03, -1.9268e-02],
            [-1.4605e-02, -6.7032e-03,  3.9675e-03]
        ]
        latent_rgb_factors_bias = [-0.0571, -0.1657, -0.2512]

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
        #frane_rate = settings.default_settings.get("fps", 24)
        gen_data["frame_rate"] = 24
        frame_number = (int(gen_data["original_image_number"]) * 8) + 1
        gen_data["width"] = int(gen_data["width"]) % 32
        gen_data["height"] = int(gen_data["height"]) % 32

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

        # latent_image
        # t2v or i2v?
        if gen_data["input_image"]:
            image = np.array(gen_data["input_image"]).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            (positive, negative, latent_image) = LTXVImgToVideo().generate(
                positive = self.conditions["+"]["cache"],
                negative = self.conditions["-"]["cache"],
                image = image,
                vae = self.model_base_patched.vae,
                width = gen_data["width"],
                height = gen_data["height"],
                length = frame_number,
                batch_size = 1,
                strength = 1,
            )
        else:
            # latent_image
            latent_image = EmptyLTXVLatentVideo().generate(
                width = gen_data["width"],
                height = gen_data["height"],
                length = frame_number,
                batch_size = 1,
            )[0]
            positive = self.conditions["+"]["cache"]

        # * EmptyLTXVLatentVideo(w, h, l) + LTXV Empty Latent Audio -> LTXVConatAVLatent
#        audio_vae_name = settings.default_settings.get("vae_audio_ltxv2", "vae/LTX2_audio_vae_bf16.safetensors")
#        audio_vae_path = path_manager.get_folder_file_path(
#            "vae",
#            audio_vae_name,
#            default = os.path.join(path_manager.model_paths["vae_path"], audio_vae_name)
#        )
#
#        print(f"Loading Audio VAE: {audio_vae_name}")
#        audio_vae = comfy.utils.load_torch_file(audio_vae_path, safe_load=True)

        empty_audio = LTXVEmptyLatentAudio().execute(
            audio_vae = self.audio_vae,
            frames_number = frame_number,
            frame_rate = gen_data["frame_rate"],
            batch_size = 1,
        )[0]
        video_latent = LTXVConcatAVLatent().execute(
            video_latent = latent_image,
            audio_latent = empty_audio,
        )[0]

        negative = self.conditions["-"]["cache"]

        # LTXVConditioning
        positive, negative = LTXVConditioning().append(
            positive = positive,
            negative = negative,
            frame_rate = gen_data("frame_rate")
        )

        # Sampler
        ksampler = KSamplerSelect().get_sampler(
            sampler_name = gen_data["sampler_name"],
        )[0]

        # Sigmas
        sigmas = LTXVScheduler().get_sigmas(
            steps = gen_data["steps"],
            max_shift = 2.05,
            base_shift = 0.95,
            stretch = True,
            terminal = 0.1,
            latent = latent_image
        )[0]

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Generating ...", None)
        )

        samples = SamplerCustomAdvanced().sample(
            model=self.model_base_patched.unet,
            add_noise=True,
            noise_seed=seed,
            cfg=float(gen_data["cfg"]),
            positive=positive,
            negative=negative,
            sampler=ksampler,
            sigmas=sigmas,
            latent_image=latent_image,
        )[0]

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE Decoding ...", None)
            )

        decoded_latent = VAEDecodeTiled().decode(
            samples=samples,
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=5,
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
