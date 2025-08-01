import gc
import numpy as np
import os
import torch
import traceback
import math

import modules.controlnet
import modules.async_worker as worker
import modules.prompt_processing as pp

from PIL import Image, ImageOps

from comfy.model_base import BaseModel, SDXL, SD3, Flux, Lumina2
from shared import path_manager, settings
import shared

from pathlib import Path
import json
import random

import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config, load_state_dict_guess_config

from tqdm import tqdm

from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
from nodes import (
    CLIPTextEncode,
    CLIPSetLastLayer,
    ControlNetApplyAdvanced,
    EmptyLatentImage,
    VAEDecode,
    VAEEncode,
    VAEEncodeForInpaint,
    CLIPLoader,
    VAELoader,
)
from comfy.sampler_helpers import (
    cleanup_additional_models,
    convert_cond,
    get_additional_models,
    prepare_mask,
)

from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from comfy_extras.nodes_flux import FluxKontextImageScale
from comfy_extras.nodes_edit_model import ReferenceLatent
from node_helpers import conditioning_set_values

from comfy.samplers import KSampler
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_canny import Canny
from comfy_extras.nodes_freelunch import FreeU
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP, VAE
from comfy.utils import load_torch_file
from comfy.sd import save_checkpoint

from modules.pipleline_utils import (
    get_previewer,
    clean_prompt_cond_caches,
    set_timestep_range,
)

#from comfyui_gguf.nodes import gguf_sd_loader, DualCLIPLoaderGGUF, GGUFModelPatcher
#from comfyui_gguf.ops import GGMLOps
from calcuis_gguf.pig import load_gguf_sd, GGMLOps, GGUFModelPatcher
from calcuis_gguf.pig import DualClipLoaderGGUF as DualCLIPLoaderGGUF

class pipeline:
    pipeline_type = ["sdxl", "ssd", "sd3", "flux", "lumina2"]

    comfy.model_management.DISABLE_SMART_MEMORY = False
    comfy.model_management.EXTRA_RESERVED_VRAM = 800 * 1024 * 1024

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

    xl_base: StableDiffusionModel = None
    xl_base_hash = ""

    xl_base_patched: StableDiffusionModel = None
    xl_base_patched_hash = ""
    xl_base_patched_extra = set()

    xl_controlnet: StableDiffusionModel = None
    xl_controlnet_hash = ""

    models = []
    inference_memory = None

    ggml_ops = GGMLOps()

    def get_clip_name(self, shortname):
        # List of short names and default names for different text encoders
        defaults = {
            "clip_g": "clip_g.safetensors",
            "clip_gemma": "gemma_2_2b_fp16.safetensors",
            "clip_l": "clip_l.safetensors",
            "clip_t5": "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
        }
        return settings.default_settings.get(shortname, defaults[shortname] if shortname in defaults else None)

    def load_base_model(self, name, unet_only=False, input_unet=None, hash=None):
        if self.xl_base_hash == name and self.xl_base_patched_extra == set():
            return

        default_name = path_manager.get_folder_file_path(
            "checkpoints",
            settings.default_settings.get("base_model", "sd_xl_base_1.0_0.9vae.safetensors"),
        )
        default = shared.models.get_file("checkpoints", default_name)

        filename = shared.models.get_model_path(
            "checkpoints",
            name,
            hash=hash,
            default=default,
        )

        if filename is None:
            return

        if Path(filename).suffix == '.merge':
            print(f"Error: Model type not supported.")
            return

        if input_unet is None: # Be quiet if we already loaded a unet
            print(f"Loading base {'unet' if unet_only else 'model'}: {name}")

        self.xl_base = None
        self.xl_base_hash = ""
        self.xl_base_patched = None
        self.xl_base_patched_hash = ""
        self.xl_base_patched_extra = set()
        self.conditions = None
        gc.collect(generation=2)

        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()

        unet = None

        filename = str(filename) # FIXME use Path and suffix instead?
        if filename.endswith(".gguf") or unet_only:
            with torch.torch.inference_mode():
                try:
                    if input_unet is not None:
                        if isinstance(input_unet, ModelPatcher):
                            unet = input_unet
                        else:
                            unet = comfy.sd.load_diffusion_model_state_dict(
                                input_unet, model_options={"custom_operations": self.ggml_ops}
                            )
                        unet = GGUFModelPatcher.clone(unet)
                        unet.patch_on_device = True
                    elif filename.endswith(".gguf"):
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

                    # Get text encoders (clip) and vae to match the unet
                    clip_names = []

                    if isinstance(unet.model, Flux):
                        clip_names.append(self.get_clip_name("clip_l"))
                        clip_names.append(self.get_clip_name("clip_t5"))
                        clip_type = comfy.sd.CLIPType.FLUX
                        vae_name = settings.default_settings.get("vae_flux", "ae.safetensors")

                    elif isinstance(unet.model, SD3):
                        clip_names.append(self.get_clip_name("clip_l"))
                        clip_names.append(self.get_clip_name("clip_g"))
                        clip_names.append(self.get_clip_name("clip_t5"))
                        clip_type = comfy.sd.CLIPType.SD3
                        vae_name = settings.default_settings.get("vae_sd3", "sd3_vae.safetensors")

                    elif isinstance(unet.model, Lumina2):
                        clip_names.append(self.get_clip_name("clip_gemma"))
                        clip_type = comfy.sd.CLIPType.LUMINA2
                        vae_name = settings.default_settings.get("vae_lumina2", "lumina2_vae_fp32.safetensors")
                        unet = ModelSamplingAuraFlow().patch_aura(
                            model=unet,
                            shift=settings.default_settings.get("lumina2_shift", 3.0),
                        )[0]

                    else: # SDXL
                        clip_names.append(self.get_clip_name("clip_l"))
                        clip_names.append(self.get_clip_name("clip_g"))
                        clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
                        vae_name = settings.default_settings.get("vae_sdxl", "sdxl_vae.safetensors")

                    clip_paths = []
                    for clip_name in clip_names:
                        clip_paths.append(
                            str(
                                path_manager.get_folder_file_path(
                                    "clip",
                                    clip_name,
                                    default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                                )
                            )
                        )

                    clip_loader = DualCLIPLoaderGGUF()
                    print(f"Loading CLIP: {clip_names}")
                    clip = clip_loader.load_patcher(
                        clip_paths,
                        clip_type,
                        clip_loader.load_data(clip_paths)
                    )

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
            sd = None
            unet = None
            try:
                with torch.torch.inference_mode():
                    sd = comfy.utils.load_torch_file(filename)
            except Exception as e:
                # Failed loading
                print(f"ERROR: Failed loading {filename}: {e}")

            if sd is not None:
                aio = load_state_dict_guess_config(sd)
                if isinstance(aio, tuple):
                    unet, clip, vae, clip_vision = aio

                    if (
                        isinstance(unet, ModelPatcher) and
                        isinstance(clip, CLIP) and
                        isinstance(vae, VAE)
                    ):
                        # If we got here, we have all models. Dump sd since we don't need it
                        sd = None
                    else:
                        if isinstance(unet, ModelPatcher):
                            sd = unet

                if sd is not None:
                    # We got something, assume it was a unet
                    self.load_base_model(
                        filename,
                        unet_only=True,
                        input_unet=sd,
                    )
                    return

            else:
                unet = None

        if unet == None:
            print(f"Failed to load {name}")
            self.xl_base = None
            self.xl_base_hash = ""
            self.xl_base_patched = None
            self.xl_base_patched_hash = ""
        else:
            self.xl_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
            )
            if not (
                isinstance(self.xl_base.unet.model, BaseModel) or
                isinstance(self.xl_base.unet.model, SDXL) or
                isinstance(self.xl_base.unet.model, SD3) or
                isinstance(self.xl_base.unet.model, Flux) or
                isinstance(self.xl_base.unet.model, Lumina2)
            ):
                print(
                    f"Model {type(self.xl_base.unet.model)} not supported. RuinedFooocus only support SD1.x/SDXL/SD3/Flux/Lumina2 models as the base model."
                )
                self.xl_base = None

            if self.xl_base is not None:
                self.xl_base_hash = name
                self.xl_base_patched = self.xl_base
                self.xl_base_patched_hash = ""
                # self.xl_base_patched.unet.model.to("cuda")
                #print(f"Base model loaded: {self.xl_base_hash}")

        return

    def freeu(self, model, b1, b2, s1, s2):
        freeu_model = FreeU()
        unet = freeu_model.patch(model=model.unet, b1=b1, b2=b2, s1=s1, s2=s2)[0]
        return self.StableDiffusionModel(
            unet=unet, clip=model.clip, vae=model.vae, clip_vision=model.clip_vision
        )

    def load_loras(self, loras):
        loaded_loras = []
        loras = sorted(loras, key=lambda x: x['name'].lower())
        if self.xl_base_patched_hash == str(loras):
            return

        model = self.xl_base
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
            print(f"Loading LoRA: {name}")
            try:
                lora = comfy.utils.load_torch_file(str(filename), safe_load=True)
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
            except Exception as e:
                print(f"Error loading LoRA: {filename} {e}")
                pass
        self.xl_base_patched = model
        # Uncomment below to enable FreeU shit
        # self.xl_base_patched = self.freeu(model, 1.01, 1.02, 0.99, 0.95)
        # self.xl_base_patched_hash = str(loras + [1.01, 1.02, 0.99, 0.95])
        self.xl_base_patched_hash = str(loras)

        print(f"LoRAs loaded: {loaded_loras}")
        return

    def refresh_controlnet(self, name=None):
        if self.xl_controlnet_hash == str(self.xl_controlnet):
            return

        filename = modules.controlnet.get_model(name)

        if filename is not None and self.xl_controlnet_hash != name:
            self.xl_controlnet = comfy.controlnet.load_controlnet(str(filename))
            self.xl_controlnet_hash = name
            print(f"ControlNet model loaded: {self.xl_controlnet_hash}")
        if self.xl_controlnet_hash != name:
            self.xl_controlnet = None
            self.xl_controlnet_hash = None

    conditions = None

    def textencode(self, id, text, clip_skip):
        update = False
        hash = f"{text} {clip_skip}"
        if hash != self.conditions[id]["text"]:
            if clip_skip > 1:
                self.xl_base_patched.clip = CLIPSetLastLayer().set_last_layer(
                    self.xl_base_patched.clip, clip_skip * -1
                )[0]
            self.conditions[id]["cache"] = CLIPTextEncode().encode(
                clip=self.xl_base_patched.clip, text=text
            )[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        try:
            if self.xl_base_patched == None or not (
                isinstance(self.xl_base_patched.unet.model, BaseModel) or
                isinstance(self.xl_base_patched.unet.model, SDXL) or
                isinstance(self.xl_base_patched.unet.model, SD3) or
                isinstance(self.xl_base_patched.unet.model, Flux) or 
                isinstance(self.xl_base_patched.unet.model, Lumina2)
            ):
                print(f"ERROR: Can only use SD1.x, SDXL, SD3, Flux or Lumina2 models")
                worker.interrupt_ruined_processing = True
                if callback is not None:
                    worker.add_result(
                        gen_data["task_id"],
                        "preview",
                        (-1, f"Can only use SDXL, SD3 or Flux models ...", "html/error.png")
                    )
                return []
        except Exception as e:
            # Something went very wrong
            print(f"ERROR: {e}")
            worker.interrupt_ruined_processing = True
            if callback is not None:
                worker.add_result(
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Error when trying to use model ...", "html/error.png")
                )
            return []

        positive_prompt = gen_data["positive_prompt"]
        negative_prompt = gen_data["negative_prompt"]
        input_image = gen_data["input_image"]
        controlnet = modules.controlnet.get_settings(gen_data)

        cfg = gen_data["cfg"]
        sampler_name = gen_data["sampler_name"]
        scheduler = gen_data["scheduler"]
        clip_skip = gen_data["clip_skip"]

        img2img_mode = False
        input_image_pil = None
        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Processing text encoding ...", None)
            )
        updated_conditions = False
        if self.conditions is None:
            self.conditions = clean_prompt_cond_caches()

        if self.textencode("+", positive_prompt, clip_skip):
            updated_conditions = True
        if self.textencode("-", negative_prompt, clip_skip):
            updated_conditions = True

        switched_prompt = []
        if "[" in positive_prompt and "]" in positive_prompt:
            if controlnet is not None and input_image is not None:
                print("ControlNet and [prompt|switching] do not work well together.")
                print("ControlNet will only be applied to the first prompt.")

            prompt_per_step = pp.prompt_switch_per_step(positive_prompt, gen_data["steps"])
            perc_per_step = round(100 / gen_data["steps"], 2)
            for i in range(len(prompt_per_step)):
                if self.textencode("switch", prompt_per_step[i], clip_skip):
                    updated_conditions = True
                positive_switch = self.conditions["switch"]["cache"]
                start_perc = round((perc_per_step * i) / 100, 2)
                end_perc = round((perc_per_step * (i + 1)) / 100, 2)
                if end_perc >= 0.99:
                    end_perc = 1
                positive_switch = set_timestep_range(
                    positive_switch, start_perc, end_perc
                )
                switched_prompt += positive_switch

        device = comfy.model_management.get_torch_device()


        if controlnet is None or not "type" in controlnet:
            controlnet = {}
            controlnet["type"] = "None"

        # FIXME need a good whay to check if we are using Flux.1 Kontext
        if (
            controlnet["type"] == "None" and
            isinstance(self.xl_base.unet.model, Flux) and
            input_image is not None
        ):
            controlnet["type"] = "kontext"

        if controlnet["type"] != "None" and input_image is not None:
            if callback is not None:
                worker.add_result(
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Powering up ...", None)
                )
            input_image_pil = input_image.convert("RGB")
            input_image = np.array(input_image_pil).astype(np.float32) / 255.0
            input_image = torch.from_numpy(input_image)[None,]
            megapixels=float(gen_data["width"])*float(gen_data["height"])/(1024*1024)
            input_image = ImageScaleToTotalPixels().upscale(
                image=input_image, upscale_method="bicubic", megapixels=megapixels
            )[0]
            self.refresh_controlnet(name=controlnet["type"])
            match controlnet["type"].lower():
                case "canny":
                    input_image = Canny().detect_edge(
                        image=input_image,
                        low_threshold=float(controlnet["edge_low"]),
                        high_threshold=float(controlnet["edge_high"]),
                    )[0]
                    updated_conditions = True
                case "depth":
                    updated_conditions = True
            if self.xl_controlnet:
                (
                    self.conditions["+"]["cache"],
                    self.conditions["-"]["cache"],
                ) = ControlNetApplyAdvanced().apply_controlnet(
                    positive=self.conditions["+"]["cache"],
                    negative=self.conditions["-"]["cache"],
                    control_net=self.xl_controlnet,
                    image=input_image,
                    strength=float(controlnet["strength"]),
                    start_percent=float(controlnet["start"]),
                    end_percent=float(controlnet["stop"]),
                )
                self.conditions["+"]["text"] = None
                self.conditions["-"]["text"] = None

            # NOTE: If we are doing img2img, reuse the previous image ("Loopback").
            #       It is not obvious that this is a good idea.
            if controlnet["type"].lower() == "img2img":
                # If this isn't the first image, do "Loopback"
                if "preview_count" in shared.state and shared.state["preview_count"] > 0:
                    input_image = Image.fromarray(shared.shared_cache["prev_image"]).convert("RGB")
                    input_image = np.array(input_image).astype(np.float32) / 255.0
                    input_image = torch.from_numpy(input_image)[None,]
            if controlnet["type"].lower() == "kontext":
                input_image = FluxKontextImageScale().scale(input_image)[0]
            if controlnet["type"].lower() in ["img2img", "kontext"]:
                latent = VAEEncode().encode(
                    vae=self.xl_base_patched.vae, pixels=input_image
                )[0]
                force_full_denoise = False
                denoise = float(controlnet.get("denoise", controlnet.get("strength", 1)))
                img2img_mode = True

        if not img2img_mode:
            if (
                isinstance(self.xl_base.unet.model, SD3) or
                isinstance(self.xl_base.unet.model, Flux) or
                isinstance(self.xl_base.unet.model, Lumina2)
            ):
                latent = EmptySD3LatentImage().generate(
                    width=gen_data["width"], height=gen_data["height"], batch_size=1
                )[0]
            else: # SDXL and unknown
                latent = EmptyLatentImage().generate(
                    width=gen_data["width"], height=gen_data["height"], batch_size=1
                )[0]
            force_full_denoise = False
            denoise = None

        if "inpaint_toggle" in gen_data and gen_data["inpaint_toggle"]:
            # This is a _very_ ugly workaround since we had to shrink the inpaint image
            # to not break the ui.
            main_image = Image.open(gen_data["main_view"])
            image = np.asarray(main_image)
#            image = image[..., :-1]
            image = torch.from_numpy(image)[None,] / 255.0

            inpaint_view = Image.fromarray(gen_data["inpaint_view"]["layers"][0])
            red, green, blue, mask = inpaint_view.split()
            mask = mask.resize((main_image.width, main_image.height), Image.Resampling.LANCZOS)
            mask = np.asarray(mask)
#            mask = mask[:, :, 0]
            mask = torch.from_numpy(mask)[None,] / 255.0

            latent = VAEEncodeForInpaint().encode(
                vae=self.xl_base_patched.vae,
                pixels=image,
                mask=mask,
                grow_mask_by=20,
            )[0]

        if updated_conditions:
            conds = {
                0: self.conditions["+"]["cache"],
                1: self.conditions["-"]["cache"],
            }
            self.models, self.inference_memory = get_additional_models(
                conds,
                self.xl_base_patched.unet.model_dtype(),
            )

        comfy.model_management.load_models_gpu([self.xl_base_patched.unet])
        # Use FluxGuidance for Flux
        positive_cond = switched_prompt if switched_prompt else self.conditions["+"]["cache"]
        if isinstance(self.xl_base.unet.model, Flux):
            if controlnet.get("type", "") == "kontext":
                positive_cond = ReferenceLatent().append(positive_cond, latent=latent)[0]
            positive_cond = conditioning_set_values(positive_cond, {"guidance": cfg})
            cfg = 1.0

        latent_image = latent["samples"]
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        previewer = get_previewer(device, self.xl_base_patched.unet.model.latent_format)

        pbar = comfy.utils.ProgressBar(gen_data["steps"])

        def callback_function(step, x0, x, total_steps):
            y = None
            if previewer:
                y = previewer.preview(x0, step, total_steps)
            if callback is not None:
                callback(step, x0, x, total_steps, y)
            pbar.update_absolute(step + 1, total_steps, None)

        if noise_mask is not None:
            noise_mask = prepare_mask(noise_mask, noise.shape, device)

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Prepare models ...", None)
            )

        comfy.model_management.load_models_gpu(self.models)

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        kwargs = {
            "cfg": cfg,
            "latent_image": latent_image,
            "start_step": 0,
            "last_step": gen_data["steps"],
            "force_full_denoise": force_full_denoise,
            "denoise_mask": noise_mask,
            "sigmas": None,
            "disable_pbar": False,
            "seed": seed,
            "callback": callback_function,
        }
        sampler = KSampler(
            self.xl_base_patched.unet,
            steps=gen_data["steps"],
            device=device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            model_options=self.xl_base_patched.unet.model_options,
        )

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Start sampling ...", None)
            )
        samples = sampler.sample(
            noise,
            positive_cond,
            self.conditions["-"]["cache"],
            **kwargs,
        )

        cleanup_additional_models(self.models)

        sampled_latent = latent.copy()
        sampled_latent["samples"] = samples

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE decoding ...", None)
            )

        decoded_latent = VAEDecode().decode(
            samples=sampled_latent, vae=self.xl_base_patched.vae
        )[0]

        images = [
            np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8)
            for y in decoded_latent
        ]

        shared.shared_cache["prev_image"] = images[0]
        if callback is not None:
            callback(gen_data["steps"], 0, 0, gen_data["steps"], images[0])

        return images
