import gc
import numpy as np
import os
import torch
import cv2
import re

import modules.controlnet
import modules.async_worker as worker
import modules.prompt_processing as pp

from PIL import Image, ImageOps

from comfy.model_base import SDXL, SD3
from modules.settings import default_settings
from shared import path_manager

from pathlib import Path
import json
import random

import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config
from tqdm import tqdm

from comfy_extras.chainner_models import model_loading
from nodes import (
    CLIPTextEncode,
    CLIPSetLastLayer,
    ControlNetApplyAdvanced,
    EmptyLatentImage,
    VAEDecode,
    VAEEncode,
    VAEEncodeForInpaint,
)
from comfy.sampler_helpers import (
    cleanup_additional_models,
    convert_cond,
    get_additional_models,
    prepare_mask,
)

from comfy_extras.nodes_sd3 import EmptySD3LatentImage

from comfy.samplers import KSampler
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_canny import Canny
from comfy_extras.nodes_freelunch import FreeU
from comfy.model_patcher import ModelPatcher
from comfy.utils import load_torch_file
from comfy.sd import save_checkpoint
from modules.layerdiffuse import TransparentVAEDecoder, ImageRenderer

from modules.pipleline_utils import (
    get_previewer,
    clean_prompt_cond_caches,
    set_timestep_range,
)


class pipeline:
    pipeline_type = ["sdxl", "ssd"]

    comfy.model_management.DISABLE_SMART_MEMORY = True

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

    def merge_models(self, name):
        print(f"Loading merge: {name}")

        self.xl_base_patched = None
        self.xl_base_patched_hash = ""
        self.xl_base_patched_extra = set()
        self.conditions = None

        filename = Path(path_manager.model_paths["modelfile_path"] / name)
        cache_name = str(Path(path_manager.model_paths["cache_path"] / "merges" / Path(name).name).with_suffix(".safetensors"))
        if Path(cache_name).exists() and Path(cache_name).stat().st_mtime <= Path(filename).stat().st_mtime:
            print(f"Loading cached version:")
            self.load_base_model(cache_name)
            return

        try:
            with filename.open() as f:
                merge_data = json.load(f)

            if 'comment' in merge_data:
                print(f"  {merge_data['comment']}")

            filename = Path(path_manager.model_paths["modelfile_path"] / merge_data["base"]["name"])
            norm = 1.0
            if "models" in merge_data and len(merge_data["models"]) > 0:
                weights = sum([merge_data["base"]["weight"]] + [x.get("weight") for x in merge_data["models"]])
                if "normalize" in merge_data:
                    norm = float(merge_data["normalize"]) / weights
                else:
                    norm = 1.0 / weights

            print(f"Loading base {merge_data['base']['name']} ({round(merge_data['base']['weight'] * norm * 100)}%)")
            with torch.torch.inference_mode():
                unet, clip, vae, clip_vision = load_checkpoint_guess_config(str(filename))

            self.xl_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
            )
            if self.xl_base is not None:
                self.xl_base_hash = name
                self.xl_base_patched = self.xl_base
                self.xl_base_patched_hash = ""
        except Exception as e:
            self.xl_base = None
            print(f"ERROR: {e}")
            return

        if "models" in merge_data and len(merge_data["models"]) > 0:
            device = comfy.model_management.get_torch_device()
            mp = ModelPatcher(self.xl_base_patched.unet, device, "cpu", size=1)

            w = float(merge_data["base"]["weight"]) * norm
            for m in merge_data["models"]:
                print(f"Merging {m['name']} ({round(m['weight'] * norm * 100)}%)")
                filename = Path(path_manager.model_paths["modelfile_path"] / m["name"])
                # FIXME add error check?`
                with torch.torch.inference_mode():
                    m_unet, m_clip, m_vae, m_clip_vision = load_checkpoint_guess_config(str(filename))
                del m_clip
                del m_vae
                del m_clip_vision
                kp = m_unet.get_key_patches("diffusion_model.")
                for k in kp:
                    mp.model.add_patches({k: kp[k]}, strength_patch=float(m['weight'] * norm), strength_model=w)
                del m_unet
                w = 1.0

            self.xl_base = self.StableDiffusionModel(
                unet=mp.model, clip=clip, vae=vae, clip_vision=clip_vision
            )

        if "loras" in merge_data and len(merge_data["loras"]) > 0:
            loras = [(x.get("name"), x.get("weight")) for x in merge_data["loras"]]
            self.load_loras(loras)
            self.xl_base = self.xl_base_patched

        if 'cache' in merge_data and merge_data['cache'] == True:
            filename = str(Path(path_manager.model_paths["cache_path"] / "merges" / Path(name).name).with_suffix(".safetensors"))
            print(f"Saving merged model: {filename}")
            with torch.torch.inference_mode():
                save_checkpoint(
                    filename,
                    self.xl_base.unet,
                    clip=self.xl_base.clip,
                    vae=self.xl_base.vae,
                    clip_vision=self.xl_base.clip_vision,
                    metadata={"rf_merge_data": str(merge_data)}
                )

        return


    def load_base_model(self, name):
        if self.xl_base_hash == name and self.xl_base_patched_extra == set():
            return

        filename = os.path.join(path_manager.model_paths["modelfile_path"], name)
        if Path(filename).suffix == '.merge':
            self.merge_models(name)
            return

        print(f"Loading base model: {name}")

        self.xl_base_patched = None
        self.xl_base_patched_hash = ""
        self.xl_base_patched_extra = set()
        self.conditions = None

        try:
            with torch.torch.inference_mode():
                unet, clip, vae, clip_vision = load_checkpoint_guess_config(filename)
            self.xl_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
            )
            if not (
                isinstance(self.xl_base.unet.model, SDXL) or
                isinstance(self.xl_base.unet.model, SD3)
            ):
                print(
                    "Model not supported. RuinedFooocus only support SDXL/SD3 models as the base model."
                )
                self.xl_base = None

            if self.xl_base is not None:
                self.xl_base_hash = name
                self.xl_base_patched = self.xl_base
                self.xl_base_patched_hash = ""
                # self.xl_base_patched.unet.model.to("cuda")
                print(f"Base model loaded: {self.xl_base_hash}")

        except:
            print(f"Failed to load {name}, loading default model instead")
            self.load_base_model(
                path_manager.default_model_names["default_base_model_name"]
            )

        return

    def freeu(self, model, b1, b2, s1, s2):
        freeu_model = FreeU()
        unet = freeu_model.patch(model=model.unet, b1=b1, b2=b2, s1=s1, s2=s2)[0]
        return self.StableDiffusionModel(
            unet=unet, clip=model.clip, vae=model.vae, clip_vision=model.clip_vision
        )

    def load_loras(self, loras):
        loaded_loras = []

        model = self.xl_base
        for name, weight in loras:
            if name == "None" or weight == 0:
                continue
            filename = os.path.join(path_manager.model_paths["lorafile_path"], name)
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

        name = modules.controlnet.get_model(name)

        if name is not None and self.xl_controlnet_hash != name:
            filename = os.path.join(path_manager.model_paths["controlnet_path"], name)
            self.xl_controlnet = comfy.controlnet.load_controlnet(filename)
            self.xl_controlnet_hash = name
            print(f"ControlNet model loaded: {self.xl_controlnet_hash}")
        if self.xl_controlnet_hash != name:
            self.xl_controlnet = None
            self.xl_controlnet_hash = None
            print(f"Controlnet model unloaded")

    conditions = None

    def textencode(self, id, text, clip_skip):
        update = False
        hash = f"{text} {clip_skip}"
        if hash != self.conditions[id]["text"]:
            self.xl_base_patched.clip = CLIPSetLastLayer().set_last_layer(
                self.xl_base_patched.clip, clip_skip * -1
            )[0]
            self.conditions[id]["cache"] = CLIPTextEncode().encode(
                clip=self.xl_base_patched.clip, text=text
            )[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    # From https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/lib_layerdiffusion/utils.py#L118
    def to_lora_patch_dict(self, state_dict: dict) -> dict:
        """Convert raw lora state_dict to patch_dict that can be applied on
        modelpatcher."""
        patch_dict = {}
        for k, w in state_dict.items():
            model_key, patch_type, weight_index = k.split("::")
            if model_key not in patch_dict:
                patch_dict[model_key] = {}
            if patch_type not in patch_dict[model_key]:
                patch_dict[model_key][patch_type] = [None] * 16
            patch_dict[model_key][patch_type][int(weight_index)] = w

        patch_flat = {}
        for model_key, v in patch_dict.items():
            for patch_type, weight_list in v.items():
                patch_flat[model_key] = (patch_type, weight_list)

        return patch_flat

    @torch.inference_mode()
    def process(
        self,
        positive_prompt,
        negative_prompt,
        input_image,
        controlnet,
        main_view,
        steps,
        width,
        height,
        image_seed,
        start_step,
        denoise,
        cfg,
        sampler_name,
        scheduler,
        clip_skip,
        callback,
        gen_data=None,
    ):
        try:
            if self.xl_base_patched == None or not (
                isinstance(self.xl_base_patched.unet.model, SDXL) or
                isinstance(self.xl_base_patched.unet.model, SD3)
                ):
                print(f"ERROR: Can only use SDXL or SD3 models")
                worker.interrupt_ruined_processing = True
                worker.outputs.append(
                    ["preview", (-1, f"Can only use SDXL or SD3 models ...", "error.png")]
                )
                return []
        except Exception as e:
            # Something went very wrong
            print(f"ERROR: {e}")
            worker.interrupt_ruined_processing = True
            worker.outputs.append(
                ["preview", (-1, f"Error when trying to use model ...", "error.png")]
            )
            return []

        img2img_mode = False
        input_image_pil = None
        layerdiffuse_mode = False
        seed = image_seed if isinstance(image_seed, int) else random.randint(1, 2**32)

        worker.outputs.append(["preview", (-1, f"Processing text encoding ...", None)])
        updated_conditions = False
        if self.conditions is None:
            self.conditions = clean_prompt_cond_caches()

        if self.textencode("+", positive_prompt, clip_skip):
            updated_conditions = True
        if self.textencode("-", negative_prompt, clip_skip):
            updated_conditions = True

        prompt_switch_mode = False
        if "[" in positive_prompt and "]" in positive_prompt:
            prompt_switch_mode = True

        if prompt_switch_mode and controlnet is not None and input_image is not None:
            print(
                "ControlNet and [prompt|switching] do not work well together. ControlNet will be applied to the first prompt only."
            )

        if prompt_switch_mode:
            prompt_switch_mode = True
            prompt_per_step = pp.prompt_switch_per_step(positive_prompt, steps)

            perc_per_step = round(100 / steps, 2)
            positive_complete = []
            for i in range(len(prompt_per_step)):
                if self.textencode("switch", prompt_per_step[i], clip_skip):
                    updated_conditions = True
                positive_switch = convert_cond(self.conditions["switch"]["cache"])
                start_perc = round((perc_per_step * i) / 100, 2)
                end_perc = round((perc_per_step * (i + 1)) / 100, 2)
                if end_perc >= 0.99:
                    end_perc = 1
                positive_switch = set_timestep_range(
                    positive_switch, start_perc, end_perc
                )

                positive_complete += positive_switch

            positive_switch = convert_cond(self.conditions["switch"]["cache"])

        device = comfy.model_management.get_torch_device()

        if controlnet is not None and "type" in controlnet and input_image is not None:
            worker.outputs.append(["preview", (-1, f"Powering up ...", None)])
            input_image_pil = input_image.convert("RGB")
            input_image = np.array(input_image_pil).astype(np.float32) / 255.0
            input_image = torch.from_numpy(input_image)[None,]
            input_image = ImageScaleToTotalPixels().upscale(
                image=input_image, upscale_method="bicubic", megapixels=1.0
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
                if prompt_switch_mode:
                    (
                        self.conditions["+"]["cache"],
                        self.conditions["-"]["cache"],
                    ) = ControlNetApplyAdvanced().apply_controlnet(
                        positive=positive_complete,
                        negative=self.conditions["-"]["cache"],
                        control_net=self.xl_controlnet,
                        image=input_image,
                        strength=float(controlnet["strength"]),
                        start_percent=float(controlnet["start"]),
                        end_percent=float(controlnet["stop"]),
                    )
                    self.conditions["+"]["text"] = None
                    self.conditions["-"]["text"] = None
                else:
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

            if controlnet["type"].lower() == "img2img":
                latent = VAEEncode().encode(
                    vae=self.xl_base_patched.vae, pixels=input_image
                )[0]
                force_full_denoise = False
                denoise = float(controlnet.get("denoise", controlnet.get("strength")))
                img2img_mode = True

        if controlnet is not None and "type" in controlnet:
            if controlnet["type"].lower() == "layerdiffuse":
                if not "layerdiffuse" in self.xl_base_patched_extra:
                    print(f"DEBUG: add layerdiffuse")
                    tmodel = ModelPatcher(
                        self.xl_base_patched.unet, device, "cpu", size=1
                    )
                    layer_lora_state_dict = load_torch_file(
                        "models/layerdiffuse/layer_xl_transparent_attn.safetensors"
                    )
                    layer_lora_patch_dict = self.to_lora_patch_dict(
                        layer_lora_state_dict
                    )
                    # weight = 1.0
                    tmodel.model.add_patches(layer_lora_patch_dict)
                    self.xl_base_patched.unet = tmodel.model
                    self.xl_base_patched_extra.add("layerdiffuse")

                    # load transparent vae
                    self.xl_base_patched.tvae = TransparentVAEDecoder(
                        load_torch_file(
                            "models/layerdiffuse/vae_transparent_decoder.safetensors"
                        ),
                        device=comfy.model_management.get_torch_device(),
                        dtype=(
                            torch.float16
                            if comfy.model_management.should_use_fp16()
                            else torch.float32
                        ),
                    )
                layerdiffuse_mode = True
            else:
                print(f"DEBUG: remove layerdiffuse")
                # FIXME try reloading model? (and loras)
                if "layerdiffuse" in self.xl_base_patched_extra:
                    self.xl_base_patched_extra.remove("layerdiffuse")
                # self.xl_base_patched.tvae = None

        if not img2img_mode:
            if isinstance(self.xl_base.unet.model, SDXL):
                latent = EmptyLatentImage().generate(
                    width=width, height=height, batch_size=1
                )[0]
            elif isinstance(self.xl_base.unet.model, SD3):
                latent = EmptySD3LatentImage().generate(
                    width=width, height=height, batch_size=1
                )[0]
            force_full_denoise = False
            denoise = None

        if gen_data["inpaint_toggle"]:
            mask = gen_data["inpaint_view"]["mask"]
            mask = mask[:, :, 0]
            mask = torch.from_numpy(mask)[None,] / 255.0

            image = gen_data["inpaint_view"]["image"]
            image = image[..., :-1]
            image = torch.from_numpy(image)[None,] / 255.0

            latent = VAEEncodeForInpaint().encode(
                vae=self.xl_base_patched.vae,
                pixels=image,
                mask=mask,
                grow_mask_by=20,
            )[0]

        latent_image = latent["samples"]
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        previewer = get_previewer(device, self.xl_base_patched.unet.model.latent_format)

        pbar = comfy.utils.ProgressBar(steps)

        def callback_function(step, x0, x, total_steps):
            y = None
            if previewer:
                y = previewer.preview(x0, step, total_steps)
            if callback is not None:
                callback(step, x0, x, total_steps, y)
            pbar.update_absolute(step + 1, total_steps, None)

        if noise_mask is not None:
            noise_mask = prepare_mask(noise_mask, noise.shape, device)

        worker.outputs.append(["preview", (-1, f"Prepare models ...", None)])
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
        comfy.model_management.load_models_gpu(self.models)

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        # FIXME: convert_cond() doesn't seem to be used anymore, will probably break prompt_switch_mode
        #        if prompt_switch_mode:
        #            positive_copy = positive_complete
        #        else:
        #            positive_copy = convert_cond(self.conditions["+"]["cache"])
        #        negative_copy = convert_cond(self.conditions["-"]["cache"])

        kwargs = {
            "cfg": cfg,
            "latent_image": latent_image,
            "start_step": start_step,
            "last_step": steps,
            "force_full_denoise": force_full_denoise,
            "denoise_mask": noise_mask,
            "sigmas": None,
            "disable_pbar": False,
            "seed": seed,
        }
        sampler = KSampler(
            self.xl_base_patched.unet,
            steps=steps,
            device=device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            model_options=self.xl_base_patched.unet.model_options,
        )
        extra_kwargs = {
            "callback": callback_function,
        }
        kwargs.update(extra_kwargs)

        worker.outputs.append(["preview", (-1, f"Start sampling ...", None)])
        samples = sampler.sample(
            noise,
            self.conditions["+"]["cache"],
            self.conditions["-"]["cache"],
            **kwargs,
        )

        samples = samples.cpu()

        cleanup_additional_models(self.models)

        sampled_latent = latent.copy()
        sampled_latent["samples"] = samples

        worker.outputs.append(["preview", (-1, f"VAE decoding ...", None)])

        decoded_latent = VAEDecode().decode(
            samples=sampled_latent, vae=self.xl_base_patched.vae
        )[0]

        images = [
            np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8)
            for y in decoded_latent
        ]

        preview = None
        if layerdiffuse_mode:
            renderer = ImageRenderer(self.xl_base_patched)
            preview, img = renderer.render_diffuse_image(
                input_image_pil, samples, decoded_latent
            )

            images = [img]

        if callback is not None:
            callback(steps, 0, 0, steps, images[0] if preview is None else preview)

        return images
