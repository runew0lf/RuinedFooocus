import gc
import numpy as np
import os
import torch
import einops
import traceback
import cv2
import logging
import json

import modules.async_worker as worker
from modules.util import generate_temp_filename, TimeIt, get_checkpoint_hashes, get_lora_hashes
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
from comfy.sample import fix_empty_latent_channels
from comfy.sd import load_checkpoint_guess_config
from latent_preview import get_previewer
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

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, Noise_RandomNoise, BasicScheduler, KSamplerSelect, BasicGuider, CFGGuider
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVImgToVideo, LTXVConditioning, LTXVScheduler, LTXVConcatAVLatent, LTXVSeparateAVLatent
from comfy_extras.nodes_lt import ModelSamplingLTXV
from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio, LTXVAudioVAEDecode
from comfy_extras.nodes_video import CreateVideo
from comfy_api.latest import Types

class pipeline:
    pipeline_type = ["ltx2_video"]

    class StableDiffusionModel:
        def __init__(self, clip, unet, vae, audio_vae=None):
            self.clip = clip
            self.unet = unet
            self.vae = vae
            self.audio_vae = audio_vae

        def to_meta(self):
            if self.unet is not None:
                self.unet.model.to("meta")
            if self.clip is not None:
                self.clip.cond_stage_model.to("meta")
            if self.vae is not None:
                self.vae.first_stage_model.to("meta")
            if self.audio_vae is not None:
                self.audio_vae.first_stage_model.to("meta")

    clip = None
    vae = None
    audio_vae = None
    model_hash = ""
    model_base = None
    model_hash_patched = ""
    model_base_patched = None
    conditions = None

    ggml_ops = GGMLOps()
    logger = logging.getLogger()

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

                        clip_name = settings.default_settings.get("clip_ltx23_text_proj", "ltx-2.3_text_projection_bf16.safetensors")
                        clip_names.append(str(clip_name))
                        clip_path = path_manager.get_folder_file_path(
                            "clip",
                            clip_name,
                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                        )
                        clip_paths.append(str(clip_path))

                        vae_name = settings.default_settings.get("vae_ltxv23_video", "LTX23_video_vae_bf16.safetensors")
                        audio_vae_name = settings.default_settings.get("vae_ltxv23_audio", "LTX23_audio_vae_bf16.safetensors")
                    else:
                        print(f"ERROR: Not a LTX2 Video model?")
                        unet = None
                        return

                    print(f"Loading CLIP: {clip_names}")
                    if all(name.endswith(".safetensors") for name in clip_paths):
                        model_options = {}
                        device = comfy.model_management.get_torch_device()
                        if device == "cpu":
                            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
                        clip = comfy.sd.load_clip(ckpt_paths=clip_paths, clip_type=clip_type, model_options=model_options)
                    else:
                        clip_loader = DualCLIPLoaderGGUF()
                        self.logger.setLevel(logging.ERROR)
                        clip = clip_loader.load_patcher(
                            clip_paths,
                            comfy.sd.CLIPType.LTXV,
                            clip_loader.load_data(clip_paths)
                        )
                        self.logger.setLevel(logging.WARNING)

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

                    print(f"Loading Audio VAE: {audio_vae_name}")
                    if str(vae_path).endswith(".gguf"):
                        sd = load_gguf_sd(str(audio_vae_path))
                        metadata = None
                    else:
                        sd, metadata = comfy.utils.load_torch_file(str(audio_vae_path), return_metadata=True)

                    # https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/nodes.py#L2453C1-L2476C22
                    try:
                        sd_audio = comfy.utils.state_dict_prefix_replace(
                            dict(sd), {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True
                        )
                        audio_vae = VAE(sd=sd_audio, metadata=metadata)
                        audio_vae.throw_exception_if_invalid()
                    except Exception:
                        from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
                        audio_vae = AudioVAE(sd, metadata)

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
                unet=unet, clip=clip, vae=vae, audio_vae=audio_vae
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
                    audio_vae=None,
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

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)
        gen_data["frame_rate"] = float(settings.default_settings.get("video_fps", 30.0))
        frame_number = int(gen_data["original_image_number"]) * gen_data["frame_rate"]) # Generate "Frame number" seconds of video
        frame_number = ((frame_number // 8) * 8) + 1 # Make sure frame_number is divisible by 8 + 1
        gen_data["width"] = (gen_data["width"] // 32) * 32
        gen_data["height"] = (gen_data["height"] // 32) * 32

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

        with TimeIt("Text encoding"):
            self.textencode("+", positive_prompt, clip_skip)
            self.textencode("-", negative_prompt, clip_skip)

        pbar = comfy.utils.ProgressBar(gen_data["steps"])

        def callback_function(step, x0, x, total_steps):
            previewer = get_previewer(self.model_base_patched.unet.load_device, self.model_base_patched.unet.model.latent_format)

            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)

                y = (preview_buytes * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                y = einops.rearrange(y, 'b c t h w -> (b h) (t w) c')

                maxw = 1920
                maxh = 1080
                image = Image.fromarray(y)
                ow, oh = image.size
                scale = min(maxh / oh, maxw / ow)
                image = image.resize((int(ow * scale), int(oh * scale)), Image.LANCZOS)
            else:
                image = None

            status = "Generating video"
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

        print("Setting up latents and getting ready to sample.")

        # Video latent
        # i2v?
        if gen_data["input_image"]:
            image = np.array(gen_data["input_image"]).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            (positive, negative, video_latent) = LTXVImgToVideo().generate(
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
            video_latent = EmptyLTXVLatentVideo().generate(
                width = gen_data["width"],
                height = gen_data["height"],
                length = frame_number,
                batch_size = 1,
            )[0]
            positive = self.conditions["+"]["cache"]

        # Audio latent
        audio_latent = LTXVEmptyLatentAudio().execute(
            audio_vae = self.audio_vae,
            frames_number = frame_number,
            frame_rate = gen_data["frame_rate"],
            batch_size = 1,
        )[0]

        # Combine to video
        latent = LTXVConcatAVLatent().execute(
            video_latent = video_latent,
            audio_latent = audio_latent,
        )[0]

        negative = self.conditions["-"]["cache"]

        # LTXVConditioning
        positive, negative = LTXVConditioning().execute(
            positive = positive,
            negative = negative,
            frame_rate = gen_data["frame_rate"],
        )

        # Sampler
        ksampler = KSamplerSelect().get_sampler(
            sampler_name = gen_data["sampler_name"],
        )[0]

        # Sigmas
        sigmas = LTXVScheduler().execute(
            steps = gen_data["steps"],
            max_shift = 2.05,
            base_shift = 0.95,
            stretch = True,
            terminal = 0.1,
            latent = latent
        )[0]

        guider = CFGGuider().execute(
            model = self.model_base_patched.unet,
            cfg = float(gen_data["cfg"]),
            positive = positive,
            negative = negative,
        )[0]

        noise = Noise_RandomNoise(seed)

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Generating ...", None)
        )

        #
        # Sample
        #

        #denoised_output = SamplerCustomAdvanced().execute(
        #    noise = noise,
        #    guider = guider,
        #    sampler = ksampler,
        #    sigmas = sigmas,
        #    latent_image = latent,
        #)

        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = fix_empty_latent_channels(guider.model_patcher, latent_image, latent.get("downscale_ratio_spacial", None))
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        #callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        samples = guider.sample(
            noise.generate_noise(latent),
            latent_image,
            ksampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback_function,
            disable_pbar=False,
            seed=noise.seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        if "x0" in x0_output:
            x0_out = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            if samples.is_nested:
                latent_shapes = [x.shape for x in samples.unbind()]
                x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
            denoised_output = latent.copy()
            denoised_output["samples"] = x0_out
        else:
            denoised_output = out



        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE Decoding ...", None)
            )

        #video_latent, audio_latent = LTXVSeparateAVLatent().execute(
        samples = LTXVSeparateAVLatent().execute(
            av_latent = denoised_output,
        )

        # Decode video

        print(f"VAE decode video.")
        decoded_latent = VAEDecodeTiled().decode(
            samples=samples[0],
            tile_size=512,
            overlap=64,
            temporal_size=4096,
            temporal_overlap=8,
            vae=self.model_base_patched.vae,
        )[0]

        # Decode audio

        print(f"VAE decode audio.")
        audio = LTXVAudioVAEDecode().execute(
            samples = samples[1],
            audio_vae = self.audio_vae,
        )[0]

        # Create Video

        video = CreateVideo().execute(
            images = decoded_latent,
            fps = gen_data["frame_rate"],
            audio = audio,
        )[0]

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Saving ...", None)
            )

        filename = generate_temp_filename(
            folder=path_manager.model_paths["temp_outputs_path"], extension="tmp"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        print("Saving video")
        # Save MP4
        codec = "auto"
        try:
            loras = []
            for lora_data in gen_data["loras"] if gen_data["loras"] is not None else []:
                if len(lora_data[0]) == 64 and all(c in '0123456789abcdefABCDEF' for c in lora_data[0]): # Looks like sha256?
                    hash = lora_data[0]
                else:
                    hash = None
                w, l  = lora_data[1].split(" - ", 1)
                if not l == "None":
                    loras.append({"name": l, "weight": float(w), "hash": hash})
            data = {
                "Prompt": gen_data["positive_prompt"],
                "Negative": gen_data["negative_prompt"],
                "steps": gen_data["steps"],
                "cfg": gen_data["cfg"],
                "width": gen_data["width"],
                "height": gen_data["height"],
                "seed": abs(int(gen_data["seed"])),
                "sampler_name": gen_data["sampler_name"],
                "scheduler": gen_data["scheduler"],
                "base_model_name": gen_data["base_model_name"],
                "base_model_hash": get_checkpoint_hashes(gen_data["base_model_name"])['SHA256'],
                "loras": [[f"{get_lora_hashes(lora['name'])['SHA256']}", f"{lora['weight']} - {lora['name']}"] for lora in loras],
                "software": "RuinedFooocus",
            }
        except:
            data = {"prompt": gen_data["positive_prompt"], "software": "RuinedFooocus"}
        metadata = {"metadata": json.dumps(data)}

        video.save_to(
            filename.with_suffix(".mp4"),
            format = Types.VideoContainer.MP4,
            codec = Types.VideoCodec(codec),
            metadata = metadata
        )

        pil_images = []
        for image in decoded_latent:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        # Save GIF
        # FIXME: scale down, gifs are too big
        compress_level=9 # Min = 0, Max = 9
        pil_images[0].save(
            filename.with_suffix(".gif"),
            compress_level=compress_level,
            save_all=True,
            duration=int(1000.0/gen_data["frame_rate"]),
            append_images=pil_images[1:],
            optimize=True,
            loop=0,
        )

        return [str(filename.with_suffix(".gif"))]
