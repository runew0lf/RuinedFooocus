from .util import remove_empty_str
from comfy.model_patcher import ModelPatcher
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import comfy.model_management as model_management
from transformers.generation.logits_process import LogitsProcessorList
from comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, DualCLIPLoaderGGUF, GGUFModelPatcher
from comfyui_gguf.ops import GGMLOps
from comfy.sd import CLIPType, load_clip
from shared import path_manager, settings
import os
import random
import sys
import torch
import math


fooocus_expansion_path = "prompt_expansion"

SEED_LIMIT_NUMPY = 2**32
neg_inf = -8192.0


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace("  ", " ")
    return x.strip(",. \r\n")


class FooocusExpansion:
    tokenizer = None
    model = None

    def __init__(self):
        self.load_model_and_tokenizer(fooocus_expansion_path)
        self.offload_device = model_management.text_encoder_offload_device()
        self.patcher = ModelPatcher(
            self.model,
            load_device=self.model.device,
            offload_device=self.offload_device,
        )

    @classmethod
    def load_model_and_tokenizer(cls, model_path):
        if cls.tokenizer is None or cls.model is None:
            cls.tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls.model = AutoModelForCausalLM.from_pretrained(model_path)
            cls.model.to("cpu")

    def __call__(self, prompt, seed):
        seed = int(seed) % SEED_LIMIT_NUMPY
        set_seed(seed)
        positive_words = (
            open(os.path.join(fooocus_expansion_path, "positive.txt"), encoding="utf-8")
            .read()
            .splitlines()
        )
        positive_words = ["Ġ" + x.lower() for x in positive_words if x != ""]
        self.logits_bias = (
            torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + neg_inf
        )
        debug_list = []
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0
                debug_list.append(k[1:])
        # print(f'Expansion: Vocab with {len(debug_list)} words.')

        text = safe_str(prompt) + ","
        tokenized_kwargs = self.tokenizer(text, return_tensors="pt")
        tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(
            self.patcher.load_device
        )
        tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data[
            "attention_mask"
        ].to(self.patcher.load_device)
        current_token_length = int(tokenized_kwargs.data["input_ids"].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length
        try:
            features = self.model.generate(
                **tokenized_kwargs,
                top_k=100,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                logits_processor=LogitsProcessorList([self.logits_processor])
            )

            response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
            result = safe_str(response[0])
        except Exception as e:
            print(f"Error during prompt expansion: {e}")
            result = prompt
        return result

    def logits_processor(self, input_ids, scores):
        assert scores.ndim == 2 and scores.shape[0] == 1
        self.logits_bias = self.logits_bias.to(scores)

        bias = self.logits_bias.clone()
        bias[0, input_ids[0].to(bias.device).long()] = neg_inf
        bias[0, 11] = 0
        return scores + bias


class PromptExpansion:
    # Define the expected input types for the node
    @staticmethod
    @torch.no_grad()
    def expand_prompt(text):
        expansion = FooocusExpansion()

        prompt = remove_empty_str([safe_str(text)], default="")[0]

        max_seed = int(1024 * 1024 * 1024)
        seed = random.randint(1, max_seed)
        if seed < 0:
            seed = -seed
        seed = seed % max_seed

        expansion_text = expansion(prompt, seed)
        final_prompt = expansion_text

        return final_prompt


class Erniehancer:
    def __init__(self):
        self.clip = None
        self.clip_names = []
        self.clip_paths = []
        self.clip_type = CLIPType.FLUX2

        clip_name = settings.default_settings.get("clip_ernie_enhancer", "ernie-image-prompt-enhancer.safetensors")
        self.clip_names.append(str(clip_name))
        clip_path = path_manager.get_folder_file_path(
            "clip",
            clip_name,
            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
        )
        self.clip_paths.append(str(clip_path))

        print(f"Loading Erniehancer: {self.clip_names}")
        if all(name.endswith(".safetensors") for name in self.clip_paths):
            model_options = {}
            device = model_management.get_torch_device()
#            if device == "cpu":
#                model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
            model_options["load_device"] = model_options["offload_device"] = device
            self.clip = load_clip(ckpt_paths=self.clip_paths, clip_type=self.clip_type, model_options=model_options)
        else:
            clip_loader = DualCLIPLoaderGGUF()
            self.clip = clip_loader.load_patcher(
                self.clip_paths,
                self.clip_type,
                clip_loader.load_data(clip_paths)
            )

    def execute(self, prompt, image=None, thinking=False):
        tokens = self.clip.tokenize(prompt, image=image, skip_template=False, min_length=1, thinking=thinking)

        # Get sampling parameters
        do_sample = True
        max_length = 512
        temperature = 0.6
        top_k = 64
        top_p = 0.8
        min_p = 0.05
        repetition_penalty = 1.5
        presence_penalty = 0.0
        seed = random.randint(1, 2**32)

        generated_ids = self.clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed
        )

        generated_text = self.clip.decode(generated_ids, skip_special_tokens=True)
        print(f"Erniehanced prompt:\n{generated_text}")
        return generated_text


# Define a mapping of node class names to their respective classes
