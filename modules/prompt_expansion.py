from .util import remove_empty_str
from comfy.model_patcher import ModelPatcher
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import comfy.model_management as model_management
from transformers.generation.logits_process import LogitsProcessorList
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


# Define a mapping of node class names to their respective classes
