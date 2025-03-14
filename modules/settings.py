import json
from os.path import exists

from shared import path_manager

DEFAULT_SETTINGS = {
    "advanced_mode": False,
    "image_number": 1,
    "seed_random": True,
    "seed": 0,
    "style": ["Style: sai-cinematic"],
    "prompt": "",
    "negative_prompt": "",
    "performance": "Speed",
    "resolution": "1152x896 (4:3)",
    "base_model": path_manager.default_model_names["default_base_model_name"],
    "lora_1_model": "None",
    "lora_1_weight": path_manager.default_model_names["default_lora_weight"],
    "lora_2_model": "None",
    "lora_2_weight": path_manager.default_model_names["default_lora_weight"],
    "lora_3_model": "None",
    "lora_3_weight": path_manager.default_model_names["default_lora_weight"],
    "lora_4_model": "None",
    "lora_4_weight": path_manager.default_model_names["default_lora_weight"],
    "lora_5_model": "None",
    "lora_5_weight": path_manager.default_model_names["default_lora_weight"],
    "theme": "None",
    "auto_negative_prompt": False,
    "OBP_preset": "Standard",
    "hint_chance": 25,
    "clip_g": "clip_g.safetensors",
    "clip_l": "clip_l.safetensors",
    "clip_t5": "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
}


def load_settings():
    if exists("settings/settings.json"):
        with open("settings/settings.json") as f:
            settings = json.load(f)
    else:
        settings = {}

    # Add any missing default settings
    changed = False
    for key, value in DEFAULT_SETTINGS.items():
        if key not in settings:
            settings[key] = value
            changed = True

    # Some sanity checks
    if not isinstance(settings["style"], list):
        settings["style"] = []

    if changed:
        with open("settings/settings.json", "w") as f:
            json.dump(settings, f, indent=2)

    return settings


default_settings = load_settings()
