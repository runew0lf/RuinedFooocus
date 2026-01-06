import gradio as gr
from pathlib import Path
from modules.sdxl_styles import load_styles
from modules.interrogate import looks

from shared import state, add_setting, performance_settings, resolution_settings, path_manager, settings, models, translate

t = translate

def save_clicked(*args):
    ui_data = {}
    # Overwrite current settings
    for key, val in zip(state["setting_name"], args):
        settings.default_settings[key] = val

        # Massage some of the data. Settings that are lists should be split
        if key in [
            "archive_folders",
            "path_checkpoints",
            "path_loras"
        ]:
            settings.default_settings[key] = settings.default_settings[key].splitlines()

        # Remove empty keys
        if settings.default_settings[key] == None or settings.default_settings[key] == "":
            settings.default_settings.pop(key)
            continue

        # Move ui_* and path_*
        if key.startswith("ui_"):
            ui_data[key] = settings.default_settings.get(key)
            settings.default_settings.pop(key)
        if key.startswith("path_"):
            path_manager.paths[key] = settings.default_settings.get(key)
            settings.default_settings.pop(key)

    settings.set_settings_path(ui_data.get("ui_settings_name", None))
    settings.save_settings()
    path_manager.set_settings_path(ui_data.get("ui_settings_name", None))
    path_manager.save_paths()

    print(t("Saved new settings to {path}.", mapping={"path": settings.settings_path}))
    gr.Info(t("Saved new settings to {path}.", mapping={"path": settings.settings_path}))

def create_settings():
    with gr.Blocks() as app_settings:
        with gr.Row():
            with gr.Column():
                gr.Markdown(t("# UI settings"))
                with gr.Row():
                    image_number = gr.Number(label=t("Image Number"), interactive=True, value=settings.default_settings.get("image_number", 1))
                    add_setting("image_number", image_number)
                    image_number_max = gr.Number(label=t("Image Number Max"), interactive=True, value=settings.default_settings.get("image_number_max", 50))
                    add_setting("image_number_max", image_number_max)
                with gr.Row():
                    seed = gr.Number(label=t("Seed"), interactive=True, value=settings.default_settings.get("seed", -1))
                    add_setting("seed", seed)
                    seed_random = gr.Checkbox(label=t("Random Seed"), interactive=True, value=settings.default_settings.get("seed_random", True))
                    add_setting("seed_random", seed_random)
                style = gr.Dropdown(
                    label=t("Style Selection"),
                    multiselect=True,
                    container=True,
                    choices=list(load_styles().keys()),
                    value=list(
                        set(settings.default_settings.get("style", [])) &
                        set(load_styles().keys())
                    ),
                )
                add_setting("style", style)
                prompt = gr.Textbox(label=t("Prompt"), interactive=True, value=settings.default_settings.get("prompt", ""))
                add_setting("prompt", prompt)
                negative_prompt = gr.Textbox(label=t("Negative Prompt"), interactive=True, value=settings.default_settings.get("negative_prompt", ""))
                add_setting("negative_prompt", negative_prompt)
                auto_negative_prompt = gr.Checkbox(label=t("Auto Negative Prompt"), interactive=True, value=settings.default_settings.get("auto_negative_prompt", False))
                add_setting("auto_negative_prompt", auto_negative_prompt)
                performance = gr.Dropdown(
                    label=t("Performance"),
                    interactive=True,
                    choices=list(performance_settings.performance_options.keys()),
                    value=settings.default_settings.get("performance", "Speed"),
                )
                add_setting("performance", performance)
                resolution = gr.Dropdown(
                    label=t("Resolution"),
                    interactive=True,
                    choices=list(resolution_settings.aspect_ratios.keys()),
                    value=settings.default_settings.get("resolution", "1344x768 (16:9)"),
                )
                add_setting("resolution", resolution)

                with gr.Row():
                    lora_min = gr.Number(label=t("LoRA weight min"), interactive=True, value=settings.default_settings.get("lora_min", 0))
                    add_setting("lora_min", lora_min)
                    lora_max = gr.Number(label=t("LoRA weight max"), interactive=True, value=settings.default_settings.get("lora_max", 2))
                    add_setting("lora_max", lora_max)

                gr.Markdown(t("# Models to load at startup"))
                base_model = gr.Dropdown(
                    label=t("Base Model"),
                    interactive=True,
                    choices=models.names['checkpoints'],
                    value=settings.default_settings.get("base_model", "sd_xl_base_1.0_0.9vae.safetensors"),
                )
                add_setting("base_model", base_model)
                with gr.Row():
                    lora_1_model = gr.Dropdown(
                        label=t("LoRA {id} Model", mapping={'id': 1}),
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_1_model", "None"),
                    )
                    lora_1_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 1}), value=settings.default_settings.get("lora_1_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_2_model = gr.Dropdown(
                        label=t("LoRA {id} Model", mapping={'id': 2}),
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_2_model", "None"),
                    )
                    lora_2_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 2}), value=settings.default_settings.get("lora_2_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_3_model = gr.Dropdown(
                        label=t("LoRA {id} Model", mapping={'id': 3}),
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_3_model", "None"),
                    )
                    lora_3_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 3}), value=settings.default_settings.get("lora_3_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_4_model = gr.Dropdown(
                        label=t("LoRA {id} Model", mapping={'id': 4}),
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_4_model", "None"),
                    )
                    lora_4_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 4}), value=settings.default_settings.get("lora_4_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_5_model = gr.Dropdown(
                        label=t("LoRA {id} Model", mapping={'id': 5}),
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_5_model", "None"),
                    )
                    lora_5_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 5}), value=settings.default_settings.get("lora_5_weight", 1.0), step=0.05)

                add_setting("lora_1_model", lora_1_model)
                add_setting("lora_2_model", lora_2_model)
                add_setting("lora_3_model", lora_3_model)
                add_setting("lora_4_model", lora_4_model)
                add_setting("lora_5_model", lora_5_model)
                add_setting("lora_1_weight", lora_1_weight)
                add_setting("lora_2_weight", lora_2_weight)
                add_setting("lora_3_weight", lora_3_weight)
                add_setting("lora_4_weight", lora_4_weight)
                add_setting("lora_5_weight", lora_5_weight)

            with gr.Column():
                gr.Markdown(t("# One Button Prompt"))
                OBP_preset = gr.Textbox(label=t("OBP Preset"), value=settings.default_settings.get("OBP_preset", "Standard"))
                add_setting("OBP_preset", OBP_preset)
                hint_chance = gr.Number(label=t("Hint Chance"), value=settings.default_settings.get("hint_chance", 25))
                add_setting("hint_chance", hint_chance)
                
                gr.Markdown(t("# Image Browser"))
                images_per_page = gr.Number(label=t("Images per page"), value=settings.default_settings.get("images_per_page", 100), minimum=1, maximum=1000, step=1)
                add_setting("images_per_page", images_per_page)
                archive_folders = gr.Code(
                    label=t("Archive Folders"),
                    interactive=True,
                    value="\n".join(settings.default_settings.get("archive_folders", [])),
                    lines=5,
                    max_lines=5
                )
                add_setting("archive_folders", archive_folders)
                gr.Markdown(t("# Paths"))
                path_checkpoints = gr.Code(
                    label=t("Checkpoint Folders"),
                    interactive=True,
                    value="\n".join(path_manager.paths.get("path_checkpoints", [])),
                    lines=5,
                    max_lines=5
                )
                add_setting("path_checkpoints", path_checkpoints)
                path_loras = gr.Code(
                    label=t("LoRA Folders"),
                    interactive=True,
                    value="\n".join(path_manager.paths.get("path_loras", [])),
                    lines=5,
                    max_lines=5
                )
                add_setting("path_loras", path_loras)
                path_inbox = gr.Textbox(label=t("Inbox Folder"), interactive=True, placeholder="", value=path_manager.paths.get("path_inbox", "../models/inbox/"))
                add_setting("path_inbox", path_inbox)
                path_outputs = gr.Textbox(label=t("Output Folder"), interactive=True, placeholder="", value=path_manager.paths.get("path_outputs", "../outputs/"))
                add_setting("path_outputs", path_outputs)

                gr.Markdown(t("# Chatbot settings"))
                curr_localfile = settings.default_settings.get("llama_localfile", None)
                llama_localfile = gr.Dropdown(label="Local Llama file", interactive=True, choices=[curr_localfile, None]+path_manager.get_folder_list("llm"), value=curr_localfile,)
                add_setting("llama_localfile", llama_localfile)
                llm_n_predict = gr.Number(label="n_predict", interactive=True, placeholder=-1, value=settings.default_settings.get("llm_n_predict", None), minimum=-1, step=1)
                add_setting("llm_n_predict", llm_n_predict)
                llm_n_ctx = gr.Number(label="n_ctx", interactive=True, placeholder=2048, value=settings.default_settings.get("llm_n_ctx", None), minimum=0, step=1)
                add_setting("llm_n_ctx", llm_n_ctx)
                llm_n_gpu_layers = gr.Number(label="n_gpu_layers", interactive=True, placeholder=0, value=settings.default_settings.get("llm_n_gpu_layers", None), minimum=-1, step=1)
                add_setting("llm_n_gpu_layers", llm_n_gpu_layers)
                llm_chat_history = gr.Number(label="chat_history", interactive=True, placeholder=7, value=settings.default_settings.get("llm_chat_history", None), minimum=0, step=1)
                add_setting("llm_chat_history", llm_chat_history)
                enable_llm_tools = gr.Checkbox(label=t("Enable image generation"), value=settings.default_settings.get("enable_llm_tools", False))
                add_setting("enable_llm_tools", enable_llm_tools)

                gr.Markdown(t("# Other"))
                interrogator = gr.Dropdown(label=t("Default Interrogator"), interactive=True, choices=list(looks.keys()), value=settings.default_settings.get("interrogator", None),)
                add_setting("interrogator", interrogator)
                save_metadata = gr.Checkbox(label=t("Save Metadata"), value=settings.default_settings.get("save_metadata", True))
                add_setting("save_metadata", save_metadata)
                theme = gr.Textbox(label=t("Theme"), interactive=True, value=settings.default_settings.get("theme", None))
                add_setting("theme", theme)

            with gr.Column():
                gr.Markdown(t("# Clip"))
                clip_aura = gr.Dropdown(label="clip_aura", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_aura", None),)
                add_setting("clip_aura", clip_aura)
                clip_g = gr.Dropdown(label="clip_g", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_g", None),)
                add_setting("clip_g", clip_g)
                clip_gemma = gr.Dropdown(label="clip_gemma", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_gemma", None),)
                add_setting("clip_gemma", clip_gemma)
                clip_l = gr.Dropdown(label="clip_l", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_l", None),)
                add_setting("clip_l", clip_l)
                clip_llama = gr.Dropdown(label="clip_llama", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_llama", None),)
                add_setting("clip_llama", clip_llama)
                clip_llava = gr.Dropdown(label="clip_llava", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_llava", None),)
                add_setting("clip_llava", clip_llava)
                clip_mistral3 = gr.Dropdown(label="clip_mistral3", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_mistral3", None),)
                add_setting("clip_mistral3", clip_mistral3)
                clip_qwen25 = gr.Dropdown(label="clip_qwen25", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_qwen25", None),)
                add_setting("clip_qwen25", clip_qwen25)
                clip_qwen3 = gr.Dropdown(label="clip_qwen3", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_qwen3", None),)
                add_setting("clip_qwen3", clip_qwen3)
                clip_oldt5 = gr.Dropdown(label="clip_oldt5", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_oldt5", None),)
                add_setting("clip_oldt5", clip_oldt5)
                clip_t5 = gr.Dropdown(label="clip_t5", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_t5", None),)
                add_setting("clip_t5", clip_t5)
                clip_umt5 = gr.Dropdown(label="clip_umt5", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_umt5", None),)
                add_setting("clip_umt5", clip_umt5)
                clip_vision = gr.Dropdown(label="clip_vision", interactive=True, choices=[None]+path_manager.get_folder_list("clip_vision"), value=settings.default_settings.get("clip_vision", None),)
                add_setting("clip_vision", clip_vision)

                gr.Markdown(t("# Shift"))
                auraflow_shift = gr.Textbox(label="AuraFlow shift", interactive=True, placeholder=1.73, value=settings.default_settings.get("auraflow_shift", None))
                add_setting("auraflow_shift", auraflow_shift)
                lumina2_shift = gr.Textbox(label="Lumina2 shift", interactive=True, placeholder=3.0, value=settings.default_settings.get("lumina2_shift", None))
                add_setting("lumina2_shift", lumina2_shift)
                hidream_shift = gr.Textbox(label="HiDream shift", interactive=True, placeholder=3.0, value=settings.default_settings.get("hidream_shift", None))
                add_setting("hidream_shift", hidream_shift)
                sd3_shift = gr.Textbox(label="SD3 shift", interactive=True, placeholder=3.0, value=settings.default_settings.get("sd3_shift", None))
                add_setting("sd3_shift", sd3_shift)
                qwen_image_shift = gr.Textbox(label="Qwen Image shift", interactive=True, placeholder=3.1, value=settings.default_settings.get("qwen_image_shift", None))
                add_setting("qwen_image_shift", qwen_image_shift)

                gr.Markdown(t("# VAE"))
                vae_auraflow = gr.Dropdown(label="AuraFlow VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_auraflow", None),)
                add_setting("vae_auraflow", vae_auraflow)
                vae_flux = gr.Dropdown(label="Flux VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_flux", None),)
                add_setting("vae_flux", vae_flux)
                vae_flux2 = gr.Dropdown(label="Flux2 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_flux2", None),)
                add_setting("vae_flux2", vae_flux2)
                vae_hunyuan_video = gr.Dropdown(label="Hunyuan Video VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_hunyuan_video", None),)
                add_setting("vae_hunyuan_video", vae_hunyuan_video)
                vae_lumina2 = gr.Dropdown(label="Lumina2 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_lumina2", None),)
                add_setting("vae_lumina2", vae_lumina2)
                vae_qwen_image = gr.Dropdown(label="Qwen Image VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_qwen_image", None),)
                add_setting("vae_qwen_image", vae_qwen_image)
                vae_pixart = gr.Dropdown(label="PixArt VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_pixart", None),)
                add_setting("vae_pixart", vae_pixart)
                vae_sd = gr.Dropdown(label="SD1.5 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_sd", None),)
                add_setting("vae_sd", vae_sd)
                vae_sd3 = gr.Dropdown(label="SD3 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_sd3", None),)
                add_setting("vae_sd3", vae_sd3)
                vae_sdxl = gr.Dropdown(label="SDXL/Pony/Illustrious VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_sdxl", None),)
                add_setting("vae_sdxl", vae_sdxl)
                vae_wan = gr.Dropdown(label="WAN 2.1 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_wan", None),)
                add_setting("vae_wan", vae_wan)

        with gr.Row(), gr.Group():
            ui_settings_name = gr.Text(
                label=t("Name"),
                interactive=True,
                placeholder=t("Optional"),
                value=settings.name,
            )
            add_setting("ui_settings_name", ui_settings_name)
            save_btn = gr.Button(t("Save"))

# Deal with this later
#            output = gr.Textbox(label="Status")
#            download_file = gr.File(label="Download File")
#        with gr.Row():
#            upload = gr.File(label="Load settings file (optional)", file_count="single", type="filepath")
#            download_btn = gr.Button("Download current settings")


        save_btn.click(
            fn=save_clicked,
            api_visibility='undocumented',
            inputs=state["setting_obj"],
        )

    return app_settings


