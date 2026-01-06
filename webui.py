import torchruntime
import platform
import os
from modules.shared_functions import broken_torch_platforms
gpus = torchruntime.device_db.get_gpus()
if "TORCH_PLATFORM" in os.environ:
    torch_platform = os.environ["TORCH_PLATFORM"]
else:
    torch_platform = torchruntime.platform_detection.get_torch_platform(gpus)
os_platform = platform.system()

# Some platform checks
torch_platform, os_platform = broken_torch_platforms(torch_platform, os_platform)

from argparser import args
import comfy.cli_args
comfy.cli_args.args.cpu = args.cpu
comfy.cli_args.args.highvram = args.highvram
comfy.cli_args.args.normalvram = args.normalvram
comfy.cli_args.args.lowvram = args.lowvram
comfy.cli_args.args.novram = args.novram
comfy.cli_args.args.reserve_vram = args.reserve_vram

comfy.cli_args.args.force_fp32 = args.force_fp32
comfy.cli_args.args.force_fp16 = args.force_fp16

comfy.cli_args.args.fp32_unet = args.fp32_unet
comfy.cli_args.args.fp64_unet = args.fp64_unet
comfy.cli_args.args.bf16_unet = args.bf16_unet
comfy.cli_args.args.fp16_unet = args.fp16_unet
comfy.cli_args.args.fp8_e4m3fn_unet = args.fp8_e4m3fn_unet
comfy.cli_args.args.fp8_e5m2_unet = args.fp8_e5m2_unet
comfy.cli_args.args.fp8_e8m0fnu_unet = args.fp8_e8m0fnu_unet

comfy.cli_args.args.fp16_vae = args.fp16_vae
comfy.cli_args.args.fp32_vae = args.fp32_vae
comfy.cli_args.args.bf16_vae = args.bf16_vae

comfy.cli_args.args.cpu_vae = args.cpu_vae

comfy.cli_args.args.fp8_e4m3fn_text_enc = args.fp8_e4m3fn_text_enc
comfy.cli_args.args.fp8_e5m2_text_enc = args.fp8_e5m2_text_enc
comfy.cli_args.args.fp16_text_enc = args.fp16_text_enc
comfy.cli_args.args.fp32_text_enc = args.fp32_text_enc
comfy.cli_args.args.bf16_text_enc = args.bf16_text_enc

# FIXME: Should brobably also check directml and other things...
if torch_platform == "cpu":
    comfy.cli_args.args.cpu = True

from pathlib import Path
import shared
from shared import (
    state,
    add_ctrl,
    performance_settings,
    resolution_settings,
    path_manager,
    settings,
)
import time
import json
import gradio as gr
import re
import traceback

import version
import modules.async_worker as worker
import modules.html
import modules.hints
import modules.ui.ui_onebutton as ui_onebutton
import modules.ui.ui_controlnet as ui_controlnet
from modules.api import add_api
from modules.interrogate import look

# Block the "Token indices sequence length is longer than the specified maximum sequence length for this model" warning
import transformers
transformers.utils.logging.set_verbosity_error()

from comfy.samplers import KSampler
from modules.sdxl_styles import load_styles, styles, allstyles, apply_style
from modules.prompt_processing import get_promptlist
from modules.util import (
    get_wildcard_files,
    load_keywords,
    get_checkpoint_thumbnail,
    get_lora_thumbnail,
    get_model_thumbnail,
    get_checkpoint_path,
    get_lora_path,
)
from modules.imagebrowser import ImageBrowser

import modules.ui.ui_image_gallery as ui_image_gallery
import modules.ui.ui_llama_chat as ui_llama_chat
import modules.ui.ui_settings as ui_settings

from PIL import Image

inpaint_toggle = None
shared.shared_cache["browser"] = ImageBrowser()
t = shared.translate

def find_unclosed_markers(s):
    markers = re.findall(r"__", s)
    for marker in markers:
        if s.count(marker) % 2 != 0:
            return s.split(marker)[-1]
    return None


def launch_app(args):
    inbrowser = not args.nobrowser
    favicon_path = "logo.ico"

    shared.gradio_root.queue(api_open=True)

    # Create theme for main interface
    if "theme" not in settings or settings["theme"] in ["", "None", None]:
        theme = gr.themes.Default()
    else:
        try:
            theme = gr.Theme.from_hub(settings["theme"])
        except:
            print(f"ERROR: Could not find theme {settings['theme']}. Check https://huggingface.co/spaces/gradio/theme-gallery for themes")
            theme = gr.themes.Default()

    # Override some settings
    theme.spacing_lg = '8px'
    theme.spacing_md = '6px'
    theme.spacing_sm = '4px'
    theme.spacing_xl = '8px'
    theme.spacing_xs = '2px'
    theme.spacing_xxl = '2px'
    theme.spacing_xxs = '1px' 
    theme.text_xxl = '8px' 

    # Create the image gallery from the new module
    app_image_browser = ui_image_gallery.create_image_gallery()
    app_llama_chat = ui_llama_chat.create_chat()
    app_settings = ui_settings.create_settings()

    main_tabs = gr.TabbedInterface(
        [shared.gradio_root, app_image_browser, app_llama_chat, app_settings],
        [t("Main"), t("Image browser"), t("Chat bots"), t("Settings")],
        title="RuinedFooocus " + version.version,
        analytics_enabled=False,
    )

    shared.server_app, shared.local_url, shared.share_url = main_tabs.launch(
        inbrowser=inbrowser,
        server_name=args.listen,
        server_port=args.port,
        share=args.share,
        auth=(
            args.auth.split("/", 1)
            if isinstance(args.auth, str) and "/" in args.auth
            else None
        ),
        theme=theme,
        favicon_path=favicon_path,
        css=modules.html.css,
        js=modules.html.scripts,
        allowed_paths=["html", "/", path_manager.model_paths["temp_outputs_path"]]
        + settings.get("archive_folders", []),
        enable_monitoring=False,
        pwa=True,
        mcp_server=args.mcp,
        prevent_thread_lock=True,
    )

def update_clicked():
    return {
        run_button: gr.update(interactive=False, visible='hidden'),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(
            visible=True,
            value=modules.html.make_progress_html(0, "Please wait ..."),
        ),
        gallery: gr.update(visible='hidden'),
        main_view: gr.update(visible=True, value="html/init_image.png"),
        inpaint_view: gr.update(
            visible='hidden',
            interactive=False,
        ),
        hint_text: gr.update(visible=True, value=modules.hints.get_hint()),
    }


def update_preview(product):
    percentage, title, image = product
    return {
        run_button: gr.update(interactive=False, visible='hidden'),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(
            visible=True, value=modules.html.make_progress_html(percentage, title)
        ),
        main_view: (
            gr.update(visible=True, value=image) if image is not None else gr.update()
        ),
    }


def update_results(product):
    metadata = {"Data": "Preview Grid"}
    if len(product) > 0:
        with Image.open(product[0]) as im:
            if im.info.get("parameters"):
                metadata = im.info["parameters"]

    return {
        run_button: gr.update(interactive=True, visible=True),
        stop_button: gr.update(interactive=False, visible='hidden'),
        progress_html: gr.update(visible='hidden'),
        main_view: gr.update(value=product[0]) if len(product) > 0 else gr.update(),
        inpaint_toggle: gr.update(value=False),
        gallery: gr.update(
            visible=True,
            selected_index=0,
            allow_preview=True,
            preview=True,
            value=product,
        ),
        metadata_json: gr.update(value=metadata),
    }


def append_work(gen_data):
    tmp_data = gen_data.copy()
    if tmp_data["obp_assume_direct_control"]:
        prompts = []
        prompts.append("")
    else:
        prompts = get_promptlist(tmp_data)
    idx = 0

    tmp_data["image_total"] = len(prompts) * tmp_data["image_number"]
    tmp_data["task_type"] = "process"
    if len(prompts) == 1:
        tmp_data["prompt"] = prompts[0]
    else:
        tmp_data["prompt"] = prompts

    task_id = worker.add_task(tmp_data.copy())
    return task_id


def generate_clicked(*args):
    global status

    yield update_clicked()
    gen_data = {}
    for key, val in zip(state["ctrls_name"], args):
        gen_data[key] = val

    # FIXME this is _ugly_ run_event gets triggerd once at page load
    #   not really gradios fault, we are doing silly things there. :)
    if gen_data["run_event"] < 1:
        yield update_results(["html/logo.png"])
        return

    gen_data["generate_forever"] = int(gen_data["image_number"]) == 0

    # Check for preset image
    if gen_data['preset_selection']:
        try:
            image = Image.open(gen_data['preset_selection'])
            info = image.info
            params = info.get("parameters", "")
            preset_data = json.loads(params)

            if preset_data["software"] == "RuinedFooocus":
                gen_data["steps"] = preset_data["steps"]
                gen_data["width"] = preset_data["width"]
                gen_data["height"] = preset_data["height"]
                gen_data["cfg"] = preset_data["cfg"]
                gen_data["sampler_name"] = preset_data["sampler_name"]
                gen_data["scheduler"] = preset_data["scheduler"]
                gen_data["clip_skip"] = preset_data["clip_skip"]
                gen_data["base_model_name"] = preset_data["base_model_name"]
                gen_data["base_model_hash"] = preset_data.get("base_model_hash", None)
                gen_data["loras"] = preset_data["loras"]
        except Exception as e:
            print(f"WARNING: Failed using preset: {e}")
            traceback.print_exc()
            pass

    task_id = append_work(gen_data)

    shared.state["interrupted"] = False
    finished = False

    while not finished:
        flag, product = worker.task_result(task_id)

        if flag == "preview":
            yield update_preview(product)

        elif flag == "results":
            yield update_results(product)
            finished = True
        time.sleep(0.1)

    shared.state["interrupted"] = False


settings = settings.default_settings

metadata_json = gr.Json()

shared.wildcards = get_wildcard_files()

shared.gradio_root = gr.Blocks()

with shared.gradio_root as block:
    block.load()
    run_event = gr.Number(visible='hidden', value=0)
    add_ctrl("run_event", run_event)

    def get_cfg_timestamp():
        return shared.state["last_config"]
    cfg_timestamp = gr.Textbox(visible='hidden', value=get_cfg_timestamp())
    cfg_timer = gr.Timer(value=5)
    cfg_timer.tick(fn=get_cfg_timestamp, api_visibility='undocumented', outputs=[cfg_timestamp])

    with gr.Row():
        with gr.Column(scale=5):
            main_view = gr.Image(
                elem_id="main_view",
                value="html/init_image.png",
                height=680,
                type="filepath",
                visible=True,
                buttons=['download', 'share', 'fullscreen'],
            )
            add_ctrl("main_view", main_view)
            inpaint_view = gr.ImageEditor(
                height=680,
                type="numpy",
                visible='hidden',
                show_label=False,
                buttons=['download', 'share', 'fullscreen'],
                layers=False,
                interactive=False,
                transforms=(),
                brush=gr.Brush(colors=["#000000"], color_mode="fixed"),
            )
            add_ctrl("inpaint_view", inpaint_view)

            progress_html = gr.HTML(
                value=modules.html.make_progress_html(32, "Progress 32%"),
                visible='hidden',
                padding=False,
                elem_id="progress-bar",
                elem_classes="progress-bar",
            )

            gallery = gr.Gallery(
                label=None,
                show_label=False,
                object_fit="scale-down",
                height=60,
                allow_preview=False,
                preview=False,
                interactive=False,
                visible=True,
                buttons=['download', 'share', 'fullscreen'],
            )

            @gallery.select(
                api_visibility='undocumented',
                inputs=[gallery],
                outputs=[main_view, metadata_json],
                show_progress="hidden",
            )
            def gallery_change(files, sd: gr.SelectData):
                name = sd.value["image"]["path"]
                with Image.open(name) as im:
                    if im.info.get("parameters"):
                        metadata = im.info["parameters"]
                    else:
                        metadata = {"Data": "Preview Grid"}
                return {
                    main_view: gr.update(value=name),
                    metadata_json: gr.update(value=metadata),
                }

            with gr.Row(elem_classes="type_row"):
                with gr.Column(scale=5):
                    with gr.Group():
                        with gr.Group(), gr.Row():
                            prompt = gr.Textbox(
                                show_label=False,
                                placeholder=t("Type prompt here."),
                                container=False,
                                autofocus=True,
                                elem_classes="type_row",
                                lines=5,
                                value=settings["prompt"],
                                scale=5,
                            )
                            add_ctrl("prompt", prompt)

                with gr.Column(scale=1, min_width=0):
                    run_button = gr.Button(value=t("Generate"), elem_id="generate")
                    stop_button = gr.Button(
                        value=t("Stop"), interactive=False, visible='hidden'
                    )
                    spellcheck = gr.Dropdown(
                        label="Wildcards",
                        visible='hidden',
                        choices=[],
                        value="",
                        scale=1,
                    )

                    @prompt.input(api_visibility='undocumented', inputs=prompt, outputs=spellcheck)
                    def checkforwildcards(text):
                        test = find_unclosed_markers(text)
                        if test is not None:
                            filtered = [s for s in shared.wildcards if test in s]
                            filtered.append(" ")
                            return {
                                spellcheck: gr.update(
                                    interactive=True,
                                    visible=True,
                                    choices=filtered,
                                    value=" ",
                                )
                            }
                        else:
                            return {
                                spellcheck: gr.update(interactive=False, visible='hidden')
                            }

                    @spellcheck.select(api_visibility='undocumented', inputs=[prompt, spellcheck], outputs=prompt)
                    def select_spellcheck(text, selection):
                        last_idx = text.rindex("__")
                        newtext = f"{text[:last_idx]}__{selection}__"
                        return {prompt: gr.update(value=newtext)}


                    @main_view.upload(
                        api_visibility='undocumented',
                        inputs=[main_view, prompt],
                        outputs=[prompt, gallery]
                    )
                    def load_images_handler(file, prompt):
                        image = Image.open(file)
                        params = look(image, prompt, gr)
                        return params, [file]

        with gr.Column(scale=2) as right_col:
            with gr.Tab(label=t("Setting")):
                preset_accordion = gr.Accordion(
                    label="Preset:",
                    open=False,
                )
                with preset_accordion:
                    with gr.Group(), gr.Column():
                        preset_image = gr.Image(
                            placeholder="Select or drop preset image here",
                            show_label=False,
                            type="filepath",
                            interactive=True,
                            sources=['upload'],
                        )
                        preset_gallery = gr.Gallery(
                            show_label=False,
                            height="auto",
                            allow_preview=False,
                            preview=False,
                            columns=[2],
                            rows=[3],
                            object_fit="contain",
                            visible=True,
                            buttons=['download', 'share', 'fullscreen'],
                            min_width=60,
                            selected_index=None,
                            value=path_manager.get_presets(),
                        )
                preset_selection = gr.Text(
                    value=None,
                    visible='hidden',
                )
                add_ctrl("preset_selection", preset_selection)

                performance_selection = gr.Dropdown(
                    label=t("Performance"),
                    choices=list(performance_settings.performance_options.keys())
                    + [performance_settings.CUSTOM_PERFORMANCE],
                    value=settings["performance"],
                )
                add_ctrl("performance_selection", performance_selection, True)
                perf_name = gr.Textbox(
                    show_label=False,
                    placeholder=t("Name"),
                    interactive=True,
                    visible='hidden',
                )
                perf_save = gr.Button(
                    value=t("Save"),
                    visible='hidden',
                )
                custom_default_values = performance_settings.get_perf_options(
                    settings["performance"]
                )
                custom_steps = gr.Slider(
                    label=t("Custom Steps"),
                    minimum=1,
                    maximum=200,
                    step=1,
                    value=custom_default_values["custom_steps"],
                    visible='hidden',
                )
                add_ctrl("custom_steps", custom_steps)

                cfg = gr.Slider(
                    label=t("CFG"),
                    minimum=0.0,
                    maximum=20.0,
                    step=0.1,
                    value=custom_default_values["cfg"],
                    visible='hidden',
                )
                add_ctrl("cfg", cfg)
                sampler_name = gr.Dropdown(
                    label=t("Sampler"),
                    choices=sorted(KSampler.SAMPLERS),
                    value=custom_default_values["sampler_name"],
                    visible='hidden',
                )
                add_ctrl("sampler_name", sampler_name)
                scheduler = gr.Dropdown(
                    label=t("Scheduler"),
                    choices=sorted(KSampler.SCHEDULERS),
                    value=custom_default_values["scheduler"],
                    visible='hidden',
                )
                add_ctrl("scheduler", scheduler)

                clip_skip = gr.Slider(
                    label=t("Clip Skip"),
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1,
                    visible='hidden',
                )

                add_ctrl("clip_skip", clip_skip)

                performance_outputs = [
                    perf_name,
                    perf_save,
                    cfg,
                    sampler_name,
                    scheduler,
                    clip_skip,
                    custom_steps,
                ]

                @perf_save.click(
                    api_visibility='undocumented',
                    inputs=performance_outputs,
                    outputs=[performance_selection],
                )
                def performance_save(
                    perf_name,
                    perf_save,
                    cfg,
                    sampler_name,
                    scheduler,
                    clip_skip,
                    custom_steps,
                ):
                    if perf_name != "":
                        perf_options = performance_settings.load_performance()
                        opts = {
                            "custom_steps": custom_steps,
                            "cfg": cfg,
                            "sampler_name": sampler_name,
                            "scheduler": scheduler,
                            "clip_skip": clip_skip,
                        }
                        perf_options[perf_name] = opts
                        performance_settings.save_performance(perf_options)
                        choices = list(perf_options.keys()) + [
                            performance_settings.CUSTOM_PERFORMANCE
                        ]
                        return gr.update(choices=choices, value=perf_name)
                    else:
                        return gr.update()

                with gr.Group():
                    aspect_ratios_selection = gr.Dropdown(
                        label=t("Aspect Ratios (width x height)"),
                        choices=list(resolution_settings.aspect_ratios.keys())
                        + [resolution_settings.CUSTOM_RESOLUTION],
                        value=settings["resolution"],
                    )
                    add_ctrl("aspect_ratios_selection", aspect_ratios_selection, True)
                    ratio_name = gr.Textbox(
                        show_label=False,
                        placeholder=t("Name"),
                        interactive=True,
                        visible='hidden',
                    )
                    default_resolution = resolution_settings.get_aspect_ratios(
                        settings["resolution"]
                    )
                    custom_width = gr.Slider(
                        label=t("Width"),
                        minimum=256,
                        maximum=4096,
                        step=2,
                        visible='hidden',
                        value=default_resolution[0],
                    )
                    add_ctrl("custom_width", custom_width)
                    custom_height = gr.Slider(
                        label=t("Height"),
                        minimum=256,
                        maximum=4096,
                        step=2,
                        visible='hidden',
                        value=default_resolution[1],
                    )
                    add_ctrl("custom_height", custom_height)
                    ratio_save = gr.Button(
                        value=t("Save"),
                        visible='hidden',
                    )

                    @ratio_save.click(
                        api_visibility='undocumented',
                        inputs=[ratio_name, custom_width, custom_height],
                        outputs=[aspect_ratios_selection],
                    )
                    def ratio_save_click(ratio_name, custom_width, custom_height):
                        if ratio_name != "":
                            ratio_options = resolution_settings.load_resolutions()
                            ratio_options[ratio_name] = (custom_width, custom_height)
                            resolution_settings.save_resolutions(ratio_options)
                            choices = list(resolution_settings.aspect_ratios.keys()) + [
                                resolution_settings.CUSTOM_RESOLUTION
                            ]
                            new_ratio_name = (
                                f"{custom_width}x{custom_height} ({ratio_name})"
                            )
                            return gr.update(choices=choices, value=new_ratio_name)
                        else:
                            return gr.update()

                    style_selection = gr.Dropdown(
                        label=t("Style Selection"),
                        multiselect=True,
                        container=True,
                        choices=list(load_styles().keys()),
                        value=list(
                            set(settings["style"]) &
                            set(load_styles().keys())
                        ),
                    )
                    add_ctrl("style_selection", style_selection)
                style_button = gr.Button(value="⬅️ " + t("Send Style to prompt"), size="sm")
                image_number = gr.Slider(
                    label=t("Image Number"),
                    minimum=0,
                    maximum=settings.get("image_number_max", 50),
                    step=1,
                    value=settings.get("image_number", 1),
                )
                add_ctrl("image_number", image_number, configurable=True)
                auto_negative_prompt = gr.Checkbox(
                    label=t("Auto Negative Prompt"),
                    show_label=True,
                    value=settings["auto_negative_prompt"],
                )
                add_ctrl("auto_negative", auto_negative_prompt)
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    show_label=True,
                    placeholder="Type prompt here.",
                    value=settings["negative_prompt"],
                )
                add_ctrl("negative", negative_prompt)
                seed_random = gr.Checkbox(
                    label=t("Random Seed"), value=settings["seed_random"]
                )
                image_seed = gr.Number(
                    label=t("Seed"),
                    value=settings["seed"],
                    precision=0,
                    visible='hidden' if settings["seed_random"] else True,
                )
                add_ctrl("seed", image_seed)

                @style_button.click(
                    api_visibility='undocumented',
                    inputs=[prompt, negative_prompt, style_selection],
                    outputs=[prompt, negative_prompt, style_selection],
                )
                def send_to_prompt(prompt_text, negative_text, style_inputs):
                    prompt_style, negative_style = apply_style(
                        style_inputs, prompt_text, negative_text, ""
                    )
                    return prompt_style, negative_style, []

                @seed_random.change(
                    api_visibility='undocumented',
                    inputs=[seed_random],
                    outputs=[image_seed]
                )
                def random_checked(r):
                    return gr.update(visible='hidden' if r else True) 

                def refresh_seed(r, s):
                    if r:
                        return -1
                    else:
                        return s

            model_tab = gr.Tab(
                label=t("Models"),
                visible=True,
            )
            with model_tab:
                with gr.Tab(label=t("Model")):
                    model_current = gr.HTML(
                        value=f"{settings['base_model']}",
                    )
                    with gr.Group():
                        modelfilter = gr.Textbox(
                            placeholder=t("Model name"),
                            value="",
                            show_label=False,
                            container=False,
                        )
                        model_gallery = gr.Gallery(
                            label=f"SDXL model: {settings['base_model']}",
                            show_label=False,
                            height="auto",
                            allow_preview=False,
                            preview=False,
                            columns=[2],
                            rows=[3],
                            object_fit="contain",
                            visible=True,
                            buttons=None,
                            min_width=60,
                            value=list(
                                map(
                                    lambda x: (get_checkpoint_thumbnail(x), x),
                                    shared.models.get_names("checkpoints"),
                                )
                            ),
                        )

                    base_model = gr.Text(
                        visible='hidden',
                        value=settings["base_model"],
                    )
                    add_ctrl("base_model_name", base_model)

                    @modelfilter.input(
                        api_visibility='undocumented',
                        inputs=modelfilter,
                        outputs=[model_gallery]
                    )
                    @modelfilter.submit(
                        api_visibility='undocumented',
                        inputs=modelfilter,
                        outputs=[model_gallery]
                    )
                    def update_model_filter(filtered):
                        filtered_filenames = filter(
                            lambda filename: filtered.lower() in filename.lower(),
                            shared.models.get_names("checkpoints"),
                        )
                        newlist = list(
                            map(
                                lambda x: (get_checkpoint_thumbnail(x), x),
                                filtered_filenames,
                            )
                        )
                        return gr.update(value=newlist)

                    def update_model_select(evt: gr.SelectData):
                        model_name = f"{evt.value['caption']}"
                        model = shared.models.get_models_by_path(
                            "checkpoints",
                            model_name
                        )
                        model_base = shared.models.get_model_base(model)

                        txt = f"{evt.value['caption']}<br>{t('Model type')}: {model_base}"

                        return {
                            model_current: gr.update(value=txt),
                            base_model: gr.update(value=model_name),
                        }

                    model_gallery.select(
                        fn=update_model_select,
                        api_visibility='undocumented',
                        outputs=[model_current, base_model]
                    )

                with gr.Tab(label="LoRAs"):
                    with gr.Group(visible=False) as lora_add:
                        lorafilter = gr.Textbox(
                            placeholder=t("Search LoRA"),
                            value="",
                            show_label=False,
                            container=False,
                        )
                        lora_gallery = gr.Gallery(
                            label=None,
                            show_label=False,
                            height="auto",
                            allow_preview=False,
                            preview=False,
                            interactive=False,
                            selected_index=None,
                            columns=[2],
                            rows=[3],
                            object_fit="contain",
                            buttons=None,
                            min_width=60,
                            value=list(
                                map(
                                    lambda x: (get_lora_thumbnail(x), x),
                                    shared.models.get_names("loras"),
                                )
                            ),
                        )
                        lora_cancel_btn = gr.Button(
                            value="cancel",
                            scale=1,
                        )

                    default_active = []
                    for i in range(1, 6):
                        m = settings.get(f"lora_{i}_model", "None")
                        w = settings.get(f"lora_{i}_weight", 0.0)
                        if m != "" and m != "None":
                            default_active.append((get_lora_thumbnail(m), f"{w} - {m}"))

                    with gr.Group(visible=True) as lora_active:
                        with gr.Row():
                            lora_weight_slider = gr.Slider(
                                label=t("Weight"),
                                show_label=True,
                                minimum=settings.get("lora_min", 0),
                                maximum=settings.get("lora_max", 2),
                                step=0.05,
                                value=1.0,
                                interactive=True,
                            )
                        lora_active_gallery = gr.Gallery(
                            label=None,
                            show_label=False,
                            height="auto",
                            allow_preview=False,
                            preview=False,
                            interactive=False,
                            columns=[2],
                            rows=[3],
                            object_fit="contain",
                            visible=True,
                            buttons=None,
                            min_width=60,
                            value=default_active,
                        )
                        add_ctrl("loras", lora_active_gallery)

                        with gr.Group(), gr.Row():
                            lora_add_btn = gr.Button(
                                value="+",
                                scale=1,
                            )

                            lora_del_btn = gr.Button(
                                value="-",
                                scale=1,
                            )

                        lora_keywords = gr.Textbox(
                            label=t("LoRA Trigger Words"), interactive=False
                        )
                        add_ctrl("lora_keywords", lora_keywords)

                def gallery_toggle():
                    result = [
                        gr.update(visible=True),
                        gr.update(visible=False),
                    ]
                    return result

                # LoRA
                @lorafilter.input(
                    api_visibility='undocumented',
                    inputs=[lorafilter, lora_active_gallery],
                    outputs=[lora_gallery]
                )
                @lorafilter.submit(
                    api_visibility='undocumented',
                    inputs=[lorafilter, lora_active_gallery],
                    outputs=[lora_gallery]
                )
                def update_lora_filter(lorafilter, lora_active_gallery):
                    if lora_active_gallery:
                        active = list(
                            map(lambda x: x[1].split(" - ", 1)[1], lora_active_gallery)
                        )
                    else:
                        active = []
                    filtered_filenames = [
                        x
                        for x in map(
                            lambda x: (get_lora_thumbnail(x), x),
                            filter(
                                lambda filename: lorafilter.lower() in filename.lower(),
                                shared.models.get_names("loras"),
                            ),
                        )
                        if x[1] not in active
                    ]
                    # Sorry for this. It is supposed to show all LoRAs matching the filter and is not currently used.

                    return gr.update(value=filtered_filenames)

                lora_active_selected = None

                def lora_select(gallery, lorafilter, evt: gr.SelectData):
                    global lora_active_selected

                    w = 1.0

                    keywords = ""
                    active = []
                    if gallery is not None:
                        for lora_data in gallery:
                            w, l = lora_data[1].split(" - ", 1)
                            keywords = f"{keywords}, {load_keywords(l)} "
                            active.append(l)
                    else:
                        gallery = []
                    keywords = f"{keywords}, {load_keywords(evt.value['caption'])} "
                    gallery.append(
                        (
                            get_lora_thumbnail(evt.value["caption"]),
                            f"{w} - {evt.value['caption']}",
                        )
                    )
                    lora_active_selected = len(gallery) - 1
                    active.append(f"{evt.value['caption']}")

                    return {
                        lora_add: gr.update(visible=False),
                        lora_gallery: gr.update(),
                        lora_active: gr.update(visible=True),
                        lora_active_gallery: gr.update(
                            value=gallery,
                            selected_index=lora_active_selected,
                        ),
                        lora_keywords: gr.update(value=keywords),
                    }

                def lora_active_select(gallery, evt: gr.SelectData):
                    global lora_active_selected
                    lora_active_selected = evt.index
                    return {
                        lora_active: gr.update(),
                        lora_active_gallery: gr.update(),
                        lora_weight_slider: gr.update(
                            value=float(evt.value["caption"].split(" - ", 1)[0])
                        ),
                    }

                def lora_delete(gallery, lorafilter):
                    global lora_active_selected
                    if gallery == None or len(gallery) == 0:
                        return {
                            lora_gallery: gr.update(),
                            lora_active_gallery: gr.update(),
                            lora_keywords: gr.update(),
                        }
                    if lora_active_selected is not None and lora_active_selected < len(
                        gallery
                    ):
                        del gallery[lora_active_selected]
                    if lora_active_selected is not None:
                        lora_active_selected = min(
                            lora_active_selected, len(gallery) - 1
                        )
                    else:
                        lora_active_selected = len(gallery) - 1
                    keywords = ""
                    active = []
                    active_names = []
                    for lora_data in gallery:
                        w, l = lora_data[1].split(" - ", 1)
                        active.append(lora_data)
                        active_names.append(l)
                        keywords = f"{keywords}, {load_keywords(l)} "

                    loras = list(
                        filter(
                            lambda filename: lorafilter.lower() in filename.lower(),
                            shared.models.get_names("loras"),
                        )
                    )
                    if len(loras) == 0:
                        loras = shared.models.get_names("loras")

                    inactive = [
                        x
                        for x in map(lambda x: (get_lora_thumbnail(x), x), loras)
                        if x[1] not in active_names
                    ]
                    return {
                        lora_gallery: gr.update(
                            value=inactive,
                            selected_index=None,
                        ),
                        lora_active_gallery: gr.update(
                            value=active,
                            selected_index=lora_active_selected,
                        ),
                        lora_keywords: gr.update(value=keywords),
                    }

                def lora_weight_slider_update(gallery, w):
                    global lora_active_selected
                    if lora_active_selected is None:
                        return {lora_active_gallery: gr.update()}

                    loras = []
                    for lora_data in gallery:
                        loras.append((lora_data[0], lora_data[1]))
                    l = gallery[lora_active_selected][1].split(" - ")[1]
                    loras[lora_active_selected] = (get_lora_thumbnail(l), f"{w} - {l}")

                    return {
                        lora_active_gallery: gr.update(value=loras),
                    }

                lora_weight_slider.release(
                    fn=lora_weight_slider_update,
                    api_visibility='undocumented',
                    inputs=[lora_active_gallery, lora_weight_slider],
                    outputs=[lora_active_gallery],
                )
                lora_add_btn.click(
                    fn=gallery_toggle,
                    api_visibility='undocumented',
                    outputs=[lora_add, lora_active],
                )
                lora_cancel_btn.click(
                    fn=gallery_toggle,
                    api_visibility='undocumented',
                    outputs=[lora_active, lora_add],
                )
                lora_del_btn.click(
                    fn=lora_delete,
                    api_visibility='undocumented',
                    inputs=[lora_active_gallery, lorafilter],
                    outputs=[lora_gallery, lora_active_gallery, lora_keywords],
                )
                lora_gallery.select(
                    fn=lora_select,
                    api_visibility='undocumented',
                    inputs=[lora_active_gallery, lorafilter],
                    outputs=[
                        lora_add,
                        lora_gallery,
                        lora_active,
                        lora_active_gallery,
                        lora_keywords,
                    ],
                )
                lora_active_gallery.select(
                    fn=lora_active_select,
                    api_visibility='undocumented',
                    inputs=[lora_active_gallery],
                    outputs=[lora_active, lora_active_gallery, lora_weight_slider],
                )

                with gr.Row():
                    model_refresh = gr.Button(
                        value=f"\U0001f504 {t('Refresh All Files')}",
                        variant="secondary",
                        elem_classes="refresh_button",
                    )

            @model_refresh.click(
                api_visibility='undocumented',
                inputs=[lora_active_gallery],
                outputs=[
                    modelfilter,
                    model_gallery,
                    lorafilter,
                    lora_gallery,
                    style_selection,
                    preset_gallery,
                ],
            )
            def model_refresh_clicked(lora_active_gallery):
                global civit_checkpoints, civit_loras, lora_active_selected
                lora_active_selected=None
                shared.models.update_all_models()

                results = {
                    modelfilter: gr.update(value=""),
                    model_gallery: update_model_filter(""),
                    lorafilter: gr.update(value=""),
                    lora_gallery: update_lora_filter("", lora_active_gallery),
                    style_selection: gr.update(choices=list(load_styles().keys())),
                    preset_gallery: gr.update(value=path_manager.get_presets()),
                }

                return results

            ui_onebutton.ui_onebutton(prompt, run_event)

            inpaint_toggle = ui_controlnet.add_controlnet_tab(
                main_view, inpaint_view, prompt, image_number, run_event
            )

            with gr.Tab(label=t("Info")):
                with gr.Row():
                    metadata_json.render()
                with gr.Row():
                    gr.HTML(
                        value="""
                        <a href="https://discord.gg/CvpAFya9Rr"><img src="gradio_api/file=html/icon_clyde_white_RGB.svg" height="16" width="16" style="display:inline-block;">&nbsp;Discord</a><br>
                        <a href="https://github.com/runew0lf/RuinedFooocus"><img src="gradio_api/file=html/github-mark-white.svg" height="16" width="16" style="display:inline-block;">&nbsp;Github</a><br>
                        <a href="gradio_api/file/html/last_image.html" style="color: gray; text-decoration: none" target="_blank">&pi;</a>
                        """,
                    )

            try:
                with open('update_log.md', 'r', encoding='utf-8') as update_log:
                    hints = update_log.read()
            except:
                hints = ''
            hint_text = gr.Markdown(
                value=hints,
                elem_id="hint-container",
                elem_classes="hint-container",
            )

            @performance_selection.change(
                api_visibility='undocumented',
                inputs=[performance_selection],
                outputs=[perf_name] + performance_outputs,
            )
            def performance_changed(selection):
                if selection == performance_settings.CUSTOM_PERFORMANCE:
                    return [gr.update(value="")] + [gr.update(visible=True)] * len(
                        performance_outputs
                    )
                else:
                    return [gr.update(visible='hidden')] + [
                        gr.update(visible='hidden')
                    ] * len(performance_outputs)

            @performance_selection.change(
                api_visibility='undocumented',
                inputs=[performance_selection],
                outputs=[custom_steps]
                + [cfg]
                + [sampler_name]
                + [scheduler]
                + [clip_skip],
            )
            def performance_changed_update_custom(selection):
                # Skip if Custom was selected
                if selection == performance_settings.CUSTOM_PERFORMANCE:
                    return [gr.update()] * 5

                # Update Custom values based on selected Performance mode
                selected_perf_options = performance_settings.get_perf_options(selection)
                return {
                    custom_steps: gr.update(
                        value=selected_perf_options["custom_steps"]
                    ),
                    cfg: gr.update(value=selected_perf_options["cfg"]),
                    sampler_name: gr.update(
                        value=selected_perf_options["sampler_name"]
                    ),
                    scheduler: gr.update(value=selected_perf_options["scheduler"]),
                    clip_skip: gr.update(value=selected_perf_options["clip_skip"]),
                }

            @aspect_ratios_selection.change(
                api_visibility='undocumented',
                inputs=[aspect_ratios_selection],
                outputs=[ratio_name, custom_width, custom_height, ratio_save],
            )
            def aspect_ratios_changed(selection):
                # Show resolution controls when selecting Custom
                if selection == resolution_settings.CUSTOM_RESOLUTION:
                    return [gr.update(visible=True)] * 4

                # Hide resolution controls and update with selected resolution
                selected_width, selected_height = resolution_settings.get_aspect_ratios(
                    selection
                )
                return {
                    ratio_name: gr.update(visible='hidden'),
                    custom_width: gr.update(visible='hidden', value=selected_width),
                    custom_height: gr.update(visible='hidden', value=selected_height),
                    ratio_save: gr.update(visible='hidden'),
                }

        def activate(): # UGLY workaround for https://github.com/gradio-app/gradio/issues/7586
            return gr.update(interactive=True)

        run_event.change(
            fn=refresh_seed,
            api_visibility='undocumented',
            inputs=[seed_random, image_seed],
            outputs=image_seed
        ).then(
            fn=activate,
            api_visibility='undocumented',
            outputs=[inpaint_view]
        ).then(
            fn=generate_clicked,
            api_visibility='undocumented',
            inputs=state["ctrls_obj"],
            outputs=[
                run_button,
                stop_button,
                progress_html,
                main_view,
                inpaint_view,
                inpaint_toggle,
                gallery,
                metadata_json,
                hint_text,
            ],
        )

        def poke(number):
            return number + 1

        run_button.click(fn=poke, api_visibility='undocumented', inputs=run_event, outputs=run_event)

        def stop_clicked():
            worker.interrupt_ruined_processing = True
            shared.state["interrupted"] = False

        stop_button.click(fn=stop_clicked, api_visibility='undocumented', queue=False)

        def update_cfg():
            # Update ui components
            # Only refresh things like minimum, maximum and choices. Assume the user already
            # have options selected and don't overwrite them. (They should restart if they want that)
            return {
                image_number: gr.update(maximum=settings.get("image_number_max", 50)),
                performance_selection: gr.update(
                    choices=list(performance_settings.performance_options.keys()) + [performance_settings.CUSTOM_PERFORMANCE]
                ),
                aspect_ratios_selection: gr.update(
                    choices=list(resolution_settings.aspect_ratios.keys())
                    + [resolution_settings.CUSTOM_RESOLUTION]
                ),
                cfg_timestamp: gr.update(value=shared.state["last_config"]),
            }
        # If cfg_timestamp has a new value, trigger an update
        cfg_timestamp.change(fn=update_cfg, api_visibility='undocumented', outputs=[cfg_timestamp] + state["cfg_items_obj"])

        # Preset functions
        def preset_select(preset_gallery, evt: gr.SelectData):
            path = evt.value['image']['path']
            preset = Path(path).with_suffix('').name
            return {
                preset_image: gr.update(value=path),
                preset_selection: gr.update(value=path),
                preset_accordion: gr.update(label=t("Preset:") + " " + preset),

                performance_selection: gr.update(visible='hidden'),
                perf_name: gr.update(visible='hidden'),
                perf_save: gr.update(visible='hidden'),
                cfg: gr.update(visible='hidden'),
                sampler_name: gr.update(visible='hidden'),
                scheduler: gr.update(visible='hidden'),
                clip_skip: gr.update(visible='hidden'),
                custom_steps: gr.update(visible='hidden'),
                aspect_ratios_selection: gr.update(visible='hidden'),
                ratio_name: gr.update(visible='hidden'),
                custom_width: gr.update(visible='hidden'),
                custom_height: gr.update(visible='hidden'),
                ratio_save: gr.update(visible='hidden'),
                model_tab: gr.update(visible=False),
            }

        def preset_unselect(performance_selection_val, aspect_ratios_selection_val):
            show_perf = True if (performance_selection_val == performance_settings.CUSTOM_PERFORMANCE) else 'hidden'
            show_size = True if (aspect_ratios_selection_val == resolution_settings.CUSTOM_RESOLUTION) else 'hidden'

            return {
                preset_selection: gr.update(value=''),
                preset_accordion: gr.update(label=t("Preset:")),

                performance_selection: gr.update(visible=True),
                perf_name: gr.update(visible=show_perf),
                perf_save: gr.update(visible=show_perf),
                cfg: gr.update(visible=show_perf),
                sampler_name: gr.update(visible=show_perf),
                scheduler: gr.update(visible=show_perf),
                clip_skip: gr.update(visible=show_perf),
                custom_steps: gr.update(visible=show_perf),

                aspect_ratios_selection: gr.update(visible=True),
                ratio_name: gr.update(visible=show_size),
                custom_width: gr.update(visible=show_size),
                custom_height: gr.update(visible=show_size),
                ratio_save: gr.update(visible=show_size),

                model_tab: gr.update(visible=True),
            }
        def preset_image_upload(preset_image):
            path = preset_image
            preset = Path(path).with_suffix('').name
            return {
                preset_selection: gr.update(value=path),
                preset_accordion: gr.update(label=t("Preset:" + " " + preset)),

                performance_selection: gr.update(visible='hidden'),
                perf_name: gr.update(visible='hidden'),
                perf_save: gr.update(visible='hidden'),
                cfg: gr.update(visible='hidden'),
                sampler_name: gr.update(visible='hidden'),
                scheduler: gr.update(visible='hidden'),
                clip_skip: gr.update(visible='hidden'),
                custom_steps: gr.update(visible='hidden'),
                aspect_ratios_selection: gr.update(visible='hidden'),
                ratio_name: gr.update(visible='hidden'),
                custom_width: gr.update(visible='hidden'),
                custom_height: gr.update(visible='hidden'),
                ratio_save: gr.update(visible='hidden'),
                model_tab: gr.update(visible=False),
            }

        preset_image.clear(
            fn=preset_unselect,
            api_visibility='undocumented',
            inputs=[
                performance_selection,
                aspect_ratios_selection,
            ],
            outputs=[
                preset_selection,
                preset_accordion,

                performance_selection,
                perf_name,
                perf_save,
                cfg,
                sampler_name,
                scheduler,
                clip_skip,
                custom_steps,
                aspect_ratios_selection,
                ratio_name,
                custom_width,
                custom_height,
                ratio_save,
                model_tab,
            ]
        )
        preset_image.upload(
            fn=preset_image_upload,
            api_visibility='undocumented',
            inputs=[
                preset_image,
            ],
            outputs=[
                preset_selection,
                preset_accordion,

                performance_selection,
                perf_name,
                perf_save,
                cfg,
                sampler_name,
                scheduler,
                clip_skip,
                custom_steps,
                aspect_ratios_selection,
                ratio_name,
                custom_width,
                custom_height,
                ratio_save,
                model_tab,
            ]
        )
        preset_gallery.select(
            fn=preset_select,
            api_visibility='undocumented',
            inputs=[preset_gallery],
            outputs=[
                preset_image,
                preset_selection,
                preset_accordion,

                performance_selection,
                perf_name,
                perf_save,
                cfg,
                sampler_name,
                scheduler,
                clip_skip,
                custom_steps,
                aspect_ratios_selection,
                ratio_name,
                custom_width,
                custom_height,
                ratio_save,
                model_tab,
            ]
        )

    add_api()

if isinstance(args.auth, str) and not "/" in args.auth:
    if len(args.auth):
        print(
            f'\nERROR! --auth need be in the form of "username/password" not "{args.auth}"\n'
        )
    if args.share:
        print(
            f"\nWARNING! Will not enable --share without proper --auth=username/password\n"
        )
        args.share = False
launch_app(args)

# Wait...
while True:
    time.sleep(100)
