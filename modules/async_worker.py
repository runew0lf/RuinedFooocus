import threading
import gc
import torch
import math
from playsound import playsound
import modules.controlnet
import pathlib
from pathlib import Path

buffer = []
outputs = []
results = []
metadatastrings = []

interrupt_ruined_processing = False


def worker():
    global buffer, outputs

    import json
    import os
    import time
    import shared
    import random

    from modules.prompt_processing import process_metadata, process_prompt, parse_loras

    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from modules.util import generate_temp_filename, TimeIt, model_hash
    import modules.pipelines
    from modules.settings import default_settings

    pipeline = modules.pipelines.update(
        {"base_model_name": default_settings["base_model"]}
    )
    if not pipeline == None:
        pipeline.load_base_model(default_settings["base_model"])

    try:
        async_gradio_app = shared.gradio_root
        flag = f"""App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}"""
        if async_gradio_app.share:
            flag += f""" or {async_gradio_app.share_url}"""
        print(flag)
    except Exception as e:
        print(e)

    def handler(gen_data):
        match gen_data["task_type"]:
            case "start":
                job_start(gen_data)
            case "stop":
                job_stop()
            case "process":
                process(gen_data)
            case _:
                print(f"WARN: Unknown task_type: {gen_data['task_type']}")

    def job_start(gen_data):
        shared.state["preview_grid"] = None
        shared.state["preview_total"] = max(gen_data["image_total"], 1)
        shared.state["preview_count"] = 0

    def job_stop():
        shared.state["preview_grid"] = None
        shared.state["preview_total"] = 0
        shared.state["preview_count"] = 0

    def process(gen_data):
        global results, metadatastrings

        gen_data = process_metadata(gen_data)

        pipeline = modules.pipelines.update(gen_data)
        if pipeline == None:
            print(f"ERROR: No pipeline")
            return

        try:
            # See if pipeline wants to pre-parse gen_data
            gen_data = pipeline.parse_gen_data(gen_data)
        except:
            pass

        image_number = gen_data["image_number"]

        loras = []

        for lora_data in gen_data["loras"]:
            w, l  = lora_data[1].split(" - ", 1)
            loras.append((l, float(w)))

        parsed_loras, pos_stripped, neg_stripped = parse_loras(
            gen_data["prompt"], gen_data["negative"]
        )
        loras.extend(parsed_loras)

        outputs.append(
            [
                "preview",
                (-1, f"Loading base model: {gen_data['base_model_name']}", None),
            ]
        )
        gen_data["modelhash"] = pipeline.load_base_model(gen_data["base_model_name"])
        outputs.append(["preview", (-1, f"Loading LoRA models ...", None)])
        pipeline.load_loras(loras)

        if (
            gen_data["performance_selection"]
            == shared.performance_settings.CUSTOM_PERFORMANCE
        ):
            steps = gen_data["custom_steps"]
        else:
            perf_options = shared.performance_settings.get_perf_options(
                gen_data["performance_selection"]
            ).copy()
            perf_options.update(gen_data)
            gen_data = perf_options

        steps = gen_data["custom_steps"]

        if (
            gen_data["aspect_ratios_selection"]
            == shared.resolution_settings.CUSTOM_RESOLUTION
        ):
            width, height = (gen_data["custom_width"], gen_data["custom_height"])
        else:
            width, height = shared.resolution_settings.aspect_ratios[
                gen_data["aspect_ratios_selection"]
            ]

        if "width" in gen_data:
            width = gen_data["width"]
        if "height" in gen_data:
            height = gen_data["height"]

        if gen_data["cn_selection"] == "Img2Img" or gen_data["cn_type"] == "Img2img":
            if gen_data["input_image"]:
                width = gen_data["input_image"].width
                height = gen_data["input_image"].height
            else:
                print(f"WARNING: CheatCode selected but no Input image selected. Ignoring PowerUp!")
                gen_data["cn_selection"] = "None"
                gen_data["cn_type"] = "None"

        seed = gen_data["seed"]

        max_seed = 2**32
        if not isinstance(seed, int) or seed < 0:
            seed = random.randint(0, max_seed)
        seed = seed % max_seed

        all_steps = steps * max(image_number, 1)
        with open("render.txt") as f:
            lines = f.readlines()
        status = random.choice(lines)
        status = f"{status}"

        class InterruptProcessingException(Exception):
            pass

        def callback(step, x0, x, total_steps, y):
            global status, interrupt_ruined_processing

            if interrupt_ruined_processing:
                shared.state["interrupted"] = True
                interrupt_ruined_processing = False
                raise InterruptProcessingException()

            # If we only generate 1 image, skip the last preview
            if (
                (not gen_data["generate_forever"])
                and shared.state["preview_total"] == 1
                and steps == step
            ):
                return

            done_steps = i * steps + step
            try:
                status
            except NameError:
                status = None
            if step % 10 == 0 or status == None:
                status = random.choice(lines)

            grid_xsize = math.ceil(math.sqrt(shared.state["preview_total"]))
            grid_ysize = math.ceil(shared.state["preview_total"] / grid_xsize)
            grid_max = max(grid_xsize, grid_ysize)
            pwidth = int(width * grid_xsize / grid_max)
            pheight = int(height * grid_ysize / grid_max)
            if shared.state["preview_grid"] is None:
                shared.state["preview_grid"] = Image.new("RGB", (pwidth, pheight))
            if y is not None:
                if isinstance(y, Image.Image):
                    image = y
                else:
                    image = Image.fromarray(y)
                grid_xpos = int(
                    (shared.state["preview_count"] % grid_xsize) * (pwidth / grid_xsize)
                )
                grid_ypos = int(
                    math.floor(shared.state["preview_count"] / grid_xsize)
                    * (pheight / grid_ysize)
                )
                image = image.resize((int(width / grid_max), int(height / grid_max)))
                shared.state["preview_grid"].paste(image, (grid_xpos, grid_ypos))

            shared.state["preview_grid"].save(
                shared.path_manager.model_paths["temp_preview_path"],
                optimize=True,
                quality=35 if step < total_steps else 70,
            )

            outputs.append(
                [
                    "preview",
                    (
                        int(
                            100
                            * (gen_data["index"][0] + done_steps / all_steps)
                            / gen_data["index"][1]
                        ),
                        f"{status} - {step}/{total_steps}",
                        shared.path_manager.model_paths["temp_preview_path"],
                    ),
                ]
            )

        stop_batch = False
        for i in range(max(image_number, 1)):
            p_txt, n_txt = process_prompt(
                gen_data["style_selection"], pos_stripped, neg_stripped, gen_data
            )
            start_step = 0
            denoise = None
            with TimeIt("Pipeline process"):
                try:
                    imgs = pipeline.process(
                        p_txt,
                        n_txt,
                        gen_data["input_image"],
                        modules.controlnet.get_settings(gen_data),
                        gen_data["main_view"],
                        steps,
                        width,
                        height,
                        seed,
                        start_step,
                        denoise,
                        gen_data["cfg"],
                        gen_data["sampler_name"],
                        gen_data["scheduler"],
                        gen_data["clip_skip"],
                        callback=callback,
                        gen_data=gen_data,
                    )
                except InterruptProcessingException as iex:
                    stop_batch = True
                    imgs = []

            for x in imgs:
                local_temp_filename = generate_temp_filename(
                    folder=shared.path_manager.model_paths["temp_outputs_path"],
                    extension="png",
                )
                dir_path = Path(local_temp_filename).parent
                dir_path.mkdir(parents=True, exist_ok=True)
                metadata = None
                prompt = {
                    "Prompt": p_txt,
                    "Negative": n_txt,
                    "steps": steps,
                    "cfg": gen_data["cfg"],
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "sampler_name": gen_data["sampler_name"],
                    "scheduler": gen_data["scheduler"],
                    "base_model_name": gen_data["base_model_name"],
                    "base_model_hash": model_hash(
                        Path(shared.path_manager.model_paths["modelfile_path"])
                        / gen_data["base_model_name"]
                    ),
                    "loras": [['', f"{lora[1]} - {lora[0]}"] for lora in loras],
                    "start_step": start_step,
                    "denoise": denoise,
                    "clip_skip": gen_data["clip_skip"],
                    "software": "RuinedFooocus",
                }
                metadata = PngInfo()
                # if True:
                #     def handle_whitespace(string: str):
                #         return (
                #             string.strip()
                #             .replace("\n", " ")
                #             .replace("\r", " ")
                #             .replace("\t", " ")
                #         )

                #     comment = f"{handle_whitespace(p_txt)}\nNegative prompt: {handle_whitespace(n_txt)}\nSteps: {round(steps, 1)}, Sampler: {gen_data['sampler_name']} {gen_data['scheduler']}, CFG Scale: {float(gen_data['cfg'])}, Seed: {seed}, Size: {width}x{height}, Model hash: {model_hash(Path(shared.path_manager.model_paths['modelfile_path']) / gen_data['base_model_name'])}, Model: {gen_data['base_model_name']}, Version: RuinedFooocus"
                #     metadata.add_text("parameters", comment)
                # else:
                metadata.add_text("parameters", json.dumps(prompt))

                shared.state["preview_count"] += 1
                if isinstance(x, str) or isinstance(
                    x, (pathlib.WindowsPath, pathlib.PosixPath)
                ):
                    local_temp_filename = x
                else:
                    if not isinstance(x, Image.Image):
                        x = Image.fromarray(x)
                    x.save(local_temp_filename, pnginfo=metadata)
                results.append(local_temp_filename)
                metadatastrings.append(json.dumps(prompt))
                shared.state["last_image"] = local_temp_filename

            seed += 1
            if stop_batch:
                break

        if len(buffer) == 0:
            if (
                shared.state["preview_grid"] is not None
                and shared.state["preview_total"] > 1
                and ("show_preview" not in gen_data or gen_data["show_preview"] == True)
            ):
                results = [
                    shared.path_manager.model_paths["temp_preview_path"]
                ] + results
            outputs.append(["results", results])
            results = []
            metadatastrings = []
        return

    while True:
        time.sleep(0.1)
        if len(buffer) > 0:
            task = buffer.pop(0)
            handler(task)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if Path("notification.mp3").exists():
                playsound("notification.mp3")


threading.Thread(target=worker, daemon=True).start()
