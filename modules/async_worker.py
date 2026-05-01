import threading
import gc
import torch
import math
import time
import pathlib
from pathlib import Path
import traceback

import json
import os
import shared
import random

from modules.prompt_processing import process_metadata, process_prompt, parse_loras
from modules.shift_attention import shift_attention

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from modules.util import generate_temp_filename, TimeIt, get_checkpoint_hashes, get_lora_hashes
import modules.pipelines
from shared import settings, path_manager

buffer = []
outputs = []
current_task = 0

interrupt_ruined_processing = False

def is_sha256_hash(input_string):
    # Check if the string is exactly 64 characters long
    if len(input_string) != 64:
        return False
    # Check if the string consists only of hexadecimal characters
    if not all(c in '0123456789abcdefABCDEF' for c in input_string):
        return False
    return True

def _process(gen_data):
    res = []
    metadatastrings = []
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

    for lora_data in gen_data["loras"] if gen_data["loras"] is not None else []:
        if is_sha256_hash(lora_data[0]):
            hash = lora_data[0]
        else:
            hash = None

        w, l  = lora_data[1].split(" - ", 1)
        if not l == "None":
            loras.append({"name": l, "weight": float(w), "hash": hash})

    if "silent" not in gen_data:
        outputs.append(
            [
                gen_data["task_id"],
                "preview",
                (-1, f"Loading base model: {gen_data['base_model_name']}", None),
            ]
        )

    # Check filename/hash for base_model
    filename = shared.models.get_model_path(
        "checkpoints",
        name=gen_data["base_model_name"],
        hash=gen_data.get("base_model_hash", None),
        default=None,
    )
    # If filename is None: trigger model-refresh and return error
    if filename is None:
        print(f"ERROR: Could not load the checkpoint:")
        return ["html/error.png"]
    for folder in path_manager.paths.get('path_checkpoints', []): # Get a pretty name for the model if we had to use the hash to find it
        if filename.startswith(folder):
            gen_data['base_model_name'] = filename.removeprefix(folder).strip("\\/")
            continue

    # FIXME: is this even used???
    gen_data["modelhash"] = pipeline.load_base_model(
        gen_data["base_model_name"],
        hash=gen_data.get("base_model_hash", None),
    )


    if "silent" not in gen_data:
        outputs.append([gen_data["task_id"], "preview", (-1, f"Loading LoRA models ...", None)])

    # FIXME move this into get_perf_options?
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
    gen_data["steps"] = steps

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
    else:
        gen_data["width"] = width
    if "height" in gen_data:
        height = gen_data["height"]
    else:
        gen_data["height"] = height

    # TODO: Leave this logic to the pipeline?
    if gen_data["cn_selection"] == "Img2Img" or gen_data["cn_type"] == "Img2img":
        if gen_data["input_image"]:
            width = gen_data["input_image"].width
            height = gen_data["input_image"].height
        else:
            print(f"WARNING: CheatCode selected but no Input image selected. Ignoring PowerUp!")
            gen_data["cn_selection"] = "None"
            gen_data["cn_type"] = "None"

    try:
        seed = int(gen_data["seed"])
    except:
        seed = -1
    max_seed = 2**32
    if not isinstance(seed, int) or seed == -1:
        seed = random.randint(0, max_seed)

    all_steps = steps * max(image_number, 1)
    with open('render.txt', 'r', encoding='utf-8') as f:
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

        # Get status based on time (for very slow generations)
        status = lines[int(time.time() // 10) % len(lines)]

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
            elif isinstance(y, str):
                image = Image.open(y)
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
            preview = shared.path_manager.model_paths["temp_preview_path"]
        else:
            preview = None

        shared.state["preview_grid"].save(
            shared.path_manager.model_paths["temp_preview_path"],
            optimize=True,
            quality=35 if step < total_steps else 70,
        )

        if gen_data.get("task_type", None) != "tool_call":
            outputs.append(
                [
                    gen_data["task_id"],
                    "preview",
                    (
                        int(
                            100
                            * (gen_data["index"][0] + done_steps / all_steps)
                            / max(gen_data["index"][1], 1)
                        ),
                        f"{status} - {step}/{total_steps}",
                        preview,
                    ),
                ]
            )

    # TODO: this should be an "inital ok gen_data" at the beginning of the function
    if "input_image" not in gen_data:
        gen_data["input_image"] = None
    if "main_view" not in gen_data:
        gen_data["main_view"] = None

    stop_batch = False
    for i in range(max(image_number, 1)):
        p_txt, n_txt = process_prompt(
            gen_data["style_selection"], gen_data["prompt"], gen_data["negative"], gen_data
        )

        distance = float(i) / max(image_number - 1.0, 1.0) # Use max() to avoid div. by 0
        p_txt = shift_attention(p_txt, distance)
        n_txt = shift_attention(n_txt, distance)

        gen_data["seed"] = abs(seed) # Update seed
        start_step = 0
        denoise = None
        with TimeIt("Pipeline process"):
            try:
                # Load LoRAs
                parsed_loras, p_txt, n_txt = parse_loras(p_txt, n_txt)
                all_loras = loras + parsed_loras
                used_loras = []

                # Get and update all lora paths/hashes
                for lora in all_loras:
                    name = lora.get("name", None)
                    weight = lora.get("weight", 0)
                    hash = lora.get("hash", None)
                    if name is None or weight == 0:
                        if filename is None:
                            print(f"ERROR: Could not load LoRA: (Skipping)")
                        continue

                    filename = shared.models.get_model_path("loras", name=name, hash=hash)
                    for folder in path_manager.paths.get('path_loras', []): # Get a pretty name for the model if we had to use the hash to find it
                        if filename.startswith(folder):
                            used_loras.append(
                                    {
                                        "name": filename.removeprefix(folder).strip("\\/"),
                                        "weight": weight,
                                        "hash": hash,
                                    }
                                )
                            continue

                pipeline.load_loras(used_loras)
                gen_data["positive_prompt"] = p_txt
                gen_data["negative_prompt"] = n_txt

                imgs = pipeline.process(
                    gen_data=gen_data,
                    callback=callback if "silent" not in gen_data else None,
                )
            except InterruptProcessingException as iex:
                stop_batch = True
                imgs = []
            except Exception as iex:
                stop_batch = True
                imgs = []
                traceback.print_exc()
                print(f"ERROR: {iex}")

        for x in imgs:
            folder=shared.path_manager.model_paths["temp_outputs_path"]
            local_temp_filename = generate_temp_filename(
                folder=folder,
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
                "seed": abs(seed),
                "sampler_name": gen_data["sampler_name"],
                "scheduler": gen_data["scheduler"],
                "base_model_name": gen_data["base_model_name"],
                "base_model_hash": get_checkpoint_hashes(gen_data["base_model_name"])['SHA256'],
                "loras": [[f"{get_lora_hashes(lora['name'])['SHA256']}", f"{lora['weight']} - {lora['name']}"] for lora in used_loras],
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

            if "preview_count" not in shared.state:
                shared.state["preview_count"] = 0
            shared.state["preview_count"] += 1
            if isinstance(x, str) or isinstance(
                x, (pathlib.WindowsPath, pathlib.PosixPath)
            ):
                local_temp_filename = x
            else:
                if not isinstance(x, Image.Image):
                    x = Image.fromarray(x)
                x.save(local_temp_filename, pnginfo=metadata)

            try:
                metadata = {
                    "parameters": json.dumps(prompt),
                    "file_path": str(Path(local_temp_filename).relative_to(folder))
                }
                if "browser" in shared.shared_cache:
                    shared.shared_cache["browser"].add_image(
                        local_temp_filename,
                        Path(local_temp_filename).relative_to(folder),
                        metadata,
                        commit=True
                    )
            except:
                pass

            res.append(local_temp_filename)
            metadatastrings.append(json.dumps(prompt))
            shared.state["last_image"] = local_temp_filename

        if seed > -1:
            seed += 1
        if stop_batch:
            break
    return res

def worker():
    global buffer, outputs

    pipeline = modules.pipelines.update(
        {"base_model_name": settings.default_settings.get("base_model")}
    )
    if not pipeline == None:
        pipeline.load_base_model(settings.default_settings.get("base_model"))

    def job_start(gen_data):
        shared.state["preview_grid"] = None
        shared.state["preview_total"] = max(gen_data["image_total"], 1)
        shared.state["preview_count"] = 0

    def job_stop():
        shared.state["preview_grid"] = None
        shared.state["preview_total"] = 0
        shared.state["preview_count"] = 0

    def reset_preview():
        shared.state["preview_grid"] = None
        shared.state["preview_count"] = 0

    def process(gen_data):
        # Check some needed items
        if not "image_total" in gen_data:
            gen_data["image_total"] = 1
        if not "generate_forever" in gen_data:
            gen_data["generate_forever"] = False

        shared.state["preview_total"] = max(gen_data["image_total"], 1)

        results = []
        metadatastrings = []
        while True:
            reset_preview()
            gen_data["index"] = (0, (gen_data["image_total"]))
            if isinstance(gen_data["prompt"], list):
                tmp_data = gen_data.copy()
                for prompt in gen_data["prompt"]:
                    tmp_data["prompt"] = prompt
                    if gen_data["generate_forever"]:
                        reset_preview()
                    results.extend(_process(tmp_data))
                    if shared.state["interrupted"]:
                        break
                    tmp_data["index"] = (tmp_data["index"][0] + 1, tmp_data["index"][1])
            else:
                gen_data["index"] = (0, 1)
                results.extend(_process(gen_data))

            if not (gen_data["generate_forever"] and shared.state["interrupted"] == False):
                break

        # Prepend preview-grid (maybe)
        if (
            "preview_grid" in shared.state and 
            shared.state["preview_grid"] is not None
            and shared.state["preview_total"] > 1
            and ("show_preview" not in gen_data or gen_data["show_preview"] == True)
            and not gen_data["generate_forever"]
        ):
            results = [
                shared.path_manager.model_paths["temp_preview_path"]
            ] + results

        outputs.append([gen_data["task_id"], "results", results])


    def txt2txt_process(gen_data):

        pipeline = modules.pipelines.update(gen_data)
        if pipeline == None:
            print(f"ERROR: No pipeline")
            return

        try:
            # See if pipeline wants to pre-parse gen_data
            gen_data = pipeline.parse_gen_data(gen_data)
        except:
            pass

        results = pipeline.process(gen_data)

        outputs.append([gen_data["task_id"], "results", results])


    def handler(gen_data):
        match gen_data["task_type"]:
            case "process":
                process(gen_data)
            case "api_process":
                gen_data["silent"] = True
                process(gen_data)
            case "llama":
                txt2txt_process(gen_data)
            case _:
                print(f"WARN: Unknown task_type: {gen_data['task_type']}")

    while True:
        time.sleep(0.01)
        if len(buffer) > 0:
            task = buffer.pop(0)
            handler(task)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

# Use this to add a task, then use task_result() to get data from the pipeline
def add_task(gen_data):
    global current_task, buffer

    current_task += 1
    task_id = current_task 
    gen_data["task_id"] = task_id
    buffer.append(gen_data.copy())
    return task_id

# Pipelines use this to add results
def add_result(task_id, flag, product):
    global outputs

    outputs.append([task_id, flag, product])

# Use the task_id from add_task() to wait for data
def task_result(task_id):
    global outputs

    while True:
        if not outputs:
            time.sleep(0.1)
            continue

        matches = [res for res in outputs if res[0] == task_id] 

        if matches:
            id, flag, product = matches.pop(0)
            outputs.remove([id, flag, product])
            break

    return (flag, product)


threading.Thread(target=worker, daemon=True).start()
