import os
import modules.async_worker as worker
from PIL import Image
import shared
import json


from comfy.samplers import KSampler


class pipeline:
    pipeline_type = ["hashbang"]

    model_hash = ""

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        # We don't need a model
        self.model_hash = name
        return

    def load_keywords(self, lora):
        return ""

    def load_loras(self, loras):
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Working ...", "html/generate_video.jpeg")
        )

        image = Image.open("html/logo.png")

        # Get command
        lines = gen_data["prompt"].splitlines()
        cmd = lines[0][2:].strip()
        data = "\n".join(lines[1:])

        match cmd:
            case "echo":
                # Simple test
                print(data)

            case "help":
                print("TODO...")

            case "add performance":
                new_perf = shared.performance_settings.default_settings
                new_perf['name'] = "Oops! Forgot to add a name." # Have a default name

                metadata = None
                if str(lines[1]).startswith("name:") and lines[2][0] == "{": # Try parsing json
                    new_perf['name'] = lines[1].split(':')[1].strip()
                    try:
                        metadata = json.loads("\n".join(lines[2:]))
                    except:
                        metadata = None

                    if metadata is not None:
                        try:
                            new_perf['custom_steps'] = metadata['steps']
                            new_perf['cfg'] = metadata['cfg']
                            new_perf['sampler_name'] = metadata['sampler_name']
                            new_perf['scheduler'] = metadata['scheduler']
                            new_perf['clip_skip'] = metadata['clip_skip']
                        except:
                            metadata = None

                if metadata is None:
                    # It wasn't json-data, try parsing as the example below:
                    #
                    # #!add performance
                    # name: WAN
                    # steps: 30
                    # cfg: 4
                    # scheduler: simple
                    # sampler: uni_pc

                    for line in lines[1:]:
                        kv = line.split(":")
                        if len(kv) < 2:
                            image = Image.open("html/error.png")
                            print(f"ERROR: Can't find key and value on line: {line}")
                            return []
                        k = kv[0].lower()

                        # Some helpful translations
                        if k in ['steps']:
                            k = 'custom_steps'
                        if k in ['sampler']:
                            k = 'sampler_name'
                        if k in ['clip-skip', 'clip skip', 'clipskip']:
                            k = 'clip_skip'

                        if k in ['custom_steps', 'clip_skip']: # Handle ints
                            v = int(kv[1])
                        elif k in ['cfg']: # Handle floats
                            v = float(kv[1])
                        elif k in ['sampler_name', 'scheduler', 'name']: # Handle strings
                            v = kv[1].strip()
                            if (k == 'sampler_name' and v not in KSampler.SAMPLERS):
                                image = Image.open("html/error.png")
                                print(f"ERROR: Unknown sampler: {v}")
                                print(f"Known Samplers: {KSampler.SAMPLERS}")
                                return []
                            if (k == 'scheduler' and v not in KSampler.SCHEDULERS):
                                image = Image.open("html/error.png")
                                print(f"ERROR: Unknown scheduler: {v}")
                                print(f"Known Schedulers: {KSampler.SCHEDULERS}")
                                return []
                        else:
                            image = Image.open("html/error.png")
                            print(f"ERROR: Unknown key: {k}")
                            return []
                        new_perf[k] = v

                perf_options = shared.performance_settings.load_performance()
                try:
                    opts = {
                        "custom_steps": new_perf['custom_steps'],
                        "cfg": new_perf['cfg'],
                        "sampler_name": new_perf['sampler_name'],
                        "scheduler": new_perf['scheduler'],
                        "clip_skip": new_perf['clip_skip'],
                    }
                    perf_options[new_perf['name']] = opts

                    shared.performance_settings.save_performance(perf_options)
                    print(f"#!: Saved performance: {new_perf['name']}: {opts}")
                    shared.update_cfg()
                except Exception as e:
                    image = Image.open("html/error.png")
                    print(f"ERROR: {e}")

            case "clean cache":
                gradio_cache = os.environ['GRADIO_TEMP_DIR']
                import shutil
                try:
                    if gradio_cache.endswith('ruinedfooocus_cache'):
                        shutil.rmtree(gradio_cache, ignore_errors=True)
                        print("Cache cleared. Please reload the browser to avoid errors.")
                    else:
                        print(f"ERROR: {gradio_cache} doesn't look like a RF cache. Not deleting.")
                except FileNotFoundError:
                    pass
                except PermissionError:
                    pass
                except Exception as e:
                    print(f"ERROR: {str(e)}")

            case "branch":
                import pygit2
                repo = pygit2.Repository(".")
                print(f"#!: Switching to branch {data}. Please restart.")
                try:
                    repo.checkout(repo.branches[data])
                except Exception as e:
                    print(f"ERROR: {e}")

            case _:
                print(f"ERROR: Unknown command #!{cmd}")
                image = Image.open("html/error.png")

        # Return finished image to preview
        if callback is not None:
            callback(gen_data["steps"], 0, 0, gen_data["steps"], image)

        return []
