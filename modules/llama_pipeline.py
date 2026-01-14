import re
try:
    import xllamacpp as xlc
    Llama = "xlc"
except:
    print("ERROR: Could not load Llama.")
    Llama = None
from txtai import Embeddings
from modules.util import TimeIt
from pathlib import Path
from modules.util import url_to_filename, load_file_from_url
from shared import path_manager, settings, local_url
import json
import xmltodict
import modules.async_worker as worker

def llama_names():
        names = []
        folder_path = Path("llamas")
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in [".txt"]:
                f = open(path, "r", encoding='utf-8')
                name = f.readline().strip()
                names.append((name, str(path)))
        names.sort(key=lambda x: x[0].casefold())
        return names

def run_llama(system_file, prompt):
        if Llama == None:
            return "Error: There is no Llama"
        name = None
        sys_pat = r"system:.*\n\n"
        system = re.match(sys_pat, prompt, flags=re.M|re.I)
        if system is not None: # Llama system-prompt provided in the ui-prompt
            name = "Llama"
            system_prompt = re.sub(r"^[^:]*: *", "", system.group(0), flags=re.M|re.I)
            prompt = re.sub(sys_pat, "", prompt)
        else:
            try:
                file = open(system_file, "r", encoding='utf-8')
                name = name if name is not None else file.readline().strip()
                system_prompt = file.read().strip()
            except:
                print(f"LLAMA ERROR: Could not open file {system_file}")
                return prompt

        llama = pipeline()
        llama.load_base_model()

        with TimeIt(""):
            print(f"# System:\n{system_prompt.strip()}\n")
            print(f"# User:\n{prompt.strip()}\n")
            print(f"# {name}: (Thinking...)")
            try:

                ret = llama.llm.handle_completions(
                    {
                        "max_tokens": settings.default_settings.get("llm_hp_maxtokens", 256),
                        "prompt": system_prompt + "\n\n" + prompt,
                    }
                )
                res = ret['choices'][0]['text']
            except Exception as e:
                print(f"LLAMA ERROR: {e}")
                res = prompt

            print(f"{res.strip()}\n")

        del llama.llm
        llama.llm = None

        return res

class pipeline:
    pipeline_type = ["llama"]

    llm = None
    embeddings = None
    embeddings_hash = ""

    def parse_gen_data(self, gen_data):
        return gen_data

    def load_base_model(self):
        localfile = settings.default_settings.get("llama_localfile", "Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_M.gguf")
        llm_path = path_manager.get_folder_file_path(
            "llm",
            localfile,
            default = Path(path_manager.model_paths["llm_path"]) / localfile
        )
        with TimeIt("Load LLM"):
            print(f"Loading {localfile}")

            params = xlc.CommonParams()
            params.prompt = ""
            params.model.path = str(llm_path)
            params.n_predict = int(settings.default_settings.get("llm_n_predict", -1))
            params.n_ctx = int(settings.default_settings.get("llm_n_ctx", 2048))
            params.n_gpu_layers = int(settings.default_settings.get("llm_n_gpu_layers", -1))
            params.ctx_shift = True
            params.cpuparams.n_threads = 4
            params.cpuparams_batch.n_threads = 2
            params.endpoint_metrics = False
            params.use_jinja = True

            self.llm = xlc.Server(params) # FIXME hide output?

        self.embeddings = None

    def index_source(self, source):
        if self.embeddings == None:
            self.embeddings = Embeddings(content=True)
            self.embeddings.initindex(reindex=True)

        match source[0]:

            case "url":
                print(f"Read {source[1]}")
                filename = load_file_from_url(
                    source[1],
                    model_dir="cache/embeds",
                    progress=True,
                    file_name=url_to_filename(source[1]),
                )
                file = open(filename, "r", encoding='utf-8')
                data = file.read()
                file.close()

                if source[1].endswith(".md"):
                    data = data.split("\n#")
                elif source[1].endswith(".txt"):
                    data = data.split("\n\n")

            case "text":
                data = source[1]

            case _:
                print("WARNING: Unknown embedding type {source[0]}")
                return

        if data:
            self.embeddings.upsert(data)


    def process(self, gen_data):
        if Llama == None:
            return "Error: There is no Llama"

        worker.add_result(
            gen_data["task_id"],
            "preview",
            gen_data["history"] + [{"role": "assistant", "content": "🤔"}]
        )

        if self.llm == None:
            self.load_base_model()

        # load embeds?
        # FIXME should dump the entire gen_data["embed"] to index_source() and have it sort it out
        embed = json.loads(gen_data['embed'])
        if self.embeddings_hash != str(embed):
            self.embeddings_hash = str(embed)
            self.embeddings = None
        if embed:
            if not self.embeddings: # If chatbot has embeddings to index, check that we have them.
                for source in embed:
                    self.index_source(source)
        else:
            self.embeddings = None

        system_prompt = gen_data["system"]

        h = gen_data["history"]

        if self.embeddings:
            q = h[-1]["content"]
            context = "This some context that will help you answer the question:\n"
            for data in self.embeddings.search(q, limit=3):
                #if data["score"] >= 0.5:
                context += data["text"] + "\n\n"
            system_prompt += context

        if settings.default_settings.get("enable_llm_tools", False):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "description": "Generates an image from a prompt.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "The prompt for the image"},
                            },
                        },
                    },
                },
            ]
            tool_prompt = "\nUse the tool when you intend to generate an image. You must make sure you use the correct format. The image will be shown to the user.\n"
        else:
            tools = None
            tool_prompt = ""
        chat = [{"role": "system", "content": system_prompt + tool_prompt}]
        history_len = settings.default_settings.get("llm_chat_history", 7) # Keep the some of the last messages in the discussion.
        history_len = -history_len if len(h) > history_len else -len(h)

        def clean_content(text):
            return re.sub('!\\[Image\\]\\([^(]*\\)', '', text) # Remove Image-markdown from LLM input
        for idx in range(history_len, 0):
            c = json.loads(json.dumps(h[idx])) # Thread safe Deep copy
            if isinstance(c['content'], str):
                c['content'] = clean_content(c['content'])
            else: 
                try:
                    c['content'][0]['text'] = clean_content(c['content'][0]['text'])
                except Exception as e:
                    print(f"LLM: ({e}): {c}")
            chat.append(c)

        print(f"Thinking...")
        with TimeIt("LLM thinking"):
            result = []

            result = {
                "text": "",
                "tool": {
                    "function": None,
                    "arguments": "",
                }
            }

            def callback(chunk):
                if len(chunk.get('choices', [])) == 0:
                    return
                try:
                    delta = chunk['choices'][0]['delta']
                except:
                    print(f"ERROR: No delta? {chunk}")
                    return

                if 'content' in delta:
                    text = delta['content']

                    if text is not None:
                        #print(text, end="")
                        result['text'] += text
                        worker.add_result(
                            gen_data["task_id"],
                            "preview",
                            gen_data["history"] + [{"role": "assistant", "content": result['text']}]
                        )

                if 'tool_calls' in delta:
                    tool_call = delta['tool_calls'][0]['function'] # Simply assume we only have a single tool call
                    if 'name' in tool_call:
                        result['tool']['function'] = tool_call['name']
                    result['tool']['arguments'] += tool_call['arguments']

            self.llm.handle_chat_completions(
                {
                    "stream": True,
                    "messages": chat,
                    "tools": tools,
                },
                lambda d: callback(d),
            )

            text = result['text']

            call = None
            tool_error = f"![Error](gradio_api/file=html/error.png)"
            if settings.default_settings.get("enable_llm_tools", False) and result['tool']['function'] is not None:
                try:
                    task_id = -1

                    args = json.loads(result['tool']['arguments'])
                    prompt = args['prompt']

                    tmp_data = {
                        'task_type': "tool_call",
                        'task_id': task_id,
                        'silent': True,
                        'prompt': prompt,
                        'negative': "",
                        'loras': [
                            ("", f"{settings.default_settings.get('lora_1_weight', 1.0)} - {settings.default_settings.get('lora_1_model', 'None')}"),
                            ("", f"{settings.default_settings.get('lora_2_weight', 1.0)} - {settings.default_settings.get('lora_2_model', 'None')}"),
                            ("", f"{settings.default_settings.get('lora_3_weight', 1.0)} - {settings.default_settings.get('lora_3_model', 'None')}"),
                            ("", f"{settings.default_settings.get('lora_4_weight', 1.0)} - {settings.default_settings.get('lora_4_model', 'None')}"),
                            ("", f"{settings.default_settings.get('lora_5_weight', 1.0)} - {settings.default_settings.get('lora_5_model', 'None')}"),
                        ],
                        'style_selection': settings.default_settings['style'],
                        'seed': -1,
                        'base_model_name': settings.default_settings['base_model'],
                        'performance_selection': settings.default_settings['performance'],
                        'aspect_ratios_selection': settings.default_settings["resolution"],
                        'cn_selection': None,
                        'cn_type': None,
                        'silent': True,
                        'image_number': 1,
                    }

                    # unload llm model from memory?
                    # TODO: make this selectable for people with more ram/vram that is socialy acceptable
                    del self.llm
                    self.llm = None

                    info_txt = "(Generating image...)"
                    tmp_text = text + "\n" + info_txt

                    worker.add_result(
                        gen_data["task_id"],
                        "preview",
                        gen_data["history"] + [{"role": "assistant", "content": tmp_text}]
                    )

                    results = worker._process(tmp_data.copy())
                    file = results[0]
                    filename = str(file.relative_to(file.cwd()).as_posix())
                    url = "gradio_api/file=" + re.sub(r'[^/]+/\.\./', '', filename)
                    markdown = f"\n*{prompt}*\n\n![Image]({url})\n"

                    text += "\n" + markdown

                except Exception as e:
                    import traceback
                    print(f"ERROR:")
                    traceback.print_exc()
                    text += f"Error: {e}\n\n"
                    text += f"Call: {call}\n\n"
                    text += "Looks like I made a mistake. I really need to make sure I use the correct format. Do you want me to try again?"

        return text
