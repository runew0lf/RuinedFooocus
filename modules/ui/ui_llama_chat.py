import gradio as gr
from shared import path_manager
import modules.async_worker as worker
from pathlib import Path
import json
from PIL import Image
import base64

def create_chat():
    def info_from_char(file):
        with Image.open(str(file)) as i:
            i.getexif()
            if 'chara' in i.info:
                d = i.info.get('chara', None)
            else:
                return None

        b = base64.b64decode(d) 
        j = json.loads(b)
        spec= j.get('spec', '')

        if spec == 'chara_card_v3':
            name = j['data']['name']
            greeting = j['data']['first_mes']
            avatar = file
            personality = j['data']['personality']
            scenario = j['data']['scenario']
            summary = j['data']['description']
        elif spec == 'chara_card_v2':
            name = j['data']['name']
            greeting = j['data']['first_mes']
            avatar = file
            personality = j['data']['personality']
            scenario = j['data']['scenario']
            summary = j['data']['description']
        elif all(key in j for key in ('name', 'first_mes', 'personality', 'scenario')):
            # chara_card_v1
            name = j['name']
            greeting = j['first_mes']
            avatar = file
            personality = j['personality']
            scenario = j['scenario']
            summary = j.get('summary', '')
        else:
            print(f"WARNING: Can not parse {file}, skipping.")
            return None

        info = {
            "name": name,
            "greeting": greeting,
            "avatar": avatar,
            "system": f"Your name is {name}.\nYou are: {personality}\nScenario: {scenario}",
            "embed": json.dumps([["text", f"Summary: {summary}"]]),
        }
        return info

    def llama_get_assistants():
        names = []
        folder_path = Path("chatbots")
        for path in folder_path.rglob("*"):
            if path.is_dir():
                try:
                    with open(path / "info.json" , "r", encoding='utf-8') as f:
                        info = json.load(f)
                    names.append((info["name"], str(path)))
                except Exception as e:
                    print(f"ERROR: in folder {path}: {e}")
                    pass
            else:
                # Ignore png's that has a info.json in the same folder
                if str(Path(path).suffix).lower() == ".png" and not Path(Path(path).parent / "info.json").exists():
                    # Try as aichar card
                    try:
                        character = info_from_char(path)
                        if character is not None:
                            names.append((character.get('name', '???'), str(path)))
                    except Exception as e:
                        print(f"ERROR: in character card {path}: {e}")
                        pass

        names.sort(key=lambda x: x[0].casefold())
        return names

    def gr_llama_get_assistants():
        return {
            llama_assistants: gr.update(
                choices=llama_get_assistants(),
            )
        }

    def _llama_select_assistant(dropdown):
        character = Path(dropdown)
        try:
            if character.is_dir():
                with open(character / "info.json", "r", encoding='utf-8') as f:
                    info = json.load(f)
                    if "avatar" not in info:
                        info["avatar"] = character / "avatar.png"
                    if "embed" in info:
                        info["embed"] = json.dumps(info["embed"])
                    else:
                        info["embed"] = json.dumps([])
            else: 
                info = info_from_char(character)

        except Exception as e:
            print(f"ERROR: {dropdown}: {e}")
            info = {
                "name": "Error",
                "greeting": "Error!",
                "avatar": "html/error.png",
                "system": "Everything is broken.",
                "embed": json.dumps([]),
            }
            pass
        info["chatstart"] = [{"role": "assistant", "content": info["greeting"]}]
        return info

    def llama_select_assistant(dropdown):
        info = _llama_select_assistant(dropdown)
        return {
            llama_chat: gr.update(value=info["chatstart"]),
            llama_msg: gr.update(value=""),
            llama_avatar: gr.update(
                value=info["avatar"],
                label=info["name"],
            ),
            llama_system: gr.update(value=info["system"]),
            llama_embed: gr.update(value=info["embed"])
        }


    with gr.Blocks() as app_llama_chat:
        with gr.Row():
            with gr.Column(scale=3), gr.Group():
                # FIXME!!! start value should be read from some info.json
                default_bot = "chatbots/rf_support_troll"
                llama_chat = gr.Chatbot(
                    label="",
                    show_label=False,
                    height=600,
                    resizable=True,
                    allow_tags=["think", "thinking"],
                    buttons=['copy_all'],
                    value=_llama_select_assistant(default_bot)["chatstart"],
                )
                llama_msg = gr.Textbox(
                    show_label=False,
                )
                llama_sent = gr.Textbox(visible='hidden')
            with gr.Column(scale=2), gr.Group():
                llama_avatar = gr.Image(
                    value=_llama_select_assistant(default_bot)["avatar"],
                    label=_llama_select_assistant(default_bot)["name"],
                    height=400,
                    width=400,
                    show_label=True,
                )
                with gr.Row():
                    llama_assistants = gr.Dropdown(
                        choices=llama_get_assistants(),
                        value=_llama_select_assistant(default_bot)["name"],
                        show_label=False,
                        interactive=True,
                        scale=7,
                    )
                    llama_reload = gr.Button(
                        value="↻",
                        scale=1,
                    )
                llama_system = gr.Textbox(
                    visible='hidden',
                    value=_llama_select_assistant(default_bot)["system"],
                )
                llama_embed = gr.Textbox(
                    visible='hidden',
                    value=_llama_select_assistant(default_bot)["embed"],
                )

        def llama_get_text(message):
            return "", message

        def llama_respond(message, system, embed, chat_history):
            chat_history.append({"role": "user", "content": message})

            gen_data = {
                "task_type": "llama",
                "system": system,
                "embed": embed,
                "history": chat_history,
            }

            # Add work
            task_id = worker.add_task(gen_data.copy())

            # Wait for result
            finished = False
            while not finished:
                flag, product = worker.task_result(task_id)
                if flag == "preview":
                    yield product
                elif flag == "results":
                    finished = True

            chat_history.append({"role": "assistant", "content": product})
            yield chat_history

        llama_msg.submit(
            fn=llama_get_text,
            api_visibility='undocumented',
            inputs=[llama_msg],
            outputs=[llama_msg, llama_sent]
        ).then(
            fn=llama_respond,
            api_visibility='undocumented',
            inputs=[llama_sent, llama_system, llama_embed, llama_chat],
            outputs=[llama_chat]
        )

        llama_assistants.select(
            fn=llama_select_assistant,
            api_visibility='undocumented',
            inputs=[llama_assistants],
            outputs=[llama_chat, llama_msg, llama_avatar, llama_system, llama_embed]
        )
        llama_reload.click(
            fn=gr_llama_get_assistants,
            api_visibility='undocumented',
            outputs=[llama_assistants]
        )

    return app_llama_chat
