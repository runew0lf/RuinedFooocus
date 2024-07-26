import random
import csv
from os.path import exists
from csv import DictReader
from pathlib import Path
from modules.prompt_expansion import PromptExpansion
from random_prompt import build_dynamic_prompt

DEFAULT_STYLES_FILE = Path("settings/styles.default")
STYLES_FILE = Path("settings/styles.csv")

prompt_expansion = PromptExpansion()


def load_styles():
    default_styles = []
    styles = []

    with open(DEFAULT_STYLES_FILE) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            default_styles.append(row)

    if exists(STYLES_FILE):
        with open(STYLES_FILE) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                styles.append(row)

    # Add any missing default styles
    changed = False
    for row in default_styles:
        if row not in styles:
            styles.append(row)
            changed = True

    if changed:
        with open(STYLES_FILE, "w", newline='') as f:
            csv_writer = csv.writer(f)
            for row in styles:
                csv_writer.writerow(row)


    with STYLES_FILE.open("r") as f:
        reader = DictReader(f)
        styles = list(reader)

    default_style = {"name": "None", "prompt": "{prompt}", "negative_prompt": ""}
    random_style = {
        "name": "Style: Pick Random",
        "prompt": "{prompt}",
        "negative_prompt": "",
    }
    lora_keywords_style = {
        "name": "LoRA keywords",
        "prompt": "{prompt} {lora_keywords}",
        "negative_prompt": "",
    }
    flufferizer_style = {
        "name": "Flufferizer",
        "prompt": "{prompt}",
        "negative_prompt": "",
    }
    hyperprompt_style = {
        "name": "Hyperprompt",
        "prompt": "{prompt}",
        "negative_prompt": "",
    }

    styles.insert(0, hyperprompt_style)
    styles.insert(0, flufferizer_style)
    styles.insert(0, lora_keywords_style)
    styles.insert(0, random_style)
    styles.insert(0, default_style)

    return {s["name"]: (s["prompt"], s["negative_prompt"]) for s in styles}


def apply_style(style, prompt, negative_prompt, lora_keywords):
    output_prompt = ""
    output_negative_prompt = ""
    temp_style_prompt = prompt
    bFlufferizer = False
    bHyperprompt = False
    artifylist = []
    artifystylelist = []
    index = 0

    if not style:
        return prompt, negative_prompt

    while "Style: Pick Random" in style:
        style[style.index("Style: Pick Random")] = random.choice(allstyles)

    for s in style.copy():
        _s = s.upper().strip()
        if _s in map(str.upper, ["Flufferizer", "Style: Flufferizer"]):
            bFlufferizer = True
            del style[style.index(s)]

        if _s in map(str.upper, ["Hyperprompt", "Style: Hyperprompt"]):
            bHyperprompt = True
            del style[style.index(s)]

        if _s in map(str.upper, ["LoRA keywords", "Style: LoRA keywords"]):
            style[style.index(s)] = "LoRA keywords" # Make sure it has the correct name

    if bHyperprompt:
        prompt = build_dynamic_prompt.one_button_superprompt(prompt=prompt)
        temp_style_prompt = prompt
        print("Hypered prompt: " + prompt)

    while index < len(style):
        if style[index].startswith("Artify"):
            artifylist.append(style[index].replace("Artify: ", ""))
            artifystylelist.append(style[index])
        index += 1

    style = [x for x in style if x not in artifystylelist]

    for s in style:
        p, n = styles.get(s, default_style)
        output_prompt = p + ", "
        output_negative_prompt += n + ", "

        temp_style_prompt = output_prompt.replace("{prompt}", temp_style_prompt)
        output_prompt = temp_style_prompt.replace(", ,", ", ")


    # prep outputprompt for use in Flufferize and Artify
    if output_prompt == "":
        output_prompt = prompt

    for artist in artifylist:
        output_prompt = build_dynamic_prompt.artify_prompt(prompt=output_prompt, artists=artist)

    if bFlufferizer:
        output_prompt = prompt_expansion.expand_prompt(output_prompt)

    output_prompt = output_prompt.replace("{lora_keywords}", lora_keywords)
    output_negative_prompt += ", " + negative_prompt

    return output_prompt, output_negative_prompt


styles = load_styles()
default_style = styles["None"]
allstyles = [x for x in load_styles() if x.startswith("Style")]
allstyles.remove("Style: Pick Random")
