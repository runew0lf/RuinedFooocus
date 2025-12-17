import os
import re
import random
import json

from modules.sdxl_styles import apply_style, allstyles
from random_prompt.build_dynamic_prompt import (
    build_dynamic_prompt,
    build_dynamic_negative,
)


def process_metadata(gen_data):
    try:
        if isinstance(gen_data["prompt"], dict):
            meta = gen_data["prompt"]
        else:
            meta = json.loads(gen_data["prompt"])
        meta = dict((k.lower(), v) for k, v in meta.items())
        gen_data.update(meta)
        if "prompt" in meta:
            gen_data["style_selection"] = []

        if "steps" in meta:
            gen_data["custom_steps"] = int(meta["steps"])
    except:
        pass
    return gen_data


def get_promptlist(gen_data):
    return gen_data["prompt"].split("---")


def process_wildcards(wildcard_text, directory="wildcards"):
    # removed regex method
    placeholders = []
    splitup = wildcard_text.split("__")
    for i in range(len(splitup)):
        if i % 2 != 0:  # check if index is odd
            placeholders.append(splitup[i])

    placeholder_choices = {}  # Store random choices for each placeholder
    official_directory = "wildcards_official"
    directories = []
    directories.append(directory)
    directories.append(official_directory)

    for placeholder in placeholders:
        # Skip onebuttonprompt wildcards for now; handled below
        if placeholder.startswith("onebutton"):
            continue

        elif placeholder not in placeholder_choices:
            found = False
            for dir in directories:
                for root, dirs, files in os.walk(dir):
                    if f"{placeholder}.txt" in files:
                        file_path = os.path.join(root, f"{placeholder}.txt")
                        with open(file_path, encoding="utf-8") as f:
                            words = [
                                word.strip()
                                for word in f.read().splitlines()
                                if not word.startswith("#")
                            ]
                        placeholder_choices[placeholder] = words
                        found = True
                        break
                if found == True:
                    break

            if not found:
                print(
                    f"Error: Could not find file {placeholder}.txt in {directory} or its subdirectories."
                )
                placeholder_choices[placeholder] = [f"{placeholder}"]

    for placeholder in placeholders:
        random_choice = ""

        # Some one button prompt specials
        if placeholder.startswith("onebutton"):
            subjectoverride = ""
            placeholdersplit = placeholder.split(":", 1)
            if len(placeholdersplit) > 1:
                subjectoverride = placeholdersplit[1]

            if placeholder.startswith("onebuttonprompt"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    givensubject=subjectoverride,
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonsubject"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonhumanoid"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonmale"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="male",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonfemale"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="female",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonanimal"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="animal - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonobject"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="object",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonlandscape"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="landscape - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonconcept"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="concept - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonartist"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    onlyartists=True,
                    artists=subjectoverride or "all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebutton1girl"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="female",
                    advancedprompting=False,
                    base_model="Anime Model",
                )
            elif placeholder.startswith("onebutton1boy"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="male",
                    advancedprompting=False,
                    base_model="Anime Model",
                )
            elif placeholder.startswith("onebuttonfurry"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="animal - all",
                    advancedprompting=False,
                    base_model="Anime Model",
                )
            # failover
            else:
                random_choice = build_dynamic_prompt(
                    insanitylevel=3,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    advancedprompting=False,
                    base_model="SDXL",
                )

        # Regular wildcards
        else:
            random_choice = random.choice(placeholder_choices[placeholder])

        wildcard_text = re.sub(
            rf"__{placeholder}__", random_choice, wildcard_text, count=1
        )

    return wildcard_text


def process_prompt(style, prompt, negative, gen_data=[]):
    if "obp_assume_direct_control" in gen_data and gen_data["obp_assume_direct_control"]:
        prompt = build_dynamic_prompt(
            insanitylevel=gen_data["obp_insanitylevel"],
            forcesubject=gen_data["obp_subject"],
            artists=gen_data["obp_artist"],
            subtypeobject=gen_data["obp_chosensubjectsubtypeobject"],
            subtypehumanoid=gen_data["obp_chosensubjectsubtypehumanoid"],
            subtypeconcept=gen_data["obp_chosensubjectsubtypeconcept"],
            gender=gen_data["obp_chosengender"],
            imagetype=gen_data["obp_imagetype"],
            imagemodechance=gen_data["obp_imagemodechance"],
            givensubject=gen_data["obp_givensubject"],
            smartsubject=gen_data["obp_smartsubject"],
            overrideoutfit=gen_data["obp_givenoutfit"],
            prefixprompt=gen_data["obp_prefixprompt"],
            suffixprompt=gen_data["obp_suffixprompt"],
            giventypeofimage=gen_data["obp_giventypeofimage"],
            antivalues=gen_data["obp_antistring"],
            OBP_preset=gen_data["OBP_preset"],
            advancedprompting=False,
            base_model=gen_data["OBP_modeltype"],
            prompt_enhancer=gen_data["OBP_promptenhance"],
        )

    # styles
    pattern = re.compile(r"<style:([^>]+)>")
    styles = [] if style is None else style.copy()
    for match in re.finditer(pattern, prompt):
        styles += [f"Style: {match.group(1)}"]
    prompt = re.sub(pattern, "", prompt)
    if "lora_keywords" in gen_data:
        keywords = gen_data["lora_keywords"]
    else:
        keywords = ""
    p_txt, n_txt = apply_style(styles, prompt, negative, keywords)

    # wildcards
    wildcard_pattern = r"__([\w\-:]+)__"
    wildcard_pattern_onebutton = r"__([\w]+:[^\s_]+(?:[^\s_]+|\s(?=[\w:]+))*)__"
    while (
        (match := re.search(wildcard_pattern, p_txt))
        or (match := re.search(wildcard_pattern_onebutton, p_txt))
    ) is not None:
        p_txt = process_wildcards(p_txt)

    # apply auto negative prompt if enabled
    if "auto_negative" in gen_data and gen_data["auto_negative"] == True:
        n_txt = build_dynamic_negative(
            positive_prompt=p_txt, existing_negative_prompt=n_txt, base_model="SDXL"
        )
    return p_txt, n_txt


def parse_loras(prompt, negative):
    pattern = re.compile(r"<lora:([^>]+):(\d*\.*\d+)>")
    loras = []
    for match in re.finditer(pattern, prompt):
        loras.append({"name": f"{match.group(1)}.safetensors", "weight": float(match.group(2)), "hash": None})
    for match in re.finditer(pattern, negative):
        loras.append({"name": f"{match.group(1)}.safetensors", "weight": float(match.group(2)), "hash": None})
    return loras, re.sub(pattern, "", prompt), re.sub(pattern, "", negative)


def prompt_switch_per_step(prompt, steps):
    # Find all occurrences of [option1|option2|...] in the input string
    # basic prompt editing:
    # [bla|bla2|bla3] -> repeat each prompt

    # A1111 style prompt editing:
    # [bla:bla2:16] --> after step 16, move on from bla1 to bla2
    # [bla:bla2:0.5] --> after 50% of steps move on from bla1 to bla2

    # [bla1::16] --> remove bla1 after step 16
    # [bla1::0.5] --> remove bla1 after 50% of steps

    # Lets explore this space more, RF!
    # [bla|bla2:bla3|bla4:16] --> after5 step 16, move on from [bla1|bla2] to [bla3|bla4]
    # [bla:bla2:16] --> after step 16, move on from bla1 to bla2
    # [bla1:0.5::0.75] --> Start and remove bla1 from the prompt at 50% and 75%

    # [bla1~bla2] --> Same as | but in steps of 10% of steps
    # [bla^bla2] --> slowly switch bla to bla2 with a peak
    # [bla?bla2] --> random switch
    # [bla1/bla2] --> slowly switch bla to bla2, but keeps bla2 after half
    # [bla1\bla2] --> start with bla1, but after half, slowly transform into bla2

    prompt_per_step = []

    # step through all steps
    for i in range(0, steps):
        try:
            prompt_per_step.append(prompt)
            while "[" in prompt_per_step[i]:
                startoflastpattern = prompt_per_step[i].rfind("[")
                startoflastpatterncomplete = prompt_per_step[i][startoflastpattern:]
                switchpattern = r"\[(.*?)\]"
                allswitchpatterns = re.findall(
                    switchpattern, startoflastpatterncomplete
                )

                for switchpattern in allswitchpatterns:
                    switchpattern = "[" + switchpattern + "]"
                    matchfound = False

                    # start with basic matching [bla:bla2:16]
                    basic_match_pattern = r"\[((?!.*::).*?):([^\]]*?):([^\]]*?)\]"
                    matches = re.finditer(basic_match_pattern, switchpattern)

                    for match in matches:
                        replacement = ""

                        exact_match = match.group(0)

                        number = match.group(3)
                        parts = number.split(".")
                        intnumber = int(parts[0])
                        decnumber = float(number) if len(parts) > 1 else 0

                        if (intnumber != 0 and intnumber > i) or (
                            intnumber == 0 and int(steps * decnumber) > i
                        ):
                            replacement = match.group(1)
                        if (intnumber != 0 and intnumber <= i) or (
                            intnumber == 0 and int(steps * decnumber) <= i
                        ):
                            replacement = match.group(2)

                        # a trick for downstream! nice ;)
                        if (
                            "|" in replacement
                            or "~" in replacement
                            or "^" in replacement
                            or "?" in replacement
                            or "/" in replacement
                            or "\\" in replacement
                        ):
                            replacement = "[" + replacement + "]"
                        prompt_to_append = prompt_per_step[i].replace(
                            exact_match, replacement, 1
                        )

                        prompt_per_step[i] = prompt_to_append
                        matchfound = True

                    # Now do basic closing with [text::16]
                    basic_match_pattern = r"\[(.*?)::(.*?)\]"
                    matches = re.finditer(basic_match_pattern, switchpattern)
                    if matchfound == False:
                        for match in matches:
                            replacement = ""

                            exact_match = match.group(0)

                            # logic for ( : ::)
                            startintnumber = 0
                            startdecnumber = 0
                            print("match group 1 thingy")
                            print(match.group(1))
                            if ":" in match.group(1):
                                number = match.group(1).split(":")
                                parts = number[1].split(".")

                                startintnumber = int(parts[0])
                                startdecnumber = (
                                    float(number[1]) if len(parts) > 1 else 0
                                )
                                tempreplacement = number[0]
                            else:
                                tempreplacement = match.group(1)

                            number = match.group(2)
                            parts = number.split(".")
                            intnumber = int(parts[0])
                            decnumber = float(number) if len(parts) > 1 else 0

                            if (intnumber != 0 and intnumber > i) or (
                                intnumber == 0 and int(steps * decnumber) > i
                            ):
                                if (
                                    (startintnumber != 0 and startintnumber <= i)
                                    or (
                                        startdecnumber != 0
                                        and int(steps * startdecnumber) <= i
                                    )
                                    or (startintnumber == 0 and startdecnumber == 0)
                                ):
                                    replacement = tempreplacement

                            if (
                                "|" in replacement
                                or "~" in replacement
                                or "^" in replacement
                                or "?" in replacement
                                or "/" in replacement
                                or "\\" in replacement
                            ):
                                replacement = "[" + replacement + "]"
                            prompt_to_append = prompt_per_step[i].replace(
                                exact_match, replacement, 1
                            )

                            prompt_per_step[i] = prompt_to_append
                            matchfound = True

                    # Now do basic starting with [text:16]
                    basic_match_pattern = r"\[(.*?):(.*?)\]"
                    matches = re.finditer(basic_match_pattern, switchpattern)
                    if matchfound == False:
                        for match in matches:
                            replacement = ""

                            exact_match = match.group(0)

                            number = match.group(2)
                            parts = number.split(".")
                            intnumber = int(parts[0])
                            decnumber = float(number) if len(parts) > 1 else 0

                            if (intnumber != 0 and intnumber <= i) or (
                                intnumber == 0 and int(steps * decnumber) <= i
                            ):
                                replacement = match.group(1)
                            if (
                                "|" in replacement
                                or "~" in replacement
                                or "^" in replacement
                                or "?" in replacement
                                or "/" in replacement
                                or "\\" in replacement
                            ):
                                replacement = "[" + replacement + "]"
                            prompt_to_append = prompt_per_step[i].replace(
                                exact_match, replacement, 1
                            )

                            prompt_per_step[i] = prompt_to_append
                            matchfound = True

                    # do long swapping
                    if "~" in switchpattern:
                        options_pattern = r"\[([^~\]]+(?:~[^~\]]+)*)\]"
                        matches = re.finditer(options_pattern, switchpattern)
                        options_list = []
                        exact_matches = []

                        for match in matches:
                            options = (
                                match.group(1).split("~")
                                if "~" in match.group(1)
                                else [match.group(1)]
                            )
                            options_list.append(options)
                            exact_matches.append(match.group(0))

                            prompt_to_append = prompt_per_step[i]
                            factor = max(
                                round(steps / 10), 2
                            )  # minimum of 2, else just use |

                            for options, exact_match in zip(
                                options_list, exact_matches
                            ):
                                replacement = options[
                                    int((i) / factor) % len(options)
                                ]  # Use modulo to cycle through options
                                if (
                                    "|" in replacement
                                    or "~" in replacement
                                    or "^" in replacement
                                    or "?" in replacement
                                    or "/" in replacement
                                    or "\\" in replacement
                                ):
                                    replacement = "[" + replacement + "]"
                                prompt_to_append = prompt_to_append.replace(
                                    exact_match, replacement, 1
                                )

                            prompt_per_step[i] = prompt_to_append

                    # do lerp flip swapping
                    if "^" in switchpattern:
                        options_pattern = r"\[([^^\]]+(?:\^[^^\]]+)*)\]"
                        matches = re.finditer(options_pattern, switchpattern)
                        options_list = []
                        exact_matches = []

                        factor = i / steps

                        for match in matches:
                            options = (
                                match.group(1).split("^")
                                if "^" in match.group(1)
                                else [match.group(1)]
                            )
                            options_list.append(options)
                            exact_matches.append(match.group(0))

                            prompt_to_append = prompt_per_step[i]

                            if i > steps / 2:
                                options.reverse()

                            for options, exact_match in zip(
                                options_list, exact_matches
                            ):
                                replacement = options[
                                    round(i + (steps - i) * factor) % len(options)
                                ]  # Use lerp type to swap
                                if (
                                    "|" in replacement
                                    or "~" in replacement
                                    or "^" in replacement
                                    or "?" in replacement
                                    or "/" in replacement
                                    or "\\" in replacement
                                ):
                                    replacement = "[" + replacement + "]"
                                prompt_to_append = prompt_to_append.replace(
                                    exact_match, replacement, 1
                                )

                            prompt_per_step[i] = prompt_to_append

                    # do starting half lerp flip swapping
                    if "/" in switchpattern:
                        options_pattern = r"\[([^\/]+(?:\/[^\/]+)*)\]"
                        matches = re.finditer(options_pattern, switchpattern)
                        options_list = []
                        exact_matches = []

                        factor = i / steps

                        for match in matches:
                            options = (
                                match.group(1).split("/")
                                if "/" in match.group(1)
                                else [match.group(1)]
                            )
                            options_list.append(options)
                            exact_matches.append(match.group(0))

                            prompt_to_append = prompt_per_step[i]

                            for options, exact_match in zip(
                                options_list, exact_matches
                            ):
                                if i > steps / 2:
                                    replacement = options[len(options) - 1]
                                else:
                                    replacement = options[
                                        round(i + (steps - i) * factor) % len(options)
                                    ]  # Use lerp type to swap
                                if (
                                    "|" in replacement
                                    or "~" in replacement
                                    or "^" in replacement
                                    or "?" in replacement
                                    or "/" in replacement
                                    or "\\" in replacement
                                ):
                                    replacement = "[" + replacement + "]"
                                prompt_to_append = prompt_to_append.replace(
                                    exact_match, replacement, 1
                                )

                            prompt_per_step[i] = prompt_to_append

                    # do closing half lerp flip swapping
                    if "\\" in switchpattern:
                        options_pattern = r"\[([^\/]+(?:\/\[^\/]+)*)\]"
                        matches = re.finditer(options_pattern, switchpattern)
                        options_list = []
                        exact_matches = []

                        factor = i / steps

                        for match in matches:
                            options = (
                                match.group(1).split("\\")
                                if "\\" in match.group(1)
                                else [match.group(1)]
                            )
                            options_list.append(options)
                            exact_matches.append(match.group(0))

                            prompt_to_append = prompt_per_step[i]

                            if i > steps / 2:
                                options.reverse()

                            for options, exact_match in zip(
                                options_list, exact_matches
                            ):
                                if i < steps / 2:
                                    replacement = options[0]
                                else:
                                    replacement = options[
                                        round(i + (steps - i) * factor) % len(options)
                                    ]  # Use lerp type to swap
                                if (
                                    "|" in replacement
                                    or "~" in replacement
                                    or "^" in replacement
                                    or "?" in replacement
                                    or "/" in replacement
                                    or "\\" in replacement
                                ):
                                    replacement = "[" + replacement + "]"
                                prompt_to_append = prompt_to_append.replace(
                                    exact_match, replacement, 1
                                )

                            prompt_per_step[i] = prompt_to_append

                    # do random prompt swapping
                    if "?" in switchpattern:
                        options_pattern = r"\[([^?]+(?:\?[^?]+)*)\]"
                        matches = re.finditer(options_pattern, switchpattern)
                        options_list = []
                        exact_matches = []

                        for match in matches:
                            options = (
                                match.group(1).split("?")
                                if "?" in match.group(1)
                                else [match.group(1)]
                            )
                            options_list.append(options)
                            exact_matches.append(match.group(0))

                            prompt_to_append = prompt_per_step[i]

                            for options, exact_match in zip(
                                options_list, exact_matches
                            ):
                                replacement = options[
                                    random.randint(0, len(options) - 1)
                                ]  # take a random value

                                if (
                                    "|" in replacement
                                    or "~" in replacement
                                    or "^" in replacement
                                    or "?" in replacement
                                    or "/" in replacement
                                    or "\\" in replacement
                                ):
                                    replacement = "[" + replacement + "]"
                                prompt_to_append = prompt_to_append.replace(
                                    exact_match, replacement, 1
                                )

                            prompt_per_step[i] = prompt_to_append

                    # do prompt swapping
                    if "|" in switchpattern:
                        options_pattern = r"\[([^|\]]+(?:\|[^|\]]+)*)\]"
                        matches = re.finditer(options_pattern, switchpattern)
                        options_list = []
                        exact_matches = []

                        for match in matches:
                            options = (
                                match.group(1).split("|")
                                if "|" in match.group(1)
                                else [match.group(1)]
                            )
                            options_list.append(options)
                            exact_matches.append(match.group(0))

                            prompt_to_append = prompt_per_step[i]

                            for options, exact_match in zip(
                                options_list, exact_matches
                            ):
                                replacement = options[
                                    i % len(options)
                                ]  # Use modulo to cycle through options
                                if (
                                    "|" in replacement
                                    or "~" in replacement
                                    or "^" in replacement
                                    or "?" in replacement
                                    or "/" in replacement
                                    or "\\" in replacement
                                ):
                                    replacement = "[" + replacement + "]"
                                prompt_to_append = prompt_to_append.replace(
                                    exact_match, replacement, 1
                                )

                            prompt_per_step[i] = prompt_to_append

                    # if there is no pattern, then just replace the value?
                    if (
                        "|" not in switchpattern
                        and "~" not in switchpattern
                        and "^" not in switchpattern
                        and "?" not in switchpattern
                        and "/" not in switchpattern
                        and "\\" not in switchpattern
                        and ":" not in switchpattern
                    ):
                        replacement = switchpattern
                        replacement = replacement.replace("[", "")
                        replacement = replacement.replace("]", "")

                        prompt_to_append = prompt_per_step[i]
                        prompt_to_append = prompt_to_append.replace(
                            switchpattern, replacement, 1
                        )
                        prompt_per_step[i] = prompt_to_append

        except ValueError as e:
            print("There seems to be a mistake in the prompt.")
            print(e)
            break

#    print("All prompts generated after applying logic:")
#    for i in range(0, steps):
#        print("Step:" + str(i + 1))
#        print(prompt_per_step[i])
#    print("")
    return prompt_per_step
