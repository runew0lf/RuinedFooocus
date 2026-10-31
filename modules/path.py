from pathlib import Path
import json
import os
try:
    # This can fail during the first run
    import requests
    from tqdm import tqdm
except:
    pass


class PathManager:
    DEFAULT_PATHS = {
        "path_checkpoints": ["../models/checkpoints/"],
        "path_diffusers": "../models/diffusers/",
        "path_diffusers_cache": "../models/diffusers_cache/",
        "path_loras": ["../models/loras/"],
        "path_controlnet": "../models/controlnet/",
        "path_vae": "../models/vae/",
        "path_preview": "../outputs/preview.jpg",
        "path_faceswap": "../models/faceswap/",
        "path_upscalers": "../models/upscale_models",
        "path_outputs": "../outputs/",
        "path_clip": "../models/clip/",
        "path_clip_vision": "../models/clip_vision/",
        "path_cache": "../cache/",
        "path_llm": "../models/llm",
        "path_inbox": "../models/inbox",
        "path_presets": "presets",
    }

    EXTENSIONS = [".pth", ".ckpt", ".bin", ".safetensors", ".gguf", ".merge"]

    # Add a dictionary to store file download information
    DOWNLOADABLE_FILES = {}

    name = None
    settings_path = None
    paths = None

    def __init__(self):
        from argparser import args
        self.name = args.settings
        self.set_settings_path(args.settings)
        self.paths = self.load_paths()
        self.model_paths = self.get_model_paths()
        self.upscaler_filenames = self.get_model_filenames(
            self.model_paths["upscaler_path"]
        )

        pathdb_folder = "modules/pathdb"
        files = os.listdir(pathdb_folder)
        for file in files:
            # Check if the file has a .json extension
            if file.endswith('.json'):
                file_path = os.path.join(pathdb_folder, file)

                try:
                    # Open and read the JSON file
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        self.DOWNLOADABLE_FILES.update(data)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    def set_settings_path(self, subfolder=None):
        self.subfolder = subfolder
        if self.subfolder in [None, "", "default"]:
            path = Path("settings/paths.json")
        else:
            path = Path(f"settings/{self.subfolder}/paths.json")
        if not path.parent.exists():
            path.parent.mkdir()
        self.settings_path = path

    def load_paths(self):
        paths = self.DEFAULT_PATHS.copy()
        if self.settings_path.exists():
            with self.settings_path.open() as f:
                paths.update(json.load(f))
        for key in self.DEFAULT_PATHS:
            if key not in paths:
                paths[key] = self.DEFAULT_PATHS[key]
        # Fix paths
        for key in ['path_checkpoints', 'path_loras']:
            if key in paths and not isinstance(paths[key], list): # Some folders should be lists
                paths[key] = [paths[key]]

        with self.settings_path.open("w") as f:
            json.dump(paths, f, indent=2)
        return paths

    def save_paths(self):
        paths = self.paths

#        for key in newpaths:
#            if key not in paths:
#                paths[key] = newpaths[key]
        with self.settings_path.open("w") as f:
            json.dump(paths, f, indent=2)
        return paths

    def get_model_paths(self):
        return {
            "modelfile_path": self.get_abspath_folder(self.paths["path_checkpoints"]),
            "diffusers_path": self.get_abspath_folder(self.paths["path_diffusers"]),
            "diffusers_cache_path": self.get_abspath_folder(
                self.paths["path_diffusers_cache"]
            ),
            "lorafile_path": self.get_abspath_folder(self.paths["path_loras"]),
            "controlnet_path": self.get_abspath_folder(self.paths["path_controlnet"]),
            "vae_path": self.get_abspath_folder(self.paths["path_vae"]),
            "temp_outputs_path": self.get_abspath_folder(self.paths["path_outputs"]),
            "temp_preview_path": self.get_abspath(self.paths["path_preview"]),
            "faceswap_path": self.get_abspath_folder(self.paths["path_faceswap"]),
            "upscaler_path": self.get_abspath_folder(self.paths["path_upscalers"]),
            "clip_path": self.get_abspath_folder(self.paths["path_clip"]),
            "clip_vision_path": self.get_abspath_folder(self.paths["path_clip_vision"]),
            "cache_path": self.get_abspath_folder(self.paths["path_cache"]),
            "llm_path": self.get_abspath_folder(self.paths["path_llm"]),
            "inbox_path": self.get_abspath_folder(self.paths["path_inbox"]),
            "preset_path": self.get_abspath_folder(self.paths["path_presets"]),
        }

    def get_abspath_folder(self, path):
        if isinstance(path, list):
            rc = []
            for folder in path:
                rc.append(self.get_abspath(folder))
        else:
            rc = self.get_abspath(path)
            if not rc.exists():
                rc.mkdir(parents=True, exist_ok=True)
        return rc

    def get_abspath(self, path):
        return Path(path) if Path(path).is_absolute() else Path(__file__).parent / path

    def get_model_filenames(self, folder_path, cache=None, isLora=False):
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"{folder_path} is not a valid directory.")
        filenames = []
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:
                if isLora:
                    txtcheck = path.with_suffix(".txt")
                    if txtcheck.exists():
                        fstats = txtcheck.stat()
                        if fstats.st_size > 0:
                            path = path.with_suffix(f"{path.suffix}")
                filenames.append(str(path.relative_to(folder_path)))
        # Return a sorted list, prepend names with 0 if they are in a folder or 1
        # if it is a plain file. This will sort folders above files in the dropdown
        return sorted(
            filenames,
            key=lambda x: (
                f"0{x.casefold()}"
                if not str(Path(x).parent) == "."
                else f"1{x.casefold()}"
            ),
        )

    def get_presets(self):
        folder_path = Path(self.paths['path_presets'])
        if not folder_path.is_dir():
            raise ValueError(f"{folder_path} is not a valid directory.")
        filenames = []
        for path in folder_path.rglob("*.png"):
            filenames.append(path)
        # Return a sorted list, prepend names with 0 if they are in a folder or 1
        presets = sorted(
            filenames,
            key=lambda x: (
                f"0{str(x).casefold()}"
                if not str(x.parent) == "."
                else f"1{str(x).casefold()}"
            ),
        )
        return map(
            lambda x: (x, str(Path(x).with_suffix('').name)), presets
        )


    def get_diffusers_filenames(self, folder_path, cache=None, isLora=False):
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"{folder_path} is not a valid directory.")
        filenames = []
        for path in folder_path.glob("*/*"):
            filenames.append(f"🤗:{path.relative_to(folder_path)}")
        return sorted(
            filenames,
            key=lambda x: (
                f"0{x.casefold()}"
                if not str(Path(x).parent) == "."
                else f"1{x.casefold()}"
            ),
        )

    def get_file_path(self, file_key, default=None):
        """
        Get the path for a file, downloading it if it doesn't exist.
        """
        if file_key not in self.DOWNLOADABLE_FILES:
            return default

        file_info = self.DOWNLOADABLE_FILES[file_key]
        folder = self.paths[file_info["path"]]
        if isinstance(folder, list): # folder might be a list of folders
            folder = folder[0] # ...select the first one
        file_path = (
            self.get_abspath(folder) / file_info["filename"]
        )

        if not file_path.exists():
            self.download_file(file_key)

        return file_path

    def get_folder_file_path(self, folder, filename, default=None):
        return self.get_file_path(f"{str(folder)}/{str(filename)}", default=default)

    def get_folder_list(self, folder):
        result = []
        for file in self.DOWNLOADABLE_FILES:
            if file.startswith(f"{folder}/"):
                result.append(self.DOWNLOADABLE_FILES[file]["filename"])
        # FIXME: also list files already in folder
        return result


    def download_file(self, file_key):
        """
        Download a file if it doesn't exist.
        """
        file_info = self.DOWNLOADABLE_FILES[file_key]
        folder = self.paths[file_info["path"]]
        if isinstance(folder, list): # folder might be a list of folders
            folder = folder[0] # ...select the first one
        file_path = (
            self.get_abspath(folder) / file_info["filename"]
        )

        print(f"Downloading {file_info['url']}...")
        response = requests.get(file_info["url"], stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(file_path, "wb") as file, tqdm(
            desc=file_info["filename"],
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

        print(f"Downloaded {file_info['filename']} to {file_path}")

    def find_lcm_lora(self):
        return self.get_file_path("lcm_lora")
