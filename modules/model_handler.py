import requests
import hashlib
import shutil
import os
import cv2
import json
import threading
import time
from pathlib import Path
import numpy as np

from shared import translate as t

class Models:
    civit_workers = []

    def civit_update_worker(self, model_type, folder_paths):
        from shared import path_manager

        try:
            import imageio.v3
        except:
            # Skip updates if we are missing imageio
            print(f"Can't find imageio.v3 module: Skip CivitAI update")
            return
        if str(model_type) in self.civit_workers:
            # Already working on this folder
            print(t("Skip CivitAI check. Update for {type} already running.", mapping={'type': model_type}))
            return
        if not Path(self.cache_paths[model_type]).is_dir():
            print(t("WARNING: Can't find {path} Will not update thumbnails.", mapping={'path': self.cache_paths[model_type]}))
            return

        self.civit_workers.append(str(model_type))
        self.ready[model_type] = False
        updated = 0

        # Quick list
        self.names[model_type] = []
        for folder in folder_paths:
            for path in folder.rglob("*"):
                if path.suffix.lower() in self.EXTENSIONS:
                    # Add to model names
                    self.names[model_type].append(str(path.relative_to(folder)))

        # Return a sorted list, prepend names with 0 if they are in a folder or 1
        # if it is a plain file. This will sort folders above files in the dropdown
        self.names[model_type] = sorted(
            self.names[model_type],
            key=lambda x: (
                f"0{x.casefold()}"
                if not str(Path(x).parent) == "."
                else f"1{x.casefold()}"
            ),
        )
        self.ready[model_type] = True

        if self.offline:
            self.civit_workers.remove(str(model_type))
            return

        if model_type == "inbox" and self.names["inbox"]:
            checkpoints = path_manager.model_paths["modelfile_path"]
            checkpoints = checkpoints[0] if isinstance(checkpoints, list) else checkpoints
            loras = path_manager.model_paths["lorafile_path"]
            loras = loras[0] if isinstance(loras, list) else loras
            folders = {
                "DoRA": (loras, self.cache_paths["loras"]),
                "LoCon": (loras, self.cache_paths["loras"]),
                "LORA": (loras, self.cache_paths["loras"]),
                "Checkpoint": (checkpoints, self.cache_paths["checkpoints"]),
            }

        # Go through and check previews
        for folder in folder_paths:
            for path in folder.rglob("*"):
                if path.suffix.lower() in self.EXTENSIONS:
                    # get file name, add cache path change suffix
                    cache_file = Path(self.cache_paths[model_type] / path.name)
                    model_data = self.get_models_by_path(model_type, str(path))
                    if model_data is None:
                        break

                    suffixes = [".jpeg", ".jpg", ".png", ".gif"]
                    has_preview = False
                    for suffix in suffixes:
                        thumbcheck = cache_file.with_suffix(suffix)
                        if Path(thumbcheck).is_file():
                            has_preview = True
                            break

                    if not has_preview:
                        self.get_image(model_data, thumbcheck)
                        updated += 1
                        time.sleep(1)

                    txtcheck = cache_file.with_suffix(".txt")
                    if model_type == "loras" and not txtcheck.exists():
                        print(
                            t(
                                "Get LoRA keywords for {name} ({base} - {type})",
                                mapping= {
                                    'name': Path(path).name,
                                    'base': self.get_model_base(model_data),
                                    'type': self.get_model_type(model_data)
                                }
                            )
                        )
                        keywords = self.get_keywords(model_data)
                        with open(txtcheck, "w") as f:
                            f.write(", ".join(keywords))
                        updated += 1

                    if model_type == "inbox":
                        name = str(path.relative_to(folder_paths[0])) # FIXME handle if inbox is a list
                        filename =  self.get_file_from_name("inbox", name)
                        baseModel = self.get_model_base(model_data)
                        folder, cache = folders.get(self.get_model_type(model_data), [None, None])
                        if folder is None or baseModel is None:
                            print(t('Skipping {name} not sure what {type} is.', mapping={'name': str(name), 'type': self.get_model_type(model_data)}))
                            updated += 1
                            continue
                        # Move model to correct folder
                        dest = Path(folder) / baseModel
                        if not dest.exists():
                            dest.mkdir(parents=True, exist_ok=True)
                        cache_file = Path(self.cache_paths[model_type] / name)
                        try:
                            hash_sha256 = model_data["files"][0]["hashes"]["SHA256"]
                        except:
                            print(f"ERROR: {name} doesn't have a sha256 hash")
                            hash_sha256 = ""

                        move = False
                        rename = False
                        if cache_file.with_suffix(".json").exists():
                            with open(cache_file.with_suffix(".json")) as old_json:
                                old_data = json.load(old_json)
                            if hash_sha256 == old_data["files"][0]["hashes"]["SHA256"]:
                                if Path(dest / name).exists():
                                    print(f"WARNING: {name} in Inbox already exists. Ignoring.")
                                else:
                                    move = True
                            else:
                                print(f"WARNING: Renaming to {name}-{sha256} to avoid overwriting old model.")
                                rename = True
                                move = True

                        if move:
                            if rename:
                                destname = Path(f"{name.stem}-{hash_sha256}{name.suffix}")
                            else:
                                destname = name
                            shutil.move(Path(filename), Path(dest) / destname)
                            # Move cache-files
                            suffixes = [".json", ".txt", ".jpeg", ".jpg", ".png", ".gif"]
                            for suffix in suffixes:
                                cachefile = cache_file.with_suffix(suffix)
                                if rename:
                                    destfile = Path(f"{cache_file.stem}-{hash_sha256}{cache_file.suffix}")
                                else:
                                    destfile = cachefile
                                if cachefile.is_file():
                                    shutil.move(cachefile, Path(cache) / destfile.name)
                            print(t("Moved {name} to {dest}", mapping={'name': name, 'dest': dest}))
                            updated += 1

                    else:
                        # If this isn't the inbox, store some info about the "live" model
                        try:
                            self.model_hash[model_type][model_data["files"][0]["hashes"]["SHA256"]] = str(path)
                        except:
                            # Some models seem to not have sha256 hashes, just ignore those.
                            pass

        if updated > 0:
            print(t("CivitAI update for {type} done.", mapping={'type': model_type}))
        self.civit_workers.remove(str(model_type))

    def get_names(self, model_type):
        while not self.ready[model_type]:
            # Wait until we have read all the filenames
            time.sleep(0.2)
        return self.names[model_type]

    def get_file(self, model_type, name):
        # Search the folders for the model
        try:
            for folder in self.model_dirs[model_type]:
                file = Path(folder) / name
                if file.is_file():
                    return file
        except:
            pass
        return None

    def update_all_models(self):
        for model_type in ["checkpoints", "loras", "inbox"]:
            threading.Thread(
                target=self.civit_update_worker,
                args=(
                    model_type,
                    self.model_dirs[model_type],
                ),
                daemon=True,
            ).start()



    def __init__(self, offline=False):
        from shared import path_manager, settings

        self.offline = offline

        self.ready = {
            "checkpoints": False,
            "loras": False,
            "inbox": False,
        }
        self.names = {
            "checkpoints": [],
            "loras": [],
            "inbox": [],
        }
        self.model_hash = {
            "checkpoints": {},
            "loras": {},
            "inbox": {},
        }
        checkpoints = path_manager.model_paths["modelfile_path"]
        checkpoints = checkpoints if isinstance(checkpoints, list) else [checkpoints]
        loras = path_manager.model_paths["lorafile_path"]
        loras = loras if isinstance(loras, list) else [loras]
        inbox = path_manager.model_paths["inbox_path"]
        inbox = inbox if isinstance(inbox, list) else [inbox]
        self.model_dirs = {
            "checkpoints": checkpoints,
            "loras": loras,
            "inbox": inbox,
        }
        self.cache_paths = {
            "checkpoints": Path(path_manager.model_paths["cache_path"] / "checkpoints"),
            "loras": Path(path_manager.model_paths["cache_path"] / "loras"),
            "inbox": Path(path_manager.model_paths["cache_path"] / "inbox"),
        }

        self.base_url = "https://civitai.com/api/v1/"
        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        self.EXTENSIONS = [".pth", ".ckpt", ".bin", ".safetensors", ".gguf"]

        self.update_all_models()

    def get_file_from_hash(self, model_type, hash):
        return self.model_hash.get(model_type, {}).get(hash, None)

    def get_file_from_name(self, model_type, model_name):
        for folder in self.model_dirs[model_type]:
            path = Path(folder) / model_name
            if path.is_file():
                return path
        return None

    def model_sha256(self, filename):
        print(t("Hashing {filename}", mapping={'filename': filename}))
        blksize = 1024 * 1024
        hash_sha256 = hashlib.sha256()
        try:
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(blksize), b""):
                    hash_sha256.update(chunk)
            f.close()
            ret = hash_sha256.hexdigest().upper()
        except Exception as e:
            print(f"model_sha256(): Failed reading {filename}")
            print(f"Error: {e}")
            ret = None
        return ret

    def search_civitai_with_hash(self, hash):
        url = f"{self.base_url}model-versions/by-hash/{hash}"
        data = None
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code in [404, 451]:
                print(
                    t("Warning: Could not get {name} from civit.ai ({code})",
                    mapping={'name': hash, 'code': response.status_code})
                )
                # Create our own data
                data = {
                    "files": [
                        {
                            "hashes": {
                                "SHA256": hash,
                            }
                        }
                    ]
                }
            elif response.status_code == 503:
                print("Error: Civit.ai Service Currently Unavailable")
            else:
                print(f"HTTP Error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        return data

    def get_models_by_path(self, model_type, path):
        data = None
        cache_path = Path(self.cache_paths[model_type]) / Path(Path(path).name)
        if cache_path.is_dir():
            # Give up
            return {}
        json_path = Path(cache_path).with_suffix(".json")

        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except:
                data = None
        if data is not None:
            return data

        if Path(path).suffix == ".merge":
            return {"baseModel": "Merge"}

        hash = self.model_sha256(path)
        data = self.search_civitai_with_hash(hash)

        if data is not None:
            print(t("Update model data: {path}", mapping={'path': json_path}))
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

        return data


    def get_model_path(self, model_type, name, hash=None, default=None):
        from shared import path_manager, settings

        # Look through folders for the filename
        filename = self.get_file(model_type, name)

        # Try looking for model using the hash
        if filename is None and hash is not None:
            filename = self.get_file_from_hash(model_type, hash)
            if filename is not None:
                print(f"INFO: Found {filename} from hash")

        # If we don't have a filename, get the default.
        if filename is None and default is not None:
            name = path_manager.get_folder_file_path(
                model_type,
                default,
            )
            filename = self.get_file(model_type, name)
            if filename is not None:
                print(f"INFO: Using {filename} (default)")

        if filename is None:
            print(f"Could not find file: {name}")
            if hash is not None and hash != "None":
                print(f"    SHA256: {hash}")
                data = self.search_civitai_with_hash(hash)
                if data is not None and data.get('modelId', None) is not None:
                    print(f"    Download link: https://civitai.com/models/{data.get('modelId')}")
                print(f"    Search here: https://civitaiarchive.com/sha256/{hash}")

        return filename


    def get_keywords(self, model):
        keywords = model.get("trainedWords", [""])
        return keywords

    def get_model_base(self, model):
        return model.get("baseModel", "Unknown")

    def get_model_type(self, model):
        res = model.get("model", None)
        if res is not None:
            res = res.get("type", "Unknown")
        else:
            res = "Unknown"
        return res

    def get_image(self, model, path):
        from shared import settings

        if "baseModel" in model and model["baseModel"] == "Merge":
            return

        import imageio.v3 as iio
        if "model_preview" in settings.default_settings:
            opts = settings.default_settings["model_preview"].split(",")
            if "caption" in opts:
                caption=True
            if "nogifzoom" in opts:
                nogifzoom=True
            if "zoom" in opts:
                zoom=True
        else:
            caption=False
            nogifzoom=False
            zoom=False

        def make_thumbnail(image, text, zoom=False, caption=False):
            max = 166  # Max width or height

            if image is None:
                return None

            if zoom:
                oh = image.shape[0]
                ow = image.shape[1]
                scale = max / oh if oh > ow else max / ow
                image = cv2.resize(
                    image,
                    dsize=(int(ow * scale), int(oh * scale)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            if caption:
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.35
                thickness = 1

                org = (3, 10)
                color = (25, 15, 11) # BGR
                image = cv2.putText(
                    image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness*2,
                    cv2.LINE_AA
                )
                org = (3, 10)
                color = (255, 215, 185) # BGR
                image = cv2.putText(
                    image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

            return image

        path = path.with_suffix(".jpeg")
        caption_text = f"{path.with_suffix('').name}"

        image_url = None
        for preview in model.get("images", [{}]):
            url = preview.get("url")
            format = preview.get("type")
            if url:
                print(t("Updating preview for {text}.", mapping={'text': caption_text}))
                image_url = url
                response = self.session.get(image_url)
                if response.status_code != 200:
                    print(f"WARNING: get_image() for {caption_text} - {response.status_code} : {response.reason}")
                    break
                image = np.asarray(bytearray(response.content), dtype="uint8") 
                out = make_thumbnail(cv2.imdecode(image, cv2.IMREAD_COLOR), caption_text, caption=caption, zoom=zoom)
                if out is not None:
                    out = cv2.imencode('.jpg', out)[1] 
                else:
                    out = response.content
                with open(path, "wb") as file:
                    file.write(out)

                fps = iio.immeta(path).get("fps", False)
                if format == "video" and fps:
                    tmp_path = f"{path}.tmp"
                    shutil.move(path, tmp_path)
                    video = iio.imiter(tmp_path)
                    video_out = []
                    for i in video:
                        out = make_thumbnail(i, caption_text, caption=caption, zoom=not nogifzoom)
                        if out is None:
                            out = i
                        video_out.append(out)
                    iio.imwrite(
                        str(path.with_suffix(".gif")), video_out, fps=fps, loop=0
                    )
                    os.remove(tmp_path)
                break
        if image_url is None:
            shutil.copyfile("html/warning.jpeg", path)
