import os
import importlib
import importlib.util
import shutil
import subprocess
import sys
import re
import logging
from pathlib import Path
import importlib.metadata
import packaging.version

logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
logging.getLogger("xformers").addFilter(
    lambda record: "A matching Triton is not available" not in record.getMessage()
)

python = sys.executable
default_command_live = os.environ.get("LAUNCH_LIVE_OUTPUT") == "1"
index_url = os.environ.get("INDEX_URL", "")

modules_path = Path(__file__).resolve().parent
script_path = modules_path.parent
dir_repos = "repositories"

re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def _pip_backend_missing_scikit_build_core(error_text: str) -> bool:
    # Common pip/pyproject-hooks failure patterns seen in issue #289 and similar reports.
    if "pyproject_hooks" not in error_text and "BackendUnavailable" not in error_text:
        return False

    return (
        "Cannot import 'scikit_build_core.build'" in error_text
        or "No module named 'scikit_build_core'" in error_text
        or "scikit_build_core.build" in error_text
    )


def git_clone(url, dir, name, hash=None):
    try:
        import pygit2
        pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

        try:
            repo = pygit2.Repository(dir)
        except:
            Path(dir).parent.mkdir(exist_ok=True)
            repo = pygit2.clone_repository(url, str(dir))
            print(f"{name} cloned.")

        remote = repo.remotes["origin"]
        remote.fetch()

        commit = repo.get(hash)

        repo.checkout_tree(commit, strategy=pygit2.GIT_CHECKOUT_FORCE)
        print(f"{name} update check complete.")
    except Exception as e:
        print(f"Git clone failed for {name}: {str(e)}")


def repo_dir(name):
    return str(Path(script_path) / dir_repos / name)


def is_installed(package):
    m = re.match(re_requirement, package)
    if m is None:
        return False

    package = m.group(1).strip()
    version_required = (m.group(2) or "").strip()

    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    try:
        version_installed = re.sub(
            r"\+.*$",
            "",
            importlib.metadata.version(package)
        )
    except Exception:
        return False

    if version_required != "" and packaging.version.parse(version_required) != packaging.version.parse(version_installed):
        return False

    return True


def run(
    command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live
) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": "utf8",
        "errors": "ignore",
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""


def run_pip(command, desc=None, live=default_command_live, _retry_scikit_build_core: bool = True):
    index_url_line = f" --index-url {index_url}" if index_url != "" else ""
    try:
        return run(
            f'"{python}" -m pip {command} --prefer-binary{index_url_line}',
            desc=f"Installing {desc}",
            errdesc=f"Couldn't install {desc}",
            live=live,
        )
    except RuntimeError as e:
        if _retry_scikit_build_core and _pip_backend_missing_scikit_build_core(str(e)):
            print(
                "Detected missing scikit-build-core build backend. "
                "Installing scikit-build-core and retrying..."
            )
            run(
                f'"{python}" -m pip install scikit-build-core --prefer-binary{index_url_line}',
                desc="Installing scikit-build-core (build backend)",
                errdesc="Couldn't install scikit-build-core",
                live=live,
            )
            retry_command = command
            if "--no-build-isolation" not in command:
                retry_command = f"{command} --no-build-isolation"
            return run_pip(
                retry_command,
                desc=desc,
                live=live,
                _retry_scikit_build_core=False,
            )
        raise

def pip_rm(pkgs, desc=None, live=default_command_live):
    return run(
        f'"{python}" -m pip uninstall -y {pkgs}',
        desc=f"Uninstalling {desc}",
        errdesc=f"Couldn't uninstall {desc}",
        live=live,
    )

def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all rerqirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    """

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            if line.strip() == "" or line.startswith("--"):
                continue

            m = re.match(re_requirement, line)
            if m is None:
                return False

            package = m.group(1).strip()
            version_required = (m.group(2) or "").strip()

            try:
                version_installed = re.sub(
                    r"\+.*$",
                    "",
                    importlib.metadata.version(package)
                )
            except Exception:
                return False

            if version_required == "":
                continue

            if packaging.version.parse(version_required) != packaging.version.parse(
                version_installed
            ):
                return False

    return True
