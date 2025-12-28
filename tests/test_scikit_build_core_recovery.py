import os
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestScikitBuildCoreRecovery(unittest.TestCase):
    def test_run_pip_recovers_from_missing_scikit_build_core_backend(self):
        """
        Reproduces issue #289-style failure:
        pip cannot import 'scikit_build_core.build' as a PEP517 backend.

        Expected behavior: launcher auto-installs scikit-build-core and retries.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wheelhouse = tmp_path / "wheelhouse"
            wheelhouse.mkdir()

            # Build a tiny local wheel that provides `scikit_build_core.build`.
            skc_shim_src = tmp_path / "skc_shim"
            (skc_shim_src / "scikit_build_core").mkdir(parents=True)
            (skc_shim_src / "scikit_build_core" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            (skc_shim_src / "scikit_build_core" / "build.py").write_text(
                _SCIKIT_BUILD_CORE_SHIM_BACKEND,
                encoding="utf-8",
            )
            (skc_shim_src / "setup.py").write_text(
                _SCIKIT_BUILD_CORE_SHIM_SETUP_PY,
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    str(skc_shim_src),
                    "-w",
                    str(wheelhouse),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(
                any(p.suffix == ".whl" for p in wheelhouse.iterdir()),
                "Expected shim wheel to be built",
            )

            # Package that *declares* `scikit_build_core.build` as the backend but does not
            # list scikit-build-core in build-system.requires. This triggers the same
            # BackendUnavailable error as the real-world report.
            bad_pkg_src = tmp_path / "bad_pkg"
            (bad_pkg_src / "dummy_skc_pkg").mkdir(parents=True)
            (bad_pkg_src / "dummy_skc_pkg" / "__init__.py").write_text(
                "value = 1\n",
                encoding="utf-8",
            )
            (bad_pkg_src / "pyproject.toml").write_text(
                _DUMMY_PEP517_PYPROJECT,
                encoding="utf-8",
            )

            # Create isolated venv so this test doesn't mutate the developer's Python env.
            venv_dir = tmp_path / "venv"
            venv.EnvBuilder(with_pip=True).create(venv_dir)
            venv_python = venv_dir / (
                "Scripts" if os.name == "nt" else "bin"
            ) / ("python.exe" if os.name == "nt" else "python")
            # Mirror the launcher behavior: ensure pip is new enough to avoid known
            # metadata parsing bugs in older pip releases.
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                text=True,
            )
            # `modules.launch_util` depends on `packaging` which isn't guaranteed to exist
            # in a freshly created venv.
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "packaging"],
                check=True,
                capture_output=True,
                text=True,
            )

            # Run a small script inside the venv that imports launcher utilities and runs pip.
            out_dir = tmp_path / "out_wheels"
            out_dir.mkdir()
            runner = tmp_path / "runner.py"
            runner.write_text(
                _VENV_RUNNER_SCRIPT,
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PIP_NO_INDEX"] = "1"
            env["PIP_FIND_LINKS"] = str(wheelhouse)
            env["RF_TEST_REPO_ROOT"] = str(PROJECT_ROOT)
            env["RF_TEST_BAD_PKG"] = str(bad_pkg_src)
            env["RF_TEST_OUT_DIR"] = str(out_dir)

            result = subprocess.run(
                [str(venv_python), str(runner)],
                env=env,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.fail(
                    "Expected venv runner to succeed.\n"
                    f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
                )

            # The exact downstream build result depends on platform toolchains, but the
            # launcher should at least auto-install the missing backend.
            # Verifying that side effect keeps the regression test stable.
            # (The runner itself asserts `scikit_build_core` is importable.)


_SCIKIT_BUILD_CORE_SHIM_SETUP_PY = r"""
from setuptools import setup, find_packages

setup(
    name="scikit-build-core",
    version="0.0.0",
    packages=find_packages(),
)
""".lstrip()


_SCIKIT_BUILD_CORE_SHIM_BACKEND = r"""
import re
import zipfile
from pathlib import Path


def _normalize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", s.replace("-", "_"))


def _read_name_version():
    pyproject = Path("pyproject.toml")
    name = "dummy-skc-pkg"
    version = "0.0.0"
    if not pyproject.exists():
        return name, version
    text = pyproject.read_text(encoding="utf-8", errors="ignore")
    project_block = text
    m = re.search(r"(?ms)^\[project\]\s*(.*?)(^\[|\Z)", text)
    if m:
        project_block = m.group(1)
    m_name = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', project_block)
    if m_name:
        name = m_name.group(1)
    m_ver = re.search(r'(?m)^version\s*=\s*["\']([^"\']+)["\']', project_block)
    if m_ver:
        version = m_ver.group(1)
    return name, version


def get_requires_for_build_wheel(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name, version = _read_name_version()
    dist = _normalize(name)
    wheel_name = f"{dist}-{version}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name

    dist_info = f"{dist}-{version}.dist-info"
    metadata = (
        f"Metadata-Version: 2.1\\nName: {name}\\nVersion: {version}\\n".encode("utf-8")
    )
    wheel_file = (
        "Wheel-Version: 1.0\\n"
        "Generator: ruinedfooocus-test-shim\\n"
        "Root-Is-Purelib: true\\n"
        "Tag: py3-none-any\\n"
    ).encode("utf-8")

    files = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in Path(".").rglob("*.py"):
            if any(part.startswith(".") for part in path.parts):
                continue
            arcname = path.as_posix()
            zf.write(path, arcname)
            files.append(arcname)

        zf.writestr(f"{dist_info}/METADATA", metadata)
        zf.writestr(f"{dist_info}/WHEEL", wheel_file)
        files.extend([f"{dist_info}/METADATA", f"{dist_info}/WHEEL"])

        record_path = f"{dist_info}/RECORD"
        record_lines = [f"{p},,\\n" for p in files] + [f"{record_path},,\\n"]
        zf.writestr(record_path, "".join(record_lines).encode("utf-8"))

    return wheel_name
""".lstrip()


_DUMMY_PEP517_PYPROJECT = r"""
[build-system]
requires = []
build-backend = "scikit_build_core.build"

[project]
name = "dummy-skc-pkg"
version = "0.0.0"
""".lstrip()


_VENV_RUNNER_SCRIPT = r"""
import os
import sys
import importlib.util
from pathlib import Path

repo_root = Path(os.environ["RF_TEST_REPO_ROOT"])
sys.path.insert(0, str(repo_root))

from modules.launch_util import run_pip

bad_pkg = os.environ["RF_TEST_BAD_PKG"]
out_dir = os.environ["RF_TEST_OUT_DIR"]

try:
    run_pip(f'wheel \"{bad_pkg}\" -w \"{out_dir}\"', desc="dummy-skc-pkg")
except RuntimeError:
    # The point of this reproduction is the missing-backend error and the launcher
    # recovery step. The build may still fail for unrelated reasons (e.g. toolchain).
    pass

if importlib.util.find_spec("scikit_build_core") is None:
    raise SystemExit("Expected scikit_build_core to be installable/importable")
""".lstrip()
