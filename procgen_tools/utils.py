import sys
import os
from pathlib import Path
import requests
import gzip
import tarfile
from tqdm import tqdm


def _extract_tgz(path):
    "Extracts a .tar.gz or .tgz file."
    assert path.endswith(".tgz") or path.endswith(".tar.gz")
    with tarfile.open(path, "r") as tar:
        members = tar.getmembers()
        t = tqdm(members)
        t.set_description(f"Extracting {path}")
        for member in t:
            tar.extract(member)


def _fetch(url, filepath: str = None, force: bool = False):
    "Fetches a file from a URL and extracts it."
    filepath = filepath or url.split("/")[-1]
    if not force and Path(filepath).exists():
        print(f"Already downloaded {url}")
        return

    # create dir for filepath if not exists
    if not Path(filepath).parent.exists():
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # download file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        t = tqdm(total=total_size, unit="iB", unit_scale=True)
        t.set_description(f"Downloading {url}")
        with open(filepath, "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            raise ValueError(
                f"Error: downloaded {t.n} bytes, expected {total_size} bytes."
            )

    # extract file (if needed)
    if filepath.endswith(".tgz") or filepath.endswith(".tar.gz"):
        _extract_tgz(filepath)


def cd_into_procgen_tools():
    """Go up until we're in the procgen-tools directory. Assumes we're in a subdirectory of procgen-tools."""
    original_path = Path.cwd()
    # Assert procgen-tools is a parent directory
    while Path.cwd().name != "procgen-tools":
        if Path.cwd().parent == Path.cwd():  # we're at the root
            os.chdir(original_path)
            raise Exception("Could not find procgen-tools directory")
        os.chdir("..")


def setup_dir():
    """
    Get into the procgen-tools directory, create it if it doesn't exist.
    """
    try:
        cd_into_procgen_tools()
    except Exception:
        Path("procgen-tools").mkdir(parents=True, exist_ok=True)
        os.chdir("procgen-tools")


def setup(
    force: bool = False,
    dl_data: bool = False,
    dl_episode_data: bool = False,
    dl_patch_data: bool = False,
    dl_stats: bool = False,
):
    """
    cd into the procgen-tools directory then download and extract data files.
    """
    setup_dir()
    assert Path.cwd().name == "procgen-tools", "must be in procgen-tools"

    # Check that data.tgz is newer than 2023-03-04
    # (this is the date of the last data update, previous data was bugged)
    force_redownload_vfields = False
    data_tgz = Path("data.tgz")
    if data_tgz.exists() and data_tgz.stat().st_mtime < 1677968658:
        force_redownload_vfields = True

    if dl_episode_data:
        _fetch("https://nerdsniper.net/mats/episode_data.tgz", force=force)
    if dl_patch_data:
        _fetch("https://nerdsniper.net/mats/patch_data.tgz", force=force)
    if dl_data:
        _fetch(
            "https://nerdsniper.net/mats/data.tgz",
            force=force or force_redownload_vfields,
        )
    if dl_stats:
        _fetch("https://nerdsniper.net/mats/episode_stats_data.tgz", force=force)

    _fetch(
        "https://nerdsniper.net/mats/model_rand_region_5.pth",
        "trained_models/maze_I/model_rand_region_5.pth",
        force=force,
    )
