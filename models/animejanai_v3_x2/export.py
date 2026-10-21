"""AnimeJaNai HD V3 UltraCompact, by the-database, CC-BY-NC-SA-4.0.

Official release archive and selected checkpoint are both SHA256 verified.
Changes: export the original native x2 SRVGG model to static ONNX for A733.
https://github.com/the-database/mpv-AnimeJaNai/releases/tag/3.0.0
"""
import hashlib
import pathlib
import runpy
import sys
import urllib.request
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
ARCHIVE = HERE / "2x_AnimeJaNai_HD_V3_ModelsOnly.zip"
ARCHIVE_SHA = "50eefb2b657a19934f33ca002ab8f701b5c960566bd54b23f658bdeb19d7788f"
NAME = "2x_AnimeJaNai_HD_V3_UltraCompact.pth"
WEIGHT_SHA = "7c74ae2f919a31b96a08afc7dee37f1d1bae8b80844b7cc24ee82d43efd64110"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if not ARCHIVE.exists() or sha(ARCHIVE) != ARCHIVE_SHA:
    urllib.request.urlretrieve(
        "https://github.com/the-database/mpv-AnimeJaNai/releases/download/"
        "3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip", str(ARCHIVE))
if sha(ARCHIVE) != ARCHIVE_SHA:
    sys.exit("AnimeJaNai release archive SHA256 mismatch")
with zipfile.ZipFile(str(ARCHIVE)) as archive:
    # Read exactly one known member; never extract arbitrary archive paths.
    weights = archive.read("2x_AnimeJaNai_HD_V3_ModelsOnly/" + NAME)
if hashlib.sha256(weights).hexdigest() != WEIGHT_SHA:
    sys.exit("AnimeJaNai checkpoint SHA256 mismatch")
(HERE / NAME).write_bytes(weights)
sys.argv += ["--weights", str(HERE / NAME), "--weights-sha256", WEIGHT_SHA,
             "--weights-size", "2456096", "--num-conv", "8", "--scale", "2"]
runpy.run_path(str(HERE.parent / "animevideov3" / "export.py"), run_name="__main__")
