import argparse
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description="Download the fixed training RGB videos used by ActWeave.")
parser.add_argument("--destination", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
destination = args.destination or root / "assets/demonstrations"
manifest = json.loads((root / "configs/demonstrations/official-training.json").read_text())
base = f"https://huggingface.co/datasets/{manifest['repository']}/resolve/{manifest['revision']}"
for task in manifest["tasks"]:
    for episode in task["episodes"]:
        index = episode["episode_index"]
        folder = destination / f"{task['source_task']}-{index:07d}"
        folder.mkdir(parents=True, exist_ok=True)
        for camera in manifest["cameras"]:
            output = folder / f"{camera}.mp4"
            if output.is_file():
                continue
            url = f"{base}/data/RoboDojo/{task['source_task']}/arx_x5/preview_video/episode_{index:07d}_{camera}.mp4"
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                print(str(output), flush=True)
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                "--connect-timeout", "10", "--max-time", "300",
                                "--output", str(temporary), url], check=True, timeout=305)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
