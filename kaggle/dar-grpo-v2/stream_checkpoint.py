"""Stream GRPO checkpoint files from Kaggle notebook output to disk."""

from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest


DEST = Path(__file__).resolve().parent / "checkpoint-download"
PREFIX = "dar_grpo/checkpoints/v0-20260923-042158/checkpoint-4100/"
api = KaggleApi()
api.authenticate()
request = ApiListKernelSessionOutputRequest()
request.user_name = "vucongaaa"
request.kernel_slug = "dar-grpo"
request.page_size = 200

files = []
with api.build_kaggle_client() as client:
    while True:
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        files.extend(item for item in response.files or [] if item.file_name.startswith(PREFIX))
        if files or not response.next_page_token:
            break
        request.page_token = response.next_page_token

if not files:
    raise RuntimeError(f"Checkpoint {PREFIX!r} was not found in notebook output")

for item in files:
    target = DEST / item.file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if target.is_file() and target.stat().st_size > 0:
        print("already present:", target.name, target.stat().st_size, flush=True)
        continue

    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(item.url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        if not append:
            offset = 0
        print("downloading:", target.name, "from", offset, flush=True)
        with part.open("ab" if append else "wb") as output:
            next_report = offset + 512 * 1024 * 1024
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    if output.tell() >= next_report:
                        print("progress:", target.name, output.tell(), flush=True)
                        next_report = output.tell() + 512 * 1024 * 1024
        expected = response.headers.get("Content-Length")
        if expected is not None and part.stat().st_size != offset + int(expected):
            raise RuntimeError(f"Incomplete download: {target.name}")
    part.replace(target)
    print("complete:", target.name, target.stat().st_size, flush=True)
