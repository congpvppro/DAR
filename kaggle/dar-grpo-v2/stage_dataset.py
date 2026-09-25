"""Prepare the complete GRPO model checkpoint for a public Kaggle dataset."""

import json
import os
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "checkpoint-download/dar_grpo/checkpoints/v0-20260923-042158/checkpoint-4100"
DEST = HERE / "dataset"
INDEX = json.loads((SOURCE / "model.safetensors.index.json").read_text(encoding="utf-8"))
SHARDS = set(INDEX["weight_map"].values())
for name in SHARDS:
    if not (SOURCE / name).is_file() or (SOURCE / name).stat().st_size == 0:
        raise FileNotFoundError(f"Incomplete weight shard: {name}")
actual_size = sum((SOURCE / name).stat().st_size for name in SHARDS)
tensor_size = INDEX["metadata"]["total_size"]
if not tensor_size <= actual_size < tensor_size + 10 * 1024 * 1024:
    raise ValueError("Weight shard sizes do not match model.safetensors.index.json")
for name in ("config.json", "preprocessor_config.json", "trainer_state.json", "tokenizer.json"):
    if not (SOURCE / name).is_file():
        raise FileNotFoundError(name)

DEST.mkdir(parents=True, exist_ok=True)
for source in SOURCE.iterdir():
    if not source.is_file() or source.name.endswith(".part"):
        continue
    target = DEST / source.name
    if target.exists():
        target.unlink()
    if source.suffix == ".safetensors":
        os.link(source, target)
    else:
        shutil.copy2(source, target)

shutil.copy2(HERE / "QWEN-RESEARCH-LICENSE.txt", DEST / "QWEN-RESEARCH-LICENSE.txt")
(DEST / "NOTICE.txt").write_text(
    "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, "
    "Copyright (c) Alibaba Cloud. All Rights Reserved.\n"
    "This checkpoint modifies Qwen2.5-VL-3B-Instruct through DAR SFT and GRPO training.\n"
    "Built with Qwen. Original model: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct\n"
    "Source run: https://www.kaggle.com/code/vucongaaa/dar-grpo (checkpoint-4100).\n",
    encoding="utf-8",
)

metadata = {
    "title": "DAR GRPO checkpoint 4100",
    "subtitle": "Qwen2.5-VL-3B DAR GRPO model weights at step 4100",
    "id": "vuhuycong/dar-grpo-checkpoint-4100",
    "licenses": [{"name": "other"}],
    "description": (
        "Full model checkpoint from vucongaaa/dar-grpo version 1 at step 4100/13646. "
        "Fine-tuned from Qwen2.5-VL-3B-Instruct with DAR SFT and GRPO. "
        "Qwen weights are governed by the Qwen RESEARCH LICENSE AGREEMENT "
        "(non-commercial use); the complete license and required attribution are included. "
        "Optimizer and scheduler states were not saved, so this checkpoint supports "
        "model-weight continuation but not exact trainer resume."
    ),
}
(DEST / "dataset-metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("Staged", len(list(DEST.iterdir())), "files at", DEST)
