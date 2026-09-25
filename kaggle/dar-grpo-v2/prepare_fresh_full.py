"""Build a fresh GRPO phase from the SFT model with a complete checkpoint."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "pull.json").read_text(encoding="utf-8"))
notebook = json.loads(source["blob"]["source"])


def replace(index: int, old: str, new: str) -> None:
    cell = notebook["cells"][index]
    body = "".join(cell["source"])
    if old not in body:
        raise ValueError(f"cell {index}: missing {old!r}")
    cell["source"] = body.replace(old, new).splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


# Keep all transient dependencies outside Kaggle's persistent 20 GiB output.
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    body = "".join(cell["source"])
    for old, new in (
        ("/kaggle/working/dar_grpo_python", "/kaggle/tmp/dar_grpo_python"),
        ("/kaggle/working/DAR", "/kaggle/tmp/DAR"),
        ("/kaggle/working/dar_sft_model", "/kaggle/tmp/dar_sft_model"),
        ("/kaggle/working/dar_grpo/train_qwen25vl_ms_grpo.jsonl", "/kaggle/tmp/dar_grpo/train_qwen25vl_ms_grpo.jsonl"),
    ):
        body = body.replace(old, new)
    cell["source"] = body.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None

replace(
    1,
    "  -print0 | sort -z)",
    "  ! -name 'transformers-4.46.3*' " + chr(92) + "\n  -print0 | sort -z)",
)
replace(
    9,
    "MODE=full  # Change to full only after the smoke test succeeds.",
    "MODE=full  # Fresh GRPO from the SFT checkpoint; save complete training state.",
)
replace(6, "mkdir -p /kaggle/working/dar_grpo", "mkdir -p /kaggle/tmp/dar_grpo /kaggle/working/dar_grpo")
replace(
    9,
    "--num_train_epochs 1 --max_completion_length 2400",
    "--max_steps 13646 --max_completion_length 2400",
)
replace(9, "--save_steps 100 --save_only_model true --save_total_limit 1", "--save_steps 1000 --save_only_model false --save_total_limit 1")
replace(
    9,
    '  --external_plugins "$PLUGIN_FILE" \\\n',
    '  --external_plugins "$PLUGIN_FILE" /kaggle/tmp/dar_grpo_phase_stop.py \\\n',
)

# The 13,646-step schedule must be fixed from step zero. A callback ends this
# Kaggle version at step 1000 after writing its full checkpoint.
notebook["cells"].insert(
    9,
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "callback = Path('/kaggle/tmp/dar_grpo_phase_stop.py')\n",
            "callback.write_text('''from transformers import TrainerCallback\n",
            "from swift.plugin import extra_callbacks\n",
            "\n",
            "class StopAtFullCheckpoint(TrainerCallback):\n",
            "    def on_step_end(self, args, state, control, **kwargs):\n",
            "        if state.global_step >= 1000:\n",
            "            control.should_save = True\n",
            "            control.should_training_stop = True\n",
            "        return control\n",
            "\n",
            "extra_callbacks.append(StopAtFullCheckpoint())\n",
            "''')\n",
            "print('Full-state phase stop callback:', callback)\n",
        ],
    },
)

notebook["cells"].append(
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "root = Path('/kaggle/working/dar_grpo/checkpoints')\n",
            "checkpoints = list(root.rglob('checkpoint-1000'))\n",
            "assert len(checkpoints) == 1, checkpoints\n",
            "checkpoint = checkpoints[0]\n",
            "required = ['model.safetensors.index.json', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json']\n",
            "missing = [name for name in required if not (checkpoint / name).is_file()]\n",
            "rng = list(checkpoint.glob('rng_state*.pth'))\n",
            "assert not missing and rng, f'Missing training state: {missing}, RNG={rng}'\n",
            "state = json.loads((checkpoint / 'trainer_state.json').read_text())\n",
            "assert state['global_step'] == 1000, state['global_step']\n",
            "sizes = {p.name: p.stat().st_size for p in checkpoint.iterdir() if p.is_file()}\n",
            "print('FULL_CHECKPOINT_OK', checkpoint, 'step=', state['global_step'], 'files=', sizes)\n",
        ],
    }
)

notebook["cells"].insert(
    0,
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# DAR-GRPO: fresh run with complete checkpoints\n",
            "\n",
            "This run starts GRPO from the SFT checkpoint at step zero. It uses the original "
            "13,646-step learning-rate schedule and stops after saving a full checkpoint at "
            "step 1000. The checkpoint includes model weights, AdamW optimizer, scheduler, "
            "RNG state, and Trainer state. Temporary runtime files live in `/kaggle/tmp` to "
            "leave space for the checkpoint in `/kaggle/working`.\n",
        ],
    },
)

target = HERE / "fresh_full_kernel"
target.mkdir(exist_ok=True)
(target / "dar-grpo.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

metadata = {
    "id": source["metadata"]["ref"],
    "id_no": source["metadata"]["id"],
    "title": source["metadata"]["title"],
    "code_file": "dar-grpo.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": False,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": source["metadata"]["datasetDataSources"],
    "kernel_sources": source["metadata"]["kernelDataSources"],
    "competition_sources": source["metadata"]["competitionDataSources"],
    "model_sources": source["metadata"]["modelDataSources"],
    "docker_image": source["metadata"]["dockerImage"],
    "machine_shape": source["metadata"]["machineShape"],
}
(target / "kernel-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
