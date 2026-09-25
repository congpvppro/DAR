"""Build the second Kaggle run from the published v1 notebook."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASET = "vuhuycong/dar-grpo-checkpoint-4100"
SOURCE = json.loads((HERE / "pull.json").read_text(encoding="utf-8"))
notebook = json.loads(SOURCE["blob"]["source"])


def replace_cell(index: int, old: str, new: str) -> None:
    cell = notebook["cells"][index]
    text = "".join(cell["source"])
    if old not in text:
        raise ValueError(f"Cell {index} does not contain expected text: {old!r}")
    cell["source"] = text.replace(old, new).splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


replace_cell(
    1,
    "  -print0 | sort -z)",
    "  ! -name 'transformers-4.46.3*' " + chr(92) + "\n  -print0 | sort -z)",
)


replace_cell(
    4,
    '# EDIT THIS PATH after attaching the SFT checkpoint as a Kaggle input.\n'
    'SFT_MODEL_SOURCE = Path(\n'
    '    "/kaggle/input/models/vucongaaa/sft/pytorch/default/1/checkpoint-214"\n'
    ')\n'
    'MODEL_ALIAS = Path("/kaggle/working/dar_sft_model")',
    '# GRPO checkpoint-4100 from vucongaaa/dar-grpo v1, published as a dataset.\n'
    'SFT_MODEL_SOURCE = Path(\n'
    '    "/kaggle/input/datasets/vuhuycong/dar-grpo-checkpoint-4100"\n'
    ')\n'
    'MODEL_ALIAS = Path("/kaggle/working/dar_grpo_resume_model")',
)
replace_cell(4, "Edit SFT_MODEL_SOURCE: checkpoint directory not found", "GRPO checkpoint dataset is not attached")
replace_cell(4, "Missing preprocessor_config.json under {SFT_MODEL_SOURCE}; publish the complete SFT checkpoint", "Missing preprocessor_config.json under {SFT_MODEL_SOURCE}; publish the complete GRPO checkpoint")
replace_cell(4, 'print("SFT checkpoint:", SFT_MODEL_SOURCE)', 'print("GRPO checkpoint:", SFT_MODEL_SOURCE)')
replace_cell(4, "SFT_MODEL_SOURCE", "GRPO_MODEL_SOURCE")
replace_cell(4, "from pathlib import Path\nimport os", "from pathlib import Path\nimport json\nimport os")
replace_cell(
    4,
    'GRPO_MODEL_SOURCE = Path(\n'
    '    "/kaggle/input/datasets/vuhuycong/dar-grpo-checkpoint-4100"\n'
    ')\n'
    'MODEL_ALIAS = Path("/kaggle/working/dar_grpo_resume_model")',
    'INPUT_ROOT = Path("/kaggle/input")\n'
    'DATASET_SLUG = "dar-grpo-checkpoint-4100"\n'
    'candidates = [\n'
    '    INPUT_ROOT / DATASET_SLUG,\n'
    '    INPUT_ROOT / "datasets" / "vuhuycong" / DATASET_SLUG,\n'
    ']\n'
    'GRPO_MODEL_SOURCE = next(\n'
    '    (p for p in candidates if (p / "model.safetensors.index.json").is_file()),\n'
    '    None,\n'
    ')\n'
    'if GRPO_MODEL_SOURCE is None:\n'
    '    matches = [\n'
    '        p.parent for p in INPUT_ROOT.rglob("model.safetensors.index.json")\n'
    '        if DATASET_SLUG in p.parts\n'
    '    ]\n'
    '    if len(matches) != 1:\n'
    '        mounts = sorted(p.as_posix() for p in INPUT_ROOT.iterdir())\n'
    '        raise FileNotFoundError(\n'
    '            f"Could not locate {DATASET_SLUG}; matches={matches}; inputs={mounts}"\n'
    '        )\n'
    '    GRPO_MODEL_SOURCE = matches[0]\n'
    'MODEL_ALIAS = Path("/kaggle/working/dar_grpo_resume_model")',
)
replace_cell(
    4,
    'if not weight_files:\n    raise FileNotFoundError(f"No *.safetensors weights under {GRPO_MODEL_SOURCE}")',
    'if not weight_files:\n    raise FileNotFoundError(f"No *.safetensors weights under {GRPO_MODEL_SOURCE}")\n'
    'index = json.loads((GRPO_MODEL_SOURCE / "model.safetensors.index.json").read_text())\n'
    'missing_shards = set(index["weight_map"].values()) - {p.name for p in weight_files}\n'
    'if missing_shards:\n    raise FileNotFoundError(f"Missing GRPO weight shards: {sorted(missing_shards)}")',
)
replace_cell(5, 'model_path = "/kaggle/working/dar_sft_model"', 'model_path = "/kaggle/working/dar_grpo_resume_model"')
replace_cell(9, 'MODEL_NAME=/kaggle/working/dar_sft_model', 'MODEL_NAME=/kaggle/working/dar_grpo_resume_model')
replace_cell(9, '--num_train_epochs 1 --max_completion_length 2400', '--max_steps 9546 --max_completion_length 2400')
replace_cell(9, 'MODE=full  # Change to full only after the smoke test succeeds.', 'MODE=smoke  # Verify checkpoint resume before any full run.')
replace_cell(
    9,
    'EXTRA_ARGS+=(--max_steps 2 --max_completion_length 512 --save_strategy no --overwrite_output_dir true)',
    'EXTRA_ARGS+=(--resume_from_checkpoint "$MODEL_NAME" --max_steps 4102 '
    '--max_completion_length 512 --save_strategy no --overwrite_output_dir true)',
)
replace_cell(
    9,
    'EXTRA_ARGS+=(--max_steps 9546 --max_completion_length 2400',
    'EXTRA_ARGS+=(--resume_from_checkpoint "$MODEL_NAME" --max_steps 13646 --max_completion_length 2400',
)
replace_cell(
    9,
    'echo "Training mode: $MODE"',
    'python - <<\'PY\'\n'
    'import json\n'
    'from pathlib import Path\n'
    'source = Path("/kaggle/working/dar_grpo_resume_model")\n'
    'state = json.loads((source / "trainer_state.json").read_text())\n'
    'assert state["global_step"] == 4100, state["global_step"]\n'
    'print(f"RESUME_SOURCE={source.resolve()} GLOBAL_STEP={state[\'global_step\']}")\n'
    'PY\n'
    'echo "Training mode: $MODE"',
)

notebook["cells"].insert(
    0,
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# DAR-GRPO — continuation from checkpoint-4100\n",
            "\n",
            "This run starts from the GRPO model weights produced by `vucongaaa/dar-grpo` version 1. "
            "That run used `save_only_model=true`, so optimizer and scheduler states are unavailable. "
            "Version 1 stopped at step 4100 of 13646. This version tests two optimizer steps with "
            "`--resume_from_checkpoint` and checks whether the first new log step is 4101. "
            "Optimizer, scheduler, and RNG files are absent, so exact numerical continuation "
            "is impossible even if Trainer restores the step counter.\n",
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
            "root = Path('/kaggle/working/dar_grpo/smoke')\n",
            "files = list(root.rglob('logging.jsonl'))\n",
            "print('Smoke log files:', [str(p) for p in files])\n",
            "assert files, 'Training did not produce logging.jsonl'\n",
            "records = [json.loads(line) for p in files for line in p.read_text().splitlines() if line.strip()]\n",
            "observed = sorted({int(record['global_step/max_steps'].split('/')[0]) for record in records if 'global_step/max_steps' in record})\n",
            "print('Observed training steps:', observed)\n",
            "assert 4101 in observed and 4102 in observed, f'Checkpoint did not resume at 4101: {observed}'\n",
            "result = {'source_step': 4100, 'observed_steps': observed, 'passed': True}\n",
            "Path('/kaggle/working/resume_smoke_result.json').write_text(json.dumps(result, indent=2))\n",
            "print('RESUME_SMOKE_PASSED', result)\n",
        ],
    }
)

(HERE / "dar-grpo.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)

source_md = SOURCE["metadata"]
metadata = {
    "id": source_md["ref"],
    "id_no": source_md["id"],
    "title": source_md["title"],
    "code_file": "dar-grpo.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": False,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": source_md["datasetDataSources"] + [DATASET],
    "kernel_sources": source_md["kernelDataSources"],
    "competition_sources": source_md["competitionDataSources"],
    "model_sources": source_md["modelDataSources"],
    "docker_image": source_md["dockerImage"],
    "machine_shape": source_md["machineShape"],
}
(HERE / "kernel-metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
