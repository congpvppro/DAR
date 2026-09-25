# DAR-STSG auxiliary SFT pilot

The adaptation is **training-data augmentation + cold-start multi-task SFT**.
Each arm starts independently from Qwen2.5-VL-3B-Instruct. The graph is an auxiliary
target, not an inference input or a new reward. Read the [Vietnamese study design](../../research/step-vot-dar-20260923/report.vi.md).

Use [dar_kaggle_stsg_pilot.ipynb](../../dar_kaggle_stsg_pilot.ipynb) for Kaggle.
It embeds these scripts and the current `test.py`, so an older attached DAR
dataset does not silently omit the new code. Configure the base model, annotation,
video directory and offline wheelhouse paths. No existing DAR checkpoint required.

## Current Kaggle workflow: VideoLLaMA3 teacher, one arm at a time

- Teacher: `DAMO-NLP-SG/VideoLLaMA3-7B` (video checkpoint, not `-Image`).
- Student: fresh Qwen2.5-VL-3B-Instruct full SFT (LLM + aligner trainable, vision frozen).
- Full-SFT settings follow the local DAR SFT launcher: BF16, LR=1e-5, AdamW, weight_decay=0, max_length=8192; gradient accumulation=32 follows `dar_kaggle_sft_clean.ipynb`. Training uses 0.5 epoch, with no max_steps override. Pilot still uses a subset and two samples per accepted video (DAR + auxiliary), so equal epochs do not imply equal baseline compute/data exposure.
- Student saves full model weights (without optimizer state) and evaluation loads that checkpoint directly, without a PEFT adapter. Checkpoints are first written under ephemeral `/kaggle/temp`; in `full` mode the final checkpoint is copied to `/kaggle/working` after evaluation if enough space remains. Evidence and evaluation reports remain under `/kaggle/working`. BF16-capable, high-memory GPU required; the baseline notebook specifies RTX 6000 96 GB. A single T4 is not the supported full-SFT configuration. No automatic fallback to LoRA. Teacher NF4 does not reduce student training memory.
- Notebook default: `ARM = 'stsg'`. Only graph generation, STSG SFT and STSG dev evaluation run.
- Later set `ARM = 'caption'` (or `'stsg_no_links'`) and run in a separate output directory. No baseline is automatically trained.
- Modes: `smoke` uses 8 train / 4 dev, `pilot` uses 256 train / 64 dev, and `full` uses every available train video except a frozen 64-video dev split. `NUM_TRAIN_EPOCHS=0.5` in all modes; checkpoint selection uses the highest saved step.
- Set `TEACHER_MODEL` and `BASE_MODEL` to distinct Kaggle Inputs. Teacher requires the **full snapshot**, including its six custom Python files, tokenizer, configs and weights. Official revision inspected: `d5b763e368861e7f5096e7ff1b49f92fbccf8ae6`.
- Teacher uses a separate venv with Transformers 4.57.1; student keeps 4.57.1 / swift 3.12.5. The notebook is **Internet Off only**: `TEACHER_WHEELHOUSE=None` auto-detects the attached 4.57.1 wheel. The small `ffmpeg-python` and `future` wheels are supplied by the attached `dar-teacher-runtime-wheels` dataset. Pip always uses `--no-index`, with no online fallback.
- Both runtimes reuse Kaggle's existing CUDA Torch stack. Student bootstrap excludes Torch, torchvision, torchaudio, Triton, torchao and NVIDIA wheels; teacher resolution pins installed CUDA-stack versions and performs an offline dry run first. If base Torch is CPU-only or no GPU is available, bootstrap stops with a diagnostic. No `--force-reinstall` of the CUDA stack.
- Student bootstrap selects ms-swift 3.12.5 and Transformers 4.57.1 from the offline wheelhouse. Teacher uses its own isolated site-packages with Transformers 4.57.1; the shared CUDA Torch stack is not replaced.
- Teacher bootstrap creates a fresh venv with `--without-pip` before invoking kernel pip via `--python` (pip >=22.3; [official pip documentation](https://pip.pypa.io/en/stable/topics/python-option/)). It prioritizes teacher site-packages while retaining Kaggle support package paths. Both checks print actual Torch/Transformers import paths and assert CUDA plus the required versions. Reruns create new subdirectories under `RUNTIME` / `TEACHER_VENV`; old environments are preserved, so failed CPU-wheel installs cannot contaminate the new overlay.
- After uploading the updated notebook, restart the Kaggle session and run configuration → student bootstrap → teacher bootstrap → remaining cells. With `TEACHER_WHEELHOUSE=None`, the teacher cell auto-detects a `transformers-4.57.1` wheel under `/kaggle/input`; set an explicit path only when multiple copies are attached.
- Default `TEACHER_4BIT=False`: the teacher loads in FP16 on the selected RTX PRO 6000 GPU. NF4 remains optional if compatible bitsandbytes wheels are supplied; it does not reduce student training memory.
- Teacher uniformly samples up to 16 frames and records indices/timestamps and processed tensor hashes. Its native image processing differs from Qwen student's processing; do not claim identical visual tensors across architectures.
- Output: `/kaggle/working/dar_videollama3_full_epoch05_<MODE>_<ARM>/`. Each run remains exclusive-write; use a fresh directory for deliberate retries.
- A single-arm evaluation reports absolute scores, no improvement claim. Separate runs can have different accepted training IDs due to invalid evidence: compare `arms/build.json` first. The already-running baseline must use compatible train/dev IDs, training budget, trainable-module policy and evaluation settings before attributing differences to graphs.
- For a matched graph/caption comparison later, generate both kinds and build both arms jointly with `--arms stsg caption` on the same frozen split; train each selected arm separately.

Official API: https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/inference/example_videollama3.py

## CLI sequence (run from repository root)

```bash
python experiments/dar_stsg/prepare.py split \
  --train /data/train.jsonl --test /data/test.jsonl --video-root /data/videos \
  --train-size 256 --dev-size 64 --output /work/split

/work/teacher-venv/bin/python experiments/dar_stsg/run.py teacher --backend videollama3 --kinds stsg --load-in-4bit --model /models/VideoLLaMA3-7B \
  --input /work/split/teacher_inputs.jsonl --output /work/evidence.jsonl

python experiments/dar_stsg/prepare.py build --split /work/split \
  --evidence /work/evidence.jsonl --output /work/arms --arms stsg

BASE_MODEL=/models/Qwen2.5-VL-3B-Instruct DATA_DIR=/work/arms \
OUTPUT_ROOT=/work/models ARM=stsg NUM_TRAIN_EPOCHS=0.5 \
bash experiments/dar_stsg/train.sh

python experiments/dar_stsg/run.py predict --model /work/models/stsg-seed1234/ACTUAL_RUN/checkpoint-ACTUAL_FINAL_STEP \
  --input /work/split/dev.jsonl --output /work/stsg-predictions.jsonl
```

Repeat training and prediction for `baseline`, `caption`, `stsg_no_links`, each
from the same **base**, using different output paths. Use smoke mode first to check the runtime, then fresh directories for the 0.5-epoch pilot. Checkpoints are saved with the epoch strategy, including the final fractional epoch.

```bash
python experiments/dar_stsg/evaluate.py --manifest /work/split/dev.jsonl \
  --predictions baseline=/work/baseline-predictions.jsonl \
  caption=/work/caption-predictions.jsonl stsg=/work/stsg-predictions.jsonl \
  stsg_no_links=/work/stsg_no_links-predictions.jsonl \
  --reference caption --output /work/comparison.json

python -m unittest discover -s experiments/dar_stsg -p test_pipeline.py -v
```

Train paths are rewritten by basename; `video-root` must contain those files
directly. Source annotations must be DAR conversation JSONL with measured
`video_duration`. No duration fallback from the last GT segment is allowed.
Test overlap is checked by basename video ID; near-duplicate content with different
IDs needs a separate dataset audit. Dev is held out from original train, not from
the official test set. `prepare.py split` never selects examples based on errors.

Graph/caption pseudo-labels are generated only for train videos. Teacher calls are
label-blind. Schema checks do not validate facts; review a preselected sample
against video before scaling. Graphs must fit 8 entities/8 events/16 temporal
links. An empty link list is valid for a single event. A model producing only DAR
JSON despite the graph instruction is unsuitable as the graph teacher without
additional grounding training.

All arms use the same videos on which **both** caption and graph parse. Rejections
are logged in `build.json`. This controls selection across arms but restricts
training to teacher-parseable videos. The baseline repeats the DAR target to match
two rows per video. The primary graph-specific comparison is against `caption`.
No-link ablation preserves timestamps: it tests explicit edges, not all temporal
information. Matching steps/rows is not matching target-token FLOPs.

Runtime: use the user's offline package set (`ms-swift 3.12.5`, `transformers
4.57.1`, `qwen-vl-utils 0.0.14`); the script intentionally avoids importing the
separate `ms-swift/` 4.0 development checkout. Model inference uses installed
Transformers/PEFT; data/scoring tools only need Python + NumPy/SciPy. No vLLM is
needed. Original prompt/scoring functions are extracted with AST from `test.py`
to preserve them without importing its GPU runtime.

Outputs are exclusive-create. Failed JSON generations are logged and not silently
retried; missing/invalid eval results count in the denominator. GPU/runtime/video
errors abort the run. A partial run is not resumable automatically; preserve it
and document a fresh run if needed. `run.py` logs raw outputs, frame hashes,
sampled fps, tokens, generation time, package versions and code/input hashes.
Sampled fps is passed through to Qwen2.5, avoiding source-fps timestamp errors.

The official compatibility score uses the repo's repair policy; primary joint F1
uses strict unmodified outputs. Neither score measures factual grounding. Dataset
format validation, source-reading and CPU tests are complete; training/inference
and VRAM fit have not been verified on Kaggle yet.
