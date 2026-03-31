# Firefighting Device Detection with SimCLR

This repository fine-tunes a SimCLR-style visual encoder on the Roboflow firefighting device dataset:
https://universe.roboflow.com/yaid-pzikt/firefighting-device-detection/dataset/6

## Approach

The dataset is exported in COCO detection format, but the assignment asks us to fine-tune SimCLR to recognize the symbols. SimCLR is a representation-learning method rather than an object detector, so this project converts the COCO boxes into symbol crops and trains in two stages:

1. Self-supervised SimCLR pretraining on cropped symbols from the training split.
2. Supervised fine-tuning of the encoder plus a classification head on the cropped symbol labels.

This keeps the solution aligned with the assignment while still using the existing annotations.

## Dataset Notes

- `Data/train`, `Data/valid`, and `Data/test` contain the images and COCO annotations.
- The training split contains 40 labeled classes with annotations.
- One class, `explosion-proof-smoke-detector`, appears in validation but not in training, so the fine-tuning workflow filters validation/test samples down to classes that actually exist in the training split.

## Files

- `firefighting_simclr/data.py`: COCO parsing, crop extraction, and augmentations.
- `firefighting_simclr/models.py`: ResNet-18 backbone, SimCLR projection head, and classifier.
- `firefighting_simclr/training.py`: training loops, evaluation, and artifact saving.
- `firefighting_simclr/notebook_utils.py`: notebook-friendly helpers for building configs and running the full pipeline.
- `notebooks/simclr_firefighting_demo.ipynb`: main demonstration notebook for the assignment.

## Install

```bash
pip install -r requirements.txt
```

## Notebook Demo

A presentation-friendly notebook is available at `notebooks/simclr_firefighting_demo.ipynb`.

The notebook is self-contained and can call the training pipeline directly through helper functions in `firefighting_simclr/notebook_utils.py`.

In VS Code, open that file with the Jupyter extension and select the same Python interpreter used for this project. If the terminal command `jupyter` is not on PATH, that is still fine here because the environment supports Jupyter through:

```bash
python -m jupyter --version
```

## Run

The recommended way to run experiments now is from the notebook:

1. Open `notebooks/simclr_firefighting_demo.ipynb`.
2. Run the bootstrap and import cells first.
3. Run the smoke-test config cell to verify the pipeline.
4. Run `run_full_pipeline(smoke_args)` for a quick check or switch to `full_args` for a longer training run.

If you prefer running from a plain Python session instead of Jupyter, the same helper module can be used like this:

```bash
python -c "from pathlib import Path; from firefighting_simclr.notebook_utils import make_experiment_args, run_full_pipeline; args = make_experiment_args(data_dir=Path('Data'), output_dir=Path('outputs/final_run'), device='cuda', amp=True); run_full_pipeline(args)"
```

## Outputs

The training pipeline saves:

- `simclr_pretrain.pt`: pretrained SimCLR encoder checkpoint.
- `best_symbol_classifier.pt`: best classifier checkpoint based on validation macro F1.
- `pretrain_history.json`: contrastive training history.
- `finetune_history.json`: fine-tuning history.
- `test_metrics.json`: final test metrics and per-class report.
- `dataset_summary.json`: crop counts and filtered-class summary.

## Default Training Choices

- Backbone: `resnet18`
- Crop size: `96x96`
- Crop padding: `15%`
- SimCLR projection head: `256 -> 128`
- Fine-tuning uses balanced sampling to reduce class-imbalance issues.
- Validation model selection uses macro F1 because the class distribution is highly imbalanced.
