# Firefighting Device Detection with SimCLR

This repository fine-tunes a SimCLR-style visual encoder on the Roboflow firefighting device dataset:
https://universe.roboflow.com/yaid-pzikt/firefighting-device-detection/dataset/6

## Approach

The dataset is exported in COCO detection format, and we train and finetune a SimCLR with ResNet as the encoder. SimCLR is a representation-learning method rather than an object detector, so this project converts the COCO boxes into symbol crops and trains in two phases:

1. Pretraining phase: apply the Self-supervised SimCLR pretraining on cropped symbols from the training set to fit the ResNet encoder. The SimCLR consists of a ResNet encoder and a projection header, which together transforms an image crop to a vector. After doing data augmentataion, we want the augmented crops coming from the same original crops to be as close as possible, while those from different original crops to be as far as possible.
2. Supervised fine-tuning of the encoder plus a classification head on the cropped symbol labels for the final classification task.

What makes this project different from the original paper is that the original paper did not evaluate the SimCLR's loss on validation set during pretraining phase, which is due to the sufficiently large size of dataset on ImageNet, but here we have implemented the model evaluation on the evaluation set to avoid overfitting. The reason is because our dataset here is much smaller than ImageNet, so the risk of overfitting is higher. If overfitted, the common practices are: increase weight decay in AdamW, reduce the pretraining epochs...

## Dataset Notes

- `Data/train`, `Data/valid`, and `Data/test` contain the images and COCO annotations.
- The training split contains 40 labeled classes with annotations.
- The finetuning workflow filters validation/test samples down to the classes that actually exist in the training split.


## Files

- `firefighting_simclr/data.py`: COCO parsing, crop extraction, and augmentations.
- `firefighting_simclr/models.py`: ResNet-18 backbone, SimCLR projection head, and classifier.
- `firefighting_simclr/training.py`: training loops, evaluation, and artifact saving.
- `firefighting_simclr/notebook_utils.py`: notebook-friendly helpers for building configs and running the full pipeline.
- `notebooks/simclr_firefighting_demo.ipynb`: main demonstration notebook

## Install

```bash
pip install -r requirements.txt
```

## Notebook Demo

A presentation-friendly notebook is available at `notebooks/simclr_firefighting_demo.ipynb`.

The notebook is self-contained and can call the training pipeline directly through helper functions in `firefighting_simclr/notebook_utils.py`.

## Run

The recommended way to run experiments now is from the notebook. Follow the code chunks in order. 



## Outputs

The training pipeline saves:

- `simclr_pretrain.pt`: pretrained SimCLR encoder checkpoint.
- `best_symbol_classifier.pt`: best classifier checkpoint based on validation macro F1.
- `pretrain_history.json`: contrastive training history.
- `finetune_history.json`: fine-tuning history.
- `test_metrics.json`: final test metrics and per-class report.
- `dataset_summary.json`: crop counts and filtered-class summary.

For the ease of comparing different configurations of the arguments, we save the results above to subfolders named `arg_setting_N`, where N is non-negative integer.

## Default Training Choices

- Backbone: `resnet18`
- Crop size: `96x96`
- Crop padding: `15%`
- SimCLR projection head: `256 -> 128`
- Fine-tuning uses balanced sampling to reduce class-imbalance issues.
- Validation model selection uses macro F1 because the class distribution is highly imbalanced.
