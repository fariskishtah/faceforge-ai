# FaceForge AI

A work-in-progress DCGAN research engine and local synthetic-face studio for Faris Mohamed Kishtah's INSTANT AI & Data Science Training, Sprint 4. **Status: research source release. A private 30-epoch CelebA model produces recognizable 64×64 faces with visible artifacts. Weights and training images are not distributed.** See PROJECT_STATUS.md before using or publishing.

The generator transforms 128-dimensional Gaussian noise through transposed convolutions and batch normalization into a 64×64 RGB image in [-1,1]. The discriminator returns logits. The trainer alternates binary cross entropy discriminator and non-saturating generator updates; fake images are detached for the discriminator step. Loss alone is not an image quality metric. Based on GAN (https://arxiv.org/abs/1406.2661) and DCGAN (https://arxiv.org/abs/1511.06434), implemented independently.

## Run

Python 3.12 recommended. `pip install -r requirements.txt` then, with a separately obtained authorized aligned-face dataset:

```sh
PYTHONPATH=. python scripts/prepare_dataset.py /path/to/faces --output data/manifest.json
PYTHONPATH=. python scripts/train.py --manifest data/manifest.json --out artifacts/run1 --epochs 10 --batch 64 --device cpu
FACEFORGE_CHECKPOINT=artifacts/run1/checkpoint.pt python app.py
PYTHONPATH=. python scripts/export_generator.py artifacts/run1/checkpoint.pt artifacts/run1/generator.pt
PYTHONPATH=. python scripts/generate_samples.py artifacts/run1/generator.pt --out artifacts/samples --seed 42 --count 4
```

Use `--device mps` or `--device cuda` only when supported and tested. Resume with the same options plus `--resume artifacts/run1/checkpoint.pt`, and increase `--epochs`. Checkpoints contain Python pickle; only load trusted files. The current checkpoint is a full training checkpoint, not an inference-only distribution. The application has seed-based batches and linear latent interpolation. The studio reads saved fixed-noise snapshots and loss curves; the generator-only export is available. FID/KID, perceptual and identity-aware deduplication, privacy audit, browser QA and public deployment remain incomplete. A training run on arbitrary data does not demonstrate face synthesis.

Dataset and weight rights are separate from source code rights. See docs/DATASET_AUDIT.md. Generated images may resemble real people and must not be used for identity verification or impersonation.

## Recorded baseline

A private Kaggle Tesla T4 run used 2,048 validated CelebA images (1,844 train), 5 epochs and 140 updates. Outputs remained colored noise and discriminator loss fell close to zero. The training logs, fixed-noise grids and checkpoints are held in the ignored local artifacts directory; see PROJECT_STATUS.md for hashes, hardware, limitations and subsequent experiments. No publication rights for weights or images are assumed. FID and KID: NOT EVALUATED.

A follow-up 30-epoch run with discriminator learning rate 0.00005 produced no recognizable faces. A new 10,000-image run produced recognizable face structure by epoch 5; its final output remains noisy. See PROJECT_STATUS.md for configurations and hashes.

The academic 64×64 GAN demonstration now produces recognizable face structures, with visible distortion. The public product is not released. A real browser screenshot of the local generator is in `artifacts/screenshots/studio_generate.png` (private, excluded from Git). The local browser test confirmed generation and the model-quality view. Mobile, download and full browser QA remain open. FID/KID and privacy metrics remain NOT EVALUATED.

## FaceForge 2.0 measured comparison

A controlled continuation trained the same DCGAN from 1,050 to 2,100 updates with the same 9,000 training images and optimizer settings. On a shared 1,000-image development reference, clean-fid 0.1.35 yielded FID 318.7571→176.0718 and KID 0.3554918→0.1750159. These are biased small-sample development estimates, not independent final scores. Visual samples became more coherent overall but remain distorted. See [evaluation](docs/EVALUATION.md), [model comparison](docs/MODEL_COMPARISON.md), and [optimization report](docs/OPTIMIZATION_REPORT.md).

The selected generator stays private because publication rights for weights and outputs trained on CelebA are unresolved. This repository contains original source and documentation, not training photographs, checkpoints or private sample screenshots. The MIT license covers original project code only. To run the studio, supply your own authorized, trusted FaceForge checkpoint with `FACEFORGE_CHECKPOINT`. Set `FACEFORGE_COMPARE_CHECKPOINT` to another compatible checkpoint for the comparison tab.
