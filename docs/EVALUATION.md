# Evaluation status

FID: NOT EVALUATED. KID: NOT EVALUATED. Perceptual diversity: NOT EVALUATED. Memorization: NOT EVALUATED. Prior runs produce texture rather than faces; numerical quality scores would not alter that visual finding. An image-level holdout exists but is not identity disjoint. Before a publishable evaluation, freeze a suitable reference split, verify duplicates and identities where metadata permits, use an established metric implementation with documented feature extractor and sample counts, and conduct a private nearest-neighbor review without redistributing source images.

## Recovery outcome
Visual inspection of actual fixed-noise grids shows recognizable but visibly distorted faces at epoch 15. Local CPU inference from the exported generator produced four 64×64 images in approximately 0.09 s on the test invocation. FID/KID remain NOT EVALUATED. No quantitative diversity, nearest-neighbor, identity leakage or privacy assessment has been completed. The held-out 1,000 images were not used for an established metric.

## FaceForge 2.0 fixed protocol
`configs/evaluation.json` freezes 64×64 preprocessing, seed 20261120, and nominal 1,000 development and 1,000 final generated samples. Both models must use the same real-reference set and clean-fid implementation before numerical comparison. The existing recovery run has only 1,000 image-level held-out images, no identity disjointness and no larger untouched final set. FID would be sample-biased and cannot be called an independent final estimate. FID/KID remain NOT EVALUATED. The 32-image fixed grid is a visual baseline and its pixel L1 dispersion is a limited diagnostic, not an established diversity score.

## Measured development comparison (FaceForge 2.0)
Private Kaggle kernel `fariskishtah/faceforge-clean-fid-comparison` ran clean-fid 0.1.35 in clean mode with InceptionV3 features on a Tesla T4. It used the same 1,000 image-level held-out CelebA images for both models, converted to RGB, centered 178×178 crop, LANCZOS resize to 64×64, saved as uint8 PNG. Each generator produced 1,000 uint8 PNG images from Gaussian latent vectors seeded 20261120. Tanh output was mapped from [-1,1] to [0,255] by clamping, scaling and rounding. Results are in `artifacts/kaggle_eval/evaluation.json`.

| Model | FID ↓ | KID ↓ |
|---|---:|---:|
| Recovery, epoch 15 | 318.7571 | 0.3554918 |
| Extended, epoch 30 | 176.0718 | 0.1750159 |

These are comparative development estimates only. With 1,000 real examples, FID has considerable finite-sample bias. The same holdout was used to compare and select models, so it is not an untouched final test. Identity overlap and near duplicates are not audited. Real-image resizing in this evaluation used LANCZOS whereas training used torchvision's default PIL resize filter; both models share the same evaluation protocol, so the comparison is consistent but the distribution gap may affect absolute values. No confidence intervals were calculated. No demographic fairness, memorization or privacy guarantee follows from these scores.
