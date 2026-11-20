# Training failure diagnosis

## Confirmed
The 30-epoch run made only 840 optimizer updates over 1,844 training images. Its final fixed-noise grid is high-frequency texture rather than recognizable faces. Decreasing discriminator loss did not demonstrate image quality. The selected set is small relative to the face variation in CelebA. This establishes inadequate outcome, not a unique causal mechanism.

## Direct checks
A local CelebA image is 178×218 RGB and after center crop, resize and normalization measured [-0.992, 1.0], mean -0.380, SD 0.449. Generator Tanh output at the stability checkpoint was [-0.469, 0.617], mean -0.016, SD 0.131. Both image-saving paths map [-1,1] to [0,1] correctly. Generator train/eval modes returned nearly identical ranges, so a BatchNorm inference mismatch was not found. The discriminator has no Sigmoid; BCEWithLogitsLoss receives logits. The alternating-step test confirms both networks' parameters change, and generator gradients do not accumulate in the discriminator. Real/fake labels have the expected directions in code. The generator spatial dimensions are 4→8→16→32→64; discriminator is 64→32→16→8→4→1.

## Suspected
Few independent faces, short optimization, limited architecture width, discriminator advantage, and possible transposed-convolution artifacts may contribute. These remain hypotheses. The 30-epoch run's final D loss 0.167 and G loss 3.072 are consistent with an adversarial imbalance; they do not isolate its cause. A separate 5-epoch run had D loss 0.006 and G loss 12.034.

## Corrective experiment
Preserve both prior runs. Use more aligned images and optimizer steps in a new, private, controlled run. Save fixed-noise samples frequently and inspect them before asserting success. Keep dataset and model artifacts private under CelebA's non-commercial terms. Do not claim FID/KID until actually measured.

## After the controlled recovery run
With 10,000 selected images (9,000 training), width 64 and 1,050 updates, fixed-noise samples developed recognizable face structure by epoch 5. Epoch-15 images have plausible eyes, noses, mouths and hair but pronounced noise and distortions. This supports the hypothesis that the previous configuration lacked adequate data/capacity/optimization exposure, but the new run changed several variables together; it does not prove which one was decisive. No display bug was found. Further controlled ablations would be needed for a unique root cause.
