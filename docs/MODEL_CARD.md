# FaceForge AI — private research model card

The selected research checkpoint is `artifacts/kaggle_recovery/run/generator.pt`. It is an independently trained DCGAN generator producing 64×64 RGB images from a 128-dimensional Gaussian latent vector. Its outputs show recognizable human-face structure with substantial blur, noise and artifacts. They are not photorealistic portraits.

Training: 10,000 selected CelebA aligned images, 9,000 used for training, 15 epochs, 1,050 updates, batch 128, Tesla T4, Adam 0.0002 for both networks. The generator has 3,806,080 parameters. FID, KID, memorization and demographic diversity: NOT EVALUATED. The split is image-level only. CelebA permits non-commercial research; public redistribution of these weights and outputs has not been cleared. The generator is local and private. Outputs may resemble real people and should not be used for identity verification or impersonation.
