# Research notes

- Goodfellow et al., Generative Adversarial Networks: https://arxiv.org/abs/1406.2661
- Radford et al., DCGAN: https://arxiv.org/abs/1511.06434
- PyTorch tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- CelebA official: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- FFHQ official: https://github.com/NVlabs/ffhq-dataset

The baseline uses a non-saturating generator objective and binary cross entropy on discriminator logits. Batch normalization and transposed convolutions follow common DCGAN principles. Adversarial losses do not reliably measure visual quality; fixed latent grids aid qualitative comparison. The present image-level holdout neither isolates identities nor eliminates perceptual duplicates. FID/KID and nearest-neighbor memorization analysis have not been performed. Model weights may require additional rights clearance independent of source code licensing. Perceptual similarity is not proof of memorization or anonymity.
