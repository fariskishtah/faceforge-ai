# Training experiments

| Run | Face images | Train images | Batch | Epochs | Steps | G/D LR | Visual outcome |
|---|---:|---:|---:|---:|---:|---|---|
| Kaggle baseline | 2,048 | 1,844 | 64 | 5 | 140 | 0.0002/0.0002 | Noise |
| Kaggle stability | 2,048 | 1,844 | 64 | 30 | 840 | 0.0002/0.00005 | Noise |

Both used 64×64 CelebA aligned crops, z=128, width=32 and Tesla T4. The baseline loop took 25.72 s and stability loop 153.07 s, excluding data preparation. Full logs, grids and checkpoints are preserved under ignored `artifacts/`. The next run will use a larger subset and retain its own directory.
| Kaggle recovery 10k | 10,000 | 9,000 | 128 | 15 | 1,050 | 0.0002/0.0002 | Recognizable low-resolution faces with heavy artifacts |

Recovery: private Kaggle Tesla T4, PyTorch CUDA, 64×64, z=128, width=64, seed 42. The training loop lasted 435.18 s. The final D loss was 0.728 and G loss 3.026; these are descriptive, not quality scores. Fixed-noise samples at epochs 1, 5, 10 and 15 are stored in `artifacts/kaggle_recovery/run/`. Faces are recognizable by epoch 5 and remain noisy at epoch 15. The checkpoint and logs are private. The manifest is image-level split only and does not establish identity independence.
| Extended recovery | 10,000 | 9,000 | 128 | 30 cumulative | 2,100 cumulative | 0.0002/0.0002 | More coherent fixed grid; artifacts remain |

The extended run reused the exact frozen manifest (SHA-256 `1f28e34e72bb912fbbb37a08954dcf0272d64071e0baed5ff9dd16623122033f`) and checkpoint. Its additional 15 epochs took 860.17 s on Tesla T4. Private Kaggle kernel: `fariskishtah/faceforge-extended-dcgan`. Checkpoint `artifacts/kaggle_extend/run/checkpoint.pt` SHA-256 `19bbb1825b56008dfcf035c744766f0f4e1daaa5ce16ee342e1ddaa58a9659e5`; generator-only export SHA-256 `1998b102398d0da2f82f5438d560966acdcd285f93b0d611acbcc170be2249d6`.
