# Model comparison — development protocol

Both checkpoints share the same 10,000-image CelebA manifest, 9,000 training images, 64×64 architecture and 128-dimensional latent input. The extended run resumed the recovery checkpoint and added 15 epochs. Both were scored using the same clean-fid 0.1.35 protocol; see EVALUATION.md.

| Measure | Recovery epoch 15 | Extended epoch 30 |
|---|---:|---:|
| Training images | 9,000 | 9,000 |
| Cumulative steps | 1,050 | 2,100 |
| Resolution | 64×64 | 64×64 |
| Generator parameters | 3,806,080 | 3,806,080 |
| Development FID ↓ | 318.7571 | 176.0718 |
| Development KID ↓ | 0.3554918 | 0.1750159 |
| Pixel L1 distance, 32 fixed seeds | 0.64128 | 0.65582 |
| Horizontal adjacent-pixel L1 | 0.10749 | 0.08811 |
| Local CPU latency, 32 images | 0.0299 s | 0.0407 s |
| Visual quality | Recognizable with heavy texture and face distortion | More coherent on fixed grid; still distorted, uneven and low resolution |

The two CPU latency measurements are single invocations, not a benchmark distribution. Pairwise pixel L1 is a limited diversity diagnostic. The FID/KID reference set is small and was used for checkpoint comparison; no independent final score exists. Fixed-seed grids are at `artifacts/quality_baseline/fixed_grid.png` and `artifacts/quality_extended/fixed_grid.png`. The extended checkpoint is selected for local inference because it improves both development metrics and the representative fixed-seed grid, while both checkpoints remain preserved.
