# Data Scaling Effects on CNN and Vision Transformer Performance

A controlled CIFAR-10 study comparing a CIFAR-adapted ResNet-18 and a ViT-Tiny trained from scratch with 10%, 50%, and 100% of a fixed training pool.

The project asks two practical questions:

1. How does training-set size affect the generalization of the two model families?
2. Is weak low-data ViT performance better explained by overfitting, insufficient optimization, or limited data?

This is a reproducible baseline study, not a claim that architecture alone causes every observed difference. The models use documented, model-specific optimization protocols and differ in both capacity and inductive bias.

## Experimental design

- **Dataset:** CIFAR-10, downloaded through `torchvision`
- **Training pool:** 45,000 images
- **Validation set:** fixed, stratified 5,000-image split
- **Test set:** official 10,000-image CIFAR-10 test split
- **Training scales:** 4,500 / 22,500 / 45,000 images
- **Subset construction:** class-balanced and nested; the 10% subset is contained in the 50% subset, which is contained in the full training pool
- **Training transforms:** random crop, horizontal flip, tensor conversion, normalization
- **Validation/test transforms:** deterministic tensor conversion and normalization
- **Checkpoint selection:** highest validation accuracy
- **Test policy:** one evaluation after the best validation checkpoint is restored

### Baseline protocols

| Model | Architecture | Optimizer | Initial LR | Weight decay | Schedule | Epochs |
|---|---|---|---:|---:|---|---:|
| CNN | CIFAR-adapted ResNet-18, trained from scratch | SGD, momentum 0.9 | 0.01 | 5e-4 | cosine decay | 30 |
| Transformer | ViT-Tiny, patch size 4, 6 layers, 3 heads, trained from scratch | AdamW | 3e-4 | 0.05 | cosine decay | 30 |

## Main results

| Training-pool share | Images | ResNet-18 test accuracy | ViT-Tiny test accuracy | ResNet advantage |
|---:|---:|---:|---:|---:|
| 10% | 4,500 | 69.37% | 49.16% | 20.21 points |
| 50% | 22,500 | 87.19% | 70.40% | 16.79 points |
| 100% | 45,000 | 91.46% | 78.47% | 12.99 points |

ResNet-18 performed better at all three observed scales, but the gap narrowed as more data became available. ViT-Tiny gained 29.31 test-accuracy points between the 10% and 100% conditions, compared with 22.09 points for ResNet-18. Under these protocols, the result is consistent with the convolutional model being more data-efficient in a small-image, from-scratch setting.

## Diagnosing the 10% ViT result

Two follow-up packages were selected and compared only on the validation set:

| ViT-Tiny configuration | Best validation accuracy | Change from baseline |
|---|---:|---:|
| 30-epoch baseline | 50.10% | — |
| 50 epochs + 5-epoch warmup + higher peak LR + label smoothing | 55.70% | +5.60 points |
| Stronger dropout + attention dropout + label smoothing + weight decay | 46.56% | −3.54 points |

The longer-training package improved both fitting and validation performance, showing that optimization contributed to the original low-data result. It did not close the gap with ResNet-18, so insufficient convergence is only a partial explanation. The stronger-regularization package reduced both training and validation accuracy, which is more consistent with additional underfitting than with corrected overfitting.

These packages change multiple settings at once. They are diagnostic comparisons rather than single-factor ablations.

## Repository structure

```text
.
├── README.md
├── requirements.txt
└── notebooks/
    └── cnn_vit_data_scaling.ipynb
```

The notebook is committed with its outputs so that the reported learning curves and metrics can be inspected without rerunning GPU training.

## Reproduce the experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/cnn_vit_data_scaling.ipynb` and run all cells. CIFAR-10 is downloaded automatically to `./data`.

GPU training is strongly recommended. The recorded run used a Kaggle NVIDIA T4 environment.

## Evidence boundary and limitations

- Each condition has one full training run. The dataset split is fixed, but seed-to-seed uncertainty has not been estimated.
- ResNet-18 and ViT-Tiny differ in parameter count, optimizer, learning rate, and inductive bias. The comparison measures the documented training systems, not architecture in isolation.
- The two follow-up configurations change several hyperparameters together.
- Results are limited to CIFAR-10, 32×32 images, and training from scratch.
- No conclusion is made about large-scale training or pretrained transformers.

## Highest-value next experiments

1. Repeat the six baseline conditions with at least three full random seeds and report mean ± standard deviation.
2. Report trainable parameters, training time, and peak GPU memory; add a capacity-matched comparison.
3. Separate training duration, warmup, learning rate, and label smoothing into controlled ablations.
4. Compare from-scratch training with pretrained fine-tuning for both model families.
5. Add per-class accuracy and confusion matrices for architecture-specific failure analysis.

## Version history

The original course-project state is retained in Git history and tagged as `v1-course-project`. The current `main` branch contains the revised validation protocol, rerun results, and evidence-bounded analysis.
