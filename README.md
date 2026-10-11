# Aerial Imagery Segmentation Project

This repository contains a collection of PyTorch segmentation experiments for aerial/rooftop imagery. The notebooks compare several encoder-decoder architectures for binary segmentation tasks such as solar-panel detection, rooftop segmentation, and related aerial object extraction.

The project is organized as multiple experiment folders, each containing the same model families trained on different data splits or benchmark datasets.

## Project structure

- `BSP/`  
  Baseline segmentation experiments on the BSP dataset.

- `BSP_augmented/`  
  Same model family, but with augmented data or alternate training runs.

- `CrowdSourcedDatasetResults/`  
  Crowd-sourced dataset experiments for comparison across model families.

- `PV01(Benchmarkdataset)/`  
  Benchmark dataset experiments, including the FCN model and multiple U-Net variants.

Inside each folder, you will find notebooks like:

- `*_unet.ipynb`
- `*_residualunet.ipynb`
- `*_multiresunet.ipynb`
- `*_unetpp.ipynb`
- `*_resnetunet.ipynb`
- `FCN1.ipynb`

These are not separate frameworks; they are repeated implementations of the same architecture families evaluated on different datasets.

---

## What the project is doing

The notebooks build segmentation models that take an aerial image as input and return a binary mask:

- Input: RGB image (for example, rooftop or solar farm imagery)
- Output: probability map or binary mask highlighting the foreground object
- Loss: usually a combination of BCE and Dice loss for imbalanced foreground/background regions
- Metrics: Dice, IoU, precision, recall, F1, accuracy

This is a binary segmentation setup, so the mask is typically a single-channel map with values 0 or 1.

---

## Common training pattern used across the notebooks

Across the notebooks, the training pipeline is very similar:

1. Load image and corresponding mask pair
2. Resize to a fixed size (commonly 256 x 256)
3. Convert mask to binary using `label > 0`
4. Pass image through the model
5. Compute logits from the network
6. Optimize a combined loss:
   - Binary Cross Entropy (BCE)
   - Dice loss
7. Track segmentation metrics after training

The dataset classes in the notebooks are usually built like this:

- `SegDataset` or `PV01Dataset`
- Reads images and masks from `img/` and `mask/` directories
- Matches files by stem name
- Uses file pairs like `image001.png` and `image001_mask.png`

This is a standard segmentation pipeline for paired image/mask training.

---

## 1) Plain U-Net

### Architecture

The plain U-Net is the base model in this project.

It has:

- Encoder path: repeated convolution blocks + max pooling
- Bottleneck: deepest feature representation
- Decoder path: transpose convolution + skip concatenation
- Final layer: 1x1 convolution producing a binary segmentation map

### Typical structure

The notebooks define a `DoubleConv` block and a `UNet` class.

Each `DoubleConv` includes:

- Conv2d
- BatchNorm2d
- ReLU
- optional Dropout2d
- second Conv2d
- BatchNorm2d
- ReLU
- optional Dropout2d

The model uses feature sizes like:

- 32, 64, 128, 256 in the encoder
- 512 in the bottleneck

This matches the classic U-Net idea:

- downsample for context
- upsample for localization
- skip connections preserve detailed spatial information

### Why it matters

This is the baseline reference architecture. It is simple, effective, and often a strong baseline for segmentation tasks.

---

## 2) Residual U-Net (ResUNet)

### Architecture

The Residual U-Net is very similar to the plain U-Net, but each convolution block is replaced with a residual block.

A residual block is designed as:

- Main path: two convolution layers
- Shortcut path: identity or 1x1 convolution
- Output = ReLU(main + shortcut)

This helps with deeper networks and improves gradient flow.

### Key idea

Instead of learning a raw transformation directly, the network learns a residual update over the input:

- `output = ReLU(conv2(conv1(x)) + shortcut(x))`

This makes training easier and improves stability when the model becomes deeper.

### Why it differs from plain U-Net

Plain U-Net uses standard `DoubleConv` blocks.
Residual U-Net keeps the same U-Net layout but changes each block into a residual unit, giving better learning behavior and sometimes higher segmentation quality on difficult aerial scenes.

---

## 3) MultiResUNet

### Architecture

This is the MultiResUNet architecture introduced in the paper by Ibtehaz and Rahman.

It is designed to replace the typical U-Net block with a richer feature extraction block called a `MultiResBlock`.

### MultiRes block design

Each MultiRes block contains:

- a 3x3 convolution branch
- a 5x5-like branch approximated by stacked 3x3 convolutions
- a 7x7-like branch approximated by stacked 3x3 convolutions
- a residual shortcut using 1x1 convolution

The outputs are concatenated and merged with the residual path.

The code implements this as:

- `conv3x3`
- `conv5x5`
- `conv7x7`
- concatenation
- skip/shortcut
- BatchNorm and ReLU

### Why it is special

The major benefit is that it captures multi-scale information inside each block without needing multiple manual branches. It combines feature richness and residual learning.

### Additional idea: ResPath

The notebook also adds `ResPath` modules on the skip connections.

This means the skip features are not passed directly; they are processed through residual connections before concatenation with the decoder stage.

This often improves the flow of features across the encoder-decoder boundary.

---

## 4) ResNet-UNet

### Architecture

This model combines a ResNet encoder with a U-Net decoder.

The encoder is a pretrained ResNet backbone such as:

- ResNet18
- ResNet34
- ResNet50
- ResNet101

The code selects the backbone using:

- `models.resnet18`, `resnet34`, etc.
- optional ImageNet pretraining via `weights=...`

### Structure

The encoder uses:

- stem conv + BN + ReLU
- max pooling
- `layer1`, `layer2`, `layer3`, `layer4`

Then the decoder rebuilds the spatial resolution using:

- ConvTranspose2d upsampling
- skip concatenation with encoder features
- `DoubleConv` block after each merge

### Why this is useful

A pretrained ResNet encoder gives strong feature extraction on natural images, which is often helpful for aerial imagery as well. This architecture is often stronger than a plain U-Net when a good pretrained backbone is available.

---

## 5) U-Net++

### Architecture

U-Net++ is a nested U-Net, meaning it has more dense skip connections than the standard U-Net.

Instead of only connecting the encoder and decoder at corresponding levels, it connects:

- current encoder features
- previous decoder features
- multiple earlier feature maps across different scales

This creates dense skip paths and a more deeply supervised structure.

### Important idea

The code defines nested blocks like:

- `conv0_1`, `conv0_2`, `conv0_3`, `conv0_4`
- plus their decoder-level counterparts

These are used to fuse multiple feature levels before final prediction.

### Why it differs from U-Net

Plain U-Net uses one skip connection per level.
U-Net++ uses a more nested/denser fusion strategy, which can help recover fine object boundaries and details.

---

## 6) FCN with ResNet-50 backbone

### Architecture

This is a Fully Convolutional Network using a pretrained ResNet-50 backbone.

The notebook uses:

- `torchvision.models.segmentation.fcn_resnet50`
- `ResNet50_Weights.IMAGENET1K_V1`

### What FCN does

The model replaces the final fully connected classifier with convolutional layers, keeping spatial dimensions and producing a dense per-pixel output map.

This makes it well-suited for dense prediction tasks such as segmentation.

### Why it is in the project

It gives a modern, pretrained segmentation baseline that is useful to compare against custom U-Net-style models.

---

## Loss and metrics used in the notebooks

A shared theme across the project is the use of a hybrid segmentation loss:

### BCE + Dice loss

`combined_loss = BCEWithLogitsLoss + DiceLoss`

This is common in segmentation because:

- BCE handles pixel-wise classification
- Dice helps with class imbalance, especially when the object is small compared to the background

### Metrics

The code tracks metrics such as:

- Dice coefficient
- IoU (Intersection over Union)
- precision
- recall
- F1 score
- accuracy
- mean IoU (mIoU in some notebooks)

These metrics are computed after applying a sigmoid and thresholding around 0.5.

---

## Typical binary segmentation pipeline

The notebooks follow this flow:

1. load dataset
2. prepare image/mask pairs
3. train model
4. predict segmentation mask
5. threshold probability map
6. compute segmentation metrics
7. compare across architectures

This is exactly the pattern you would expect in an aerial segmentation benchmark project.

---

## Why there are many repeated notebooks

The repository has repeated versions of the same architecture in different folders such as:

- `BSP/`
- `BSP_augmented/`
- `CrowdSourcedDatasetResults/`
- `PV01(Benchmarkdataset)/`

These are not fundamentally different model families. They are experiments on different datasets and different data splits, often with the same architecture repeated for benchmarking.

So the real project is:

- benchmark multiple segmentation models
- on multiple aerial datasets
- using similar training and evaluation code

---

## Recommended reading order

If you want to understand the project quickly, read in this order:

1. Open one U-Net notebook
2. Open one Residual U-Net notebook
3. Open one MultiResUNet notebook
4. Open one ResNet-UNet notebook
5. Open the U-Net++ notebook
6. Open the FCN notebook
7. Compare shared dataset and training cells once you understand the structure

That sequence will make the architecture differences obvious.

---

## Quick summary

| Model | Main idea | Typical strength |
|---|---|---|
| U-Net | classic encoder-decoder with skip connections | simple and effective baseline |
| ResUNet | U-Net + residual blocks | better gradient flow |
| MultiResUNet | multi-scale residual blocks in a U-Net | richer spatial feature extraction |
| ResNet-UNet | ResNet encoder + U-Net decoder | strong pretrained feature extractor |
| U-Net++ | nested dense skip connections | better boundary recovery |
| FCN-ResNet50 | fully convolutional model with deep backbone | strong modern benchmark baseline |

---

## Practical takeaway

This project is a comparison study of segmentation architectures for aerial imagery. The architecture differences are mostly in how the encoder extracts features and how the decoder reconstructs spatial detail.

If you are learning the project:

- focus on the `DoubleConv` / `ResBlock` / `MultiResBlock` patterns
- understand the U-Net skip connection idea
- then compare how each model modifies that idea

That is the heart of the project.

---

## Example model usage

Each notebook follows the same idea:

```python
model = UNet(in_channels=3, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for images, masks in train_loader:
    logits = model(images)
    loss = combined_loss(logits, masks)
    loss.backward()
    optimizer.step()
```

Similarly, you can swap in:

- `ResUNet(...)`
- `MultiResUNet(...)`
- `ResNetUNet(...)`
- `UNetPP(...)`
- `FCNResNet50(...)`

---

## Final note

This repository is best understood as a benchmark of segmentation models for aerial image extraction. The code is educational and experimental, but the architecture concepts are standard and widely used in computer vision research.

If you want, the next useful step would be to add a single consolidated training script that loads the same dataset and runs all of these models from one place.
