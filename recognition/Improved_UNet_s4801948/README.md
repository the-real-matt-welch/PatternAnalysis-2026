# Improved U-Net for Prostate MRI Segmentation (HipMRI)

## Overview

In prostate radiotherapy planning, the prostate and the surrounding organs at risk (bladder, rectum and bone) must be delineated on MRI with millimetre-level accuracy. Under-contouring the prostate risks tumour recurrence, while over-contouring irradiates healthy rectal and bladder wall and causes radiation toxicity. Manual delineation is slow, and an annotation assistant that proposes contours for a specialist to review and adjust could save considerable clinical time. This project builds the segmentation engine for such an assistant: a 2D convolutional network that assigns every pixel of a HipMRI pelvic MRI slice to one of six classes (background, body, bone, bladder, rectum, prostate). The task is hard for three reasons. The prostate is tiny (about 0.4% of all pixels in the training data), so a model can score well overall while barely segmenting it. Soft-tissue boundaries are low-contrast, and the thin interface between prostate, rectum and bladder is exactly where errors are clinically costly. And a good average Dice score can hide failures at the top (base) and bottom (apex) of the gland. We therefore implement the Improved U-Net, an extension of the original U-Net, and evaluate it against a standard 2D U-Net baseline, paying particular attention to where along the gland errors occur and not only to the global score.

## How it works

The network is an encoder–decoder with skip connections (see the figure below). A slice is z-score normalised, padded to a fixed 256×144 canvas and passed through a stem convolution. The encoder then halves the resolution four times with stride-2 convolutions, doubling the channels each time (32 → 512), and after each step applies a context module: a pre-activation residual block (instance normalisation → leaky ReLU → 3×3 conv, then dropout and a second normalisation → activation → conv, added back to its input). Putting normalisation and activation before the convolutions leaves the identity path untouched, which improves gradient flow, and instance normalisation keeps training stable with small batches. The decoder reverses the process: each level upsamples the coarser features, concatenates the encoder features from the same resolution (the skip connection, which restores the fine spatial detail lost by downsampling), and refines them in a localization module. In addition, the network uses deep supervision: 1×1 convolutions at the three finest decoder levels each produce a segmentation map, and these maps are upsampled and summed element-wise into the final prediction, so that earlier decoder layers receive a direct training signal. The model outputs six per-pixel scores, and the argmax gives the predicted label map. It is trained with a combined cross-entropy and soft-Dice loss (the Dice term counteracts the class imbalance, see below for the specific function), computed only over real, non-padded pixels. The 2D baseline uses the same data pipeline and loss, but plain double-convolution blocks with batch normalisation, no dropout and no deep supervision.

<p align="center">
  <img src="resources/architecture.png" alt="Architecture of the 2D Improved U-Net" width="900">
</p>

<p align="center"><em>Figure 1. The 2D Improved U-Net. Blue: encoder context modules; green: decoder localization modules; dashed grey: skip connections; purple: deep-supervision heads whose upsampled outputs are summed into the final logits.</em></p>

### Loss function

Let $\Omega$ be the set of valid (non-padded) pixels in a batch, $p_{i,c}$ the softmax probability of class $c$ at pixel $i$, and $y_{i,c} \in \{0,1\}$ the one-hot ground truth. The model is trained with an equally weighted sum of cross-entropy and soft Dice loss:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \mathcal{L}_{\mathrm{Dice}}
$$

$$
\mathcal{L}_{\mathrm{CE}} = -\frac{1}{|\Omega|} \sum_{i \in \Omega} \sum_{c=0}^{C-1} y_{i,c} \,\log p_{i,c}
$$

$$
\mathcal{L}_{\mathrm{Dice}} = 1 - \frac{1}{C-1} \sum_{c=1}^{C-1} \frac{2\sum_{i \in \Omega} p_{i,c}\, y_{i,c} + \varepsilon}{\sum_{i \in \Omega} p_{i,c} + \sum_{i \in \Omega} y_{i,c} + \varepsilon}
$$

where $C = 6$ is the number of classes and $\varepsilon = 1$ is a smoothing constant. The Dice term is computed per class over the whole batch (not per slice), which keeps it stable for classes that appear in only some slices, and it averages over the five foreground classes only, so the large background class does not dominate. Cross-entropy provides well-behaved gradients early in training, while the Dice term directly optimises overlap and counteracts the extreme class imbalance.
