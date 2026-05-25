# Lane Detection and Lane Departure Warning

This project explores lane detection for driving videos using two different
approaches:

1. **A deep learning technique** based on a LaneNet-style architecture with a
   ResNet34 encoder, dual decoder heads, instance embeddings, and postprocessing
   for lane clustering.
2. **A classical image processing technique** based on perspective transform,
   HSV thresholding, histogram search, and sliding windows.

The main deliverable is a Streamlit application that lets the user compare both
approaches on uploaded media or predefined example videos. The deep learning
path also includes an optional lane departure warning visualization.

The project is intentionally structured as both an engineering demo and a study
of why modern autonomous-driving perception has moved away from simple fixed
threshold image processing toward learned, data-driven perception models.

## Project Focus

The focus of this project is lane detection under realistic driving-video
conditions. Lane detection is a key perception task for driver assistance and
autonomous driving systems because lane boundaries help estimate:

- the drivable region,
- the ego lane,
- lane center offset,
- lane departure risk,
- road curvature and direction.

The project compares two families of methods:

### Classical image processing

The image processing pipeline is useful for understanding older lane-detection
systems and the basic geometry of the problem. It is fast and explainable, but
it is sensitive to:

- lighting changes,
- shadows,
- night scenes,
- worn lane markings,
- camera angle,
- road curves,
- reflections,
- manually selected threshold values.

### Deep learning

The deep learning pipeline is the main focus of the project. It learns lane
features from data instead of relying only on fixed color thresholds or edge
rules. The model predicts:

- a binary lane/background segmentation map,
- an embedding vector for each pixel, used to separate individual lane
  instances.

The binary segmentation head answers: "Which pixels are lane markings?"

The embedding head answers: "Which lane instance does each lane pixel belong
to?"

This makes the model closer to modern perception systems, where the network
learns visual patterns from data and postprocessing converts dense predictions
into usable lane curves.

## Repository Structure

```text
.
├── app/
│   ├── streamlit_app.py          # Main Streamlit user interface
│   ├── lane_warner.py            # Ego-lane selection and lane departure logic
│   ├── video_processor.py        # Reserved for app utilities
│   ├── visualization.py          # Reserved for app visualization helpers
│   └── checkpoints/              # Local model checkpoints, ignored by Git
├── DeepLearningTechnique/
│   ├── src/
│   │   ├── config.py             # Training image size, batch size, LR, weights
│   │   ├── loss.py               # Binary, Dice, and discriminative losses
│   │   ├── postprocess.py        # DBSCAN clustering and lane polynomial fitting
│   │   ├── utils.py              # Checkpointing, seed setup, IoU metric
│   │   ├── data/
│   │   │   ├── dataset.py        # PyTorch Dataset for lane manifests
│   │   │   └── transforms.py     # Augmentation and tensor conversion
│   │   └── models/
│   │       ├── lanenet.py        # LaneNet and ResNet34-LaneNet models
│   │       ├── encoder.py        # ENet encoder blocks
│   │       └── ResNetEncoder.py  # ResNet34 encoder wrapper
│   ├── train.ipynb               # Training notebook
│   ├── trainCULane.ipynb         # CULane training/preparation notebook
│   ├── realTime.ipynb            # Real-time testing notebook
│   └── trial.ipynb               # Experiments and debugging
├── ImageProcessingTechnique/
│   ├── main.py                   # Original OpenCV script
│   ├── main2.py                  # Alternate original OpenCV script
│   └── processor.py              # Streamlit-friendly image processing pipeline
├── exampleImages/                # Optional predefined images for the app
├── exampleVideos/                # Optional predefined videos for the app
├── requirements.txt
└── README.md
```

## Deep Learning Model Overview

The primary model is implemented in:

```text
DeepLearningTechnique/src/models/lanenet.py
```

There are two model variants in the code:

1. **LaneNet with ENet encoder**
2. **LaneNetResNet34 with ResNet34 encoder and U-Net-style decoders**

The Streamlit application can load checkpoints for either architecture. It
detects the architecture by inspecting checkpoint state dictionary keys.

## LaneNet Concept

LaneNet is an instance segmentation approach for lane detection. Instead of
directly predicting a fixed number of lanes, it predicts dense pixel-level
outputs:

- **Binary segmentation output:** classifies each pixel as background or lane.
- **Instance embedding output:** assigns each lane pixel an embedding vector.

After inference, lane pixels are clustered in embedding space. Pixels that
belong to the same physical lane should have similar embeddings, while pixels
from different lanes should be separated.

This is useful because the number of lanes can vary between frames. A fixed
output head such as "left lane, right lane, third lane" would be brittle.
Embedding-based instance segmentation allows the model to detect a variable
number of lanes.

## ResNet34-LaneNet Architecture

The strongest architecture in this project is:

```python
LaneNetResNet34
```

It consists of:

- a ResNet34 encoder,
- a binary segmentation decoder,
- an instance embedding decoder.

### Encoder

The encoder is implemented in:

```text
DeepLearningTechnique/src/models/ResNetEncoder.py
```

It uses the convolutional part of ResNet34. The classification head is removed
because lane detection needs dense spatial predictions, not image-level labels.

The encoder returns four feature maps:

```text
s4, s8, s16, s32
```

These correspond to feature maps at strides:

- `s4`: 1/4 input resolution, 64 channels
- `s8`: 1/8 input resolution, 128 channels
- `s16`: 1/16 input resolution, 256 channels
- `s32`: 1/32 input resolution, 512 channels

The model uses these as U-Net-style skip connections.

### Decoder

The decoder is implemented as `ResNet34Decoder` in:

```text
DeepLearningTechnique/src/models/lanenet.py
```

The decoder starts from the deepest feature map `s32` and upsamples step by
step:

```text
s32 -> s16 -> s8 -> s4 -> s2 -> s1
```

At each compatible scale, it concatenates encoder skip features with the
upsampled decoder features. This gives the decoder both:

- semantic understanding from deep layers,
- spatial precision from shallow layers.

Each decoder block uses:

- bilinear upsampling,
- concatenation with skip features when available,
- two `3x3` convolution layers,
- batch normalization,
- ReLU activation,
- dropout.

The code uses bilinear upsampling instead of transposed convolution in the
ResNet34 decoder to reduce checkerboard artifacts, which are especially
problematic for thin structures such as lane markings.

### Two Output Heads

The model has two independent decoders:

```python
self.binary_decoder
self.embedding_decoder
```

The binary decoder outputs:

```text
(B, 2, H, W)
```

The two channels represent:

- background,
- lane.

The embedding decoder outputs:

```text
(B, embedding_dim, H, W)
```

In this project, `embedding_dim` is usually `4`.

The forward pass returns:

```python
binary_logits, embedding = model(x)
```

## Input Resolution

The configured training and inference resolution is:

```python
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 384
```

This keeps a `2:1` aspect ratio and remains divisible by 32. Divisibility by 32
matters because the ResNet34 encoder downsamples to stride 32. If the input
dimensions are not compatible with the encoder/decoder scale factors, feature
alignment becomes more awkward.

## Dataset Format

The dataset loader is:

```text
DeepLearningTechnique/src/data/dataset.py
```

The dataset expects a manifest file such as `train.txt` or `val.txt`. Each line
contains three relative paths:

```text
image_path binary_mask_path instance_mask_path
```

Example:

```text
train_set/clips/0313-1/11100/20.jpg train_set/seg_label/clips/0313-1/11100/20.png train_set/instance_label/clips/0313-1/11100/20.png
```

The dataset returns:

```python
image, binary_mask, instance_mask
```

Where:

- `image` is the RGB road image,
- `binary_mask` marks lane vs background,
- `instance_mask` contains unique lane instance IDs.

The instance mask is required for the discriminative embedding loss.

## Data Preprocessing and Augmentation

The preprocessing pipeline is implemented in:

```text
DeepLearningTechnique/src/data/transforms.py
```

Each sample is processed as follows:

1. Load image with OpenCV.
2. Resize image to `768x384`.
3. Convert image from BGR to RGB.
4. Load binary lane mask as grayscale.
5. Resize binary mask with nearest-neighbor interpolation.
6. Convert binary mask to `0` background and `1` lane.
7. Load instance mask as grayscale.
8. Resize instance mask with nearest-neighbor interpolation.
9. Apply training augmentations if `training=True`.
10. Normalize image using ImageNet mean and standard deviation.
11. Convert image and masks to PyTorch tensors.

Nearest-neighbor interpolation is used for masks because bilinear interpolation
would create invalid label values.

### Augmentations

Training augmentations include:

- color jitter,
- random Gaussian blur,
- random horizontal flip,
- random translation,
- random perspective perturbation.

These augmentations improve robustness to lighting, camera movement, small
geometric changes, and visual noise.

The image is normalized using ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

This is especially appropriate for the ResNet34 encoder because it is designed
around ImageNet-style input normalization.

## Loss Functions

The loss functions are implemented in:

```text
DeepLearningTechnique/src/loss.py
```

The total loss combines:

1. binary segmentation loss,
2. discriminative embedding loss.

```python
total = binary_weight * binary + disc_weight * disc
```

## Binary Segmentation Loss

The binary segmentation loss combines cross entropy and Dice loss:

```python
binary = ce_weight * binary_ce + dice_weight * binary_dice
```

By default:

```python
ce_weight = 0.5
dice_weight = 0.5
```

### Weighted Cross Entropy

Lane detection is highly imbalanced. Most pixels are background, and only a
small fraction are lane markings. Without class weighting, the model could
achieve a deceptively low loss by predicting background everywhere.

The project uses class weights from `config.py`:

```python
CLASS_WEIGHTS = torch.tensor([1.4540, 20.1856])
```

These give lane pixels much more weight than background pixels.

### Dice Loss

Dice loss directly optimizes overlap between predicted lane pixels and ground
truth lane pixels. It is helpful for thin objects because small segmentation
errors can be severe even when the total pixel count is small.

The Dice term is computed on the softmax probability of the lane channel.

## Discriminative Embedding Loss

The embedding head is trained with a discriminative loss. This loss encourages:

- pixels from the same lane instance to be close together in embedding space,
- pixels from different lane instances to be far apart,
- cluster centers to stay bounded.

It contains three terms:

### Variance term

Pulls pixel embeddings toward their lane instance mean.

### Distance term

Pushes different lane instance means away from each other.

### Regularization term

Keeps embedding means from growing too large.

The loss uses these default margins:

```python
delta_var = 0.3
delta_dist = 1.5
```

This makes the embedding output useful for clustering during postprocessing.

## Training Process

Training is primarily performed in notebooks:

```text
DeepLearningTechnique/train.ipynb
DeepLearningTechnique/trainCULane.ipynb
```

The training workflow is:

1. Prepare dataset manifests.
2. Create `LaneDataset` instances for training and validation.
3. Wrap datasets with PyTorch `DataLoader`.
4. Instantiate `LaneNetResNet34`.
5. Choose optimizer and learning rate.
6. For each epoch:
   - load batch,
   - run model forward pass,
   - compute binary and embedding losses,
   - backpropagate,
   - clip gradients,
   - update weights,
   - evaluate on validation set.
7. Save checkpoints.

Important configuration values:

```python
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
EPOCHS = 150
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 384
```

The code also includes checkpoint utilities in:

```text
DeepLearningTechnique/src/utils.py
```

These functions save and load:

- epoch number,
- model state dictionary,
- optimizer state dictionary.

## Postprocessing the Deep Learning Output

The model does not directly output lane curves. It outputs dense tensors.
Postprocessing converts those tensors into lane polynomials.

Postprocessing is implemented in:

```text
DeepLearningTechnique/src/postprocess.py
```

The main function is:

```python
my_postprocess(binary_logits, embedding, ...)
```

The process is:

1. Convert binary logits into lane probabilities with softmax.
2. Threshold lane probability to create a binary lane mask.
3. Remove small connected components.
4. Find the horizon row where lane pixels begin.
5. Ignore pixels above the horizon plus padding.
6. Collect lane pixels.
7. Build feature vectors for clustering:
   - embedding vector,
   - normalized x coordinate,
   - normalized y coordinate.
8. Cluster pixels with DBSCAN.
9. Remove clusters with too few pixels.
10. Fit polynomial curves to lane clusters.
11. Sort lanes left-to-right.

The polynomial is fitted as:

```text
x = f(y)
```

This is better than fitting `y = f(x)` because road lanes are often close to
vertical in image coordinates.

## DBSCAN Clustering

The project uses `sklearn.cluster.DBSCAN` to cluster lane pixels.

Important parameters:

```python
eps = 0.35
min_samples = 20
spatial_weight = 0.5
min_pixels = 100
poly_degree = 2
```

DBSCAN is useful because the number of lane instances is not fixed. It can
discover however many clusters exist in the prediction.

## Lane Drawing

Detected lane clusters are drawn by sampling the fitted polynomial over the
lane's valid vertical range:

```python
xs = np.polyval(lane["poly"], ys)
```

Each lane receives a color from a predefined palette. The Streamlit app also
shows binary output, embedding visualization, and lane probability heatmap.

## Ego Lane Detection

Ego-lane selection is implemented in:

```text
app/lane_warner.py
```

The current algorithm uses a fixed reference point:

```python
DEFAULT_EGO_REFERENCE_POINT = (0.5, 0.9)
```

This means:

- `x = 50%` of image width,
- `y = 90%` of image height.

For a `768x384` deep-learning frame, this is approximately:

```text
(384, 345)
```

The algorithm:

1. Evaluates each detected lane at the reference y-coordinate.
2. Ignores lanes that do not exist at that y-coordinate.
3. Finds the closest lane to the left of the reference point.
4. Finds the closest lane to the right of the reference point.
5. If either side is missing, no ego lane pair is selected.

This design avoids extrapolating short or distant lane detections into the ego
region.

## Lane Departure Warning

Lane departure warning is optional in the Streamlit app. When enabled, the app:

1. selects ego lanes,
2. estimates the ego lane center,
3. compares the vehicle center to the lane center,
4. computes offset ratio,
5. displays a warning if the offset exceeds a threshold.

The warning threshold is:

```python
warning_threshold = 0.28
```

The offset is:

```text
(vehicle_center_x - lane_center_x) / lane_width
```

If the absolute value is high, the vehicle is considered to be drifting left or
right.

When lane departure is disabled, lane drawings, lane canvas, and ego-lane
visualization remain active. Only the warning overlay is disabled.

## Classical Image Processing Pipeline

The Streamlit-compatible image-processing pipeline is implemented in:

```text
ImageProcessingTechnique/processor.py
```

The pipeline is:

1. Resize frame to `640x480`.
2. Select a perspective transform preset for the media file.
3. Warp the road region into a bird's-eye view.
4. Convert warped frame to HSV.
5. Apply HSV thresholding.
6. Build a lower-half histogram of lane-like pixels.
7. Initialize left and right lane bases from the histogram.
8. Search upward using sliding windows.
9. Fit straight lines to detected left and right points.
10. Warp lane overlay back to the original perspective.

### Perspective Presets

Perspective points are defined per video:

```python
PERSPECTIVE_PRESETS = {
    "testVideo1": ...,
    "testVideo2": ...,
    ...
}
```

The file-to-preset mapping is:

```python
MEDIA_PERSPECTIVE_PRESETS = {
    "testVideo1.mp4": "testVideo1",
    "testVideo2.mp4": "testVideo2",
    ...
}
```

This is necessary because classical perspective warping is camera-specific.

### HSV Presets

HSV threshold presets are also defined per video:

```python
MEDIA_HSV_PRESETS = {
    "testVideo1.mp4": {
        "lower": (...),
        "upper": (...),
    },
    ...
}
```

HSV values mean:

- `H`: hue, the color family,
- `S`: saturation, how colorful the pixel is,
- `V`: value, how bright the pixel is.

The classical method keeps only pixels inside the selected HSV range. This is
why it can fail at night: if lane markings are too dark, the mask can become
empty.

## Streamlit Application

The user interface is:

```text
app/streamlit_app.py
```

Run it with:

```bash
streamlit run app/streamlit_app.py
```

The app supports:

- selecting deep learning or image processing,
- choosing a model checkpoint,
- enabling or disabling lane departure warning,
- selecting image or video input,
- selecting predefined media from `exampleImages/` or `exampleVideos/`,
- uploading custom images or videos,
- viewing intermediate model outputs.

### Deep Learning Display Outputs

For deep learning, the app can show:

- raw input,
- detected lane overlay,
- lane canvas,
- binary segmentation output,
- embedding visualization,
- lane probability heatmap,
- lane departure state.

### Image Processing Display Outputs

For image processing, the app can show:

- raw input,
- image processing result,
- bird's-eye transformed view,
- HSV threshold mask,
- sliding-window visualization,
- detected left/right point counts.

## Installation

Create and activate a Python environment, then install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On macOS with the existing local environment, the app can be launched with:

```bash
venv/bin/streamlit run app/streamlit_app.py
```

## Checkpoints

Model checkpoints are expected in:

```text
app/checkpoints/
```

They are not tracked by Git because checkpoint files are large. The app lists
available `.pt` files in that directory and lets the user choose one from the
sidebar.

The loader supports checkpoints saved either as:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...
}
```

or directly as a raw model state dictionary.

## Current Limitations

### Deep learning limitations

- Performance depends strongly on training data quality and diversity.
- The model may struggle with domains not represented in training.
- Postprocessing parameters such as DBSCAN `eps`, lane probability threshold,
  and minimum cluster size are hand-tuned.
- Lane departure warning assumes camera-centered geometry and a fixed reference
  point for ego-lane selection.

### Image processing limitations

- HSV thresholding is sensitive to lighting.
- Night scenes can produce empty masks.
- Perspective presets must be tuned per camera/video.
- The method assumes visible lane markings.
- Strong shadows, glare, rain, reflections, or worn markings can break the
  pipeline.
- It is less general than the deep learning approach.

## Why Compare Both Techniques?

The classical pipeline is fast, explainable, and useful for learning the
geometry of lane detection. However, it depends heavily on manually chosen
rules. The deep learning pipeline is more complex, but it learns richer visual
features and can generalize better when trained on diverse data.

This comparison demonstrates the practical tradeoff:

- image processing is easier to understand but brittle,
- deep learning is harder to train but more robust.

That tradeoff is the central engineering lesson of the project.

