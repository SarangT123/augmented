# augmented

**augmented** is a cross-platform Python library for augmented reality. It provides image overlay and ArUco marker-based AR using OpenCV.

## Features

- **Image Overlay** — Detect a target image in a live camera feed and overlay another image on top of it using ORB feature matching and homography.
- **ArUco Marker AR** — Detect ArUco markers in real-time and augment them with custom images.
- Debug mode for visualizing feature matches and warping steps.
- Simple, minimal API.

## Installation

```bash
pip install augmented
```

**Requirements:**
- Python 3.9+
- opencv-contrib-python >= 4.8.0
- numpy >= 1.21.0

## Usage

### Image Overlay (`ar_overlay`)

```python
import augmented

ar = augmented.ar_overlay(capture=0)
ar.setup(
    targetImage="target.jpg",
    overlayImage="overlay.png",
    nfeatures=1000,
    debug=True,
    confidence=25,
    displayName="Augmented"
)
ar.start(display=True)
```

Run in a loop for continuous tracking:

```python
while True:
    ar.start(display=True)
```

### ArUco Marker AR (`arucoar`)

```python
import augmented

arucoar = augmented.arucoar(cam=0)
imgAug = {0: "assets/unnamed.jpg"}
arucoar.setup(
    imgAug=imgAug,
    markerSize=6,
    totalMarkers=250,
    debug=True,
    displayName="Augmented AR"
)

while True:
    arucoar.start(display=True)
```

## API Reference

### `ar_overlay(capture: int)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `capture` | `int` | Camera device index |

#### `setup(targetImage, overlayImage, nfeatures, debug, confidence, displayName)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `targetImage` | `str` | — | Path to the image to detect |
| `overlayImage` | `str` | — | Path to the overlay image |
| `nfeatures` | `int` | — | Number of ORB features to detect (1000 recommended) |
| `debug` | `bool` | `True` | Show debug visualizations |
| `confidence` | `int` | `25` | Minimum feature matches to trigger overlay |
| `displayName` | `str` | `"Augmented by sarang"` | OpenCV window title |

#### `start(display: bool) -> list`

Process one frame. Returns `[stacked_frame, augmented_frame]`.

### `arucoar(cam: int)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cam` | `int` | `0` | Camera device index |

#### `setup(imgAug, markerSize, totalMarkers, debug, displayName)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imgAug` | `dict` | — | `{aruco_id: image_path}` mapping |
| `markerSize` | `int` | `6` | ArUco marker grid size |
| `totalMarkers` | `int` | `250` | Total markers in the dictionary |
| `debug` | `bool` | `True` | Enable debug output |
| `displayName` | `str` | `"Augmented by Sarang"` | OpenCV window title |

#### `start(display: bool) -> numpy.ndarray`

Process one frame. Returns the augmented frame.

## License

BSD-3-Clause. See [LICENSE](./license).

## Links

- **PyPI**: [pypi.org/project/augmented](https://pypi.org/project/augmented/)
- **Source**: [github.com/SarangT123/augmented-documentation](https://github.com/SarangT123/augmented-documentation)
- **Docs**: [sarangt123.github.io/augmented-documentation](https://sarangt123.github.io/augmented-documentation/)
