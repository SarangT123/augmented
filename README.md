# augmented 🧚
augmented is a cross platform module used to do augmented reality using python

# Requirements 🟥
----
- opencv-contrib-python
- Numpy
* requirements may vary on diffrent operating systems

## instalation ⬇️
`pip install augmented`

Thats it 🌟

# How to use ❓ 
## Overlaying images 🖼️
## initializing 🏁
-----------
```py
import augmented
ar = augmented.ar_overlay(capture:int)#capture = camera number
ar.setup(targetImage: str, overlayImage: str, nfeatures: int, debug: bool=True, confidence: int=25, displayName: str="Augmented by sarang")
```

- targetImage = Image to overlay on top of

- overlayImage =Image to overlay

- nfeatures = Features to detect on target image the bigger the more accurate and the more resource intensive 1000 recomended

#### Not required but can tweak the ones below 💻


- debug = debug mode

- confidence  = How many feature matches to confirm

- displayname = title name```

### Overlaying
```py
ar.start(display=bool)
```

- display =  Enabling display output

## Aruco scanning 📱
### setup 🖱️
```py
import augmented
arucoar = augmented.arucoar(cap:int=0)
imgAug = {0: 'assets/unnamed.jpg'}
arucoar.setup(imgAug: dict, markerSize: int = 6, totalMarkers: int = 250, debug: bool = True, cam: int = 0, displayName: str = 'Augmented by Sarang')
```

- imgAug = a dict containing the aruco id as value and the image(location) to display when the value is True

#### not neccessry but still can tweak 💻

- Markersize, totalMarkers = aruco code properties

- debug = to use debug mode

- cam = camera number

- displayName = tite of the display window
---
### Scanning and overlaying 🖼️

```py
arucoar.start(display=bool)
```
- display = wheather to display the output or not

#### recomended way of using is to put the code inside a loop 🌠

--------------------------
## version - 2.1.0Stable
## contributions are appreciated and will be credited in the package
# Thank you 
# Happy augmenting
