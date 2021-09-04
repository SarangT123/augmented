# ar-python

Ar python is a cross platform module used to do augmented reality using python

`pip install ar-python`

# Requirements
----
-Open cv (opencv-python)
-Numpy
-keyboard
* requirements may vary on diffrent operating systems

# How to use 
## initializing
-----------
```py
import ar-python as ar
augmented = ar.ar(<capture:int"""Camera number""">, <targetImage:str"""location of the image to replace""">, <overlayImage:str"""Image to overlay""", >)
```
## Overlaying