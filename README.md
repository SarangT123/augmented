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
import ar
augmented = ar.ar(<capture:int"""Camera number""">, <targetImage:str"""location of the image to replace""">, <overlayImage:str"""Image to overlay""", >)
```
## Overlaying
```py
augmented.ar_overlay(<nfeatures:int>"""Used for opencv calcs""", <debug:bool>"""choosing to debug or not""", <confidence:int"""confidence of the ai model""">,<displayName:str"""Title for the window of the app""">, <exit:str"""key to exit the program""">)
```