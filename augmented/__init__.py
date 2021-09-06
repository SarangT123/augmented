import cv2
import numpy as np
import cv2.aruco as aruco
print("""Beta version of ar-python (Copyright (c) 2021, Sarangt123 (Sarang T))
Created by : Sarang T (india,kerala)
Report bugs and issues at the githubpage
Email : sarangthekkedathpr@gmail.com
Contributers : Sarang""")
"""
Copyright (c) 2021, Sarangt123 (Sarang T)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the augmented-python nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""


class ar_overlay():
    def __init__(self, capture: int):
        import cv2
        import numpy as np
        if not isinstance(capture, int):
            raise TypeError("Expected an int")
        self.cap = capture
        return None

    def setup(self, targetImage: str, overlayImage: str, nfeatures: int, debug: bool = True, confidence: int = 25, displayName: str = "Augmented by sarang"):
        if not isinstance(nfeatures, int):
            raise TypeError('Expected a int')
        if not isinstance(debug, bool):
            raise TypeError
        if not isinstance(confidence, int):
            raise TypeError('Expected an int')
        if not isinstance(targetImage, str):
            raise TypeError("Expected a str")
        self.targetImage = targetImage
        if not isinstance(overlayImage, str):
            raise TypeError("Expected a str")
        self.overlayImage = overlayImage
        """

        """
        cap = cv2.VideoCapture(self.cap)
        imgTarget = cv2.imread(self.targetImage)
        Overlay = cv2.imread(self.overlayImage)
        self.nfeatures = nfeatures
        imgOverlay = Overlay
        """

        """
        """
        """
        height, width, char = imgTarget.shape
        # imgOverlay = cv2.resize(imgOverlay, (height, width))
        """

        """
        orb = cv2.ORB_create(nfeatures=nfeatures)
        kp1, des1 = orb.detectAndCompute(imgTarget, None)
        imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)
        imgAug = []
        self.cap = cap
        self.imgTarget = imgTarget
        self.imgOverlay = imgOverlay
        self.dimensions = [height, width]
        self.orb = orb
        self.kpds1 = [kp1, des1]
        self.imgAug = imgAug
        self.debug = debug
        self.confidence = confidence
        self.displayName = displayName

    def start(self, display: bool = True):
        try:
            success, Webcam = self.cap.read()
            imgAug = Webcam.copy()
            kp2, des2, = self.orb.detectAndCompute(Webcam, None)
            Webcam = cv2.drawKeypoints(Webcam, self.kpds1[0], None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(self.kpds1[1], des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if self.debug:
                print('length of good matches :' + len(good))
            imgFeatures = cv2.drawMatches(
                self.imgTarget, self.kpds1[0], Webcam, kp2, good, None, flags=2)
            if len(good) > self.confidence:
                srcpt = np.float32(
                    [self.kpds1[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dstpt = np.float32(
                    [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(
                    srcpt, dstpt, cv2.RANSAC, 5)
                if self.debug:
                    print(f'matrix : {matrix}')
                pts = np.float32([[0, 0], [0, self.dimensions[0]], [self.dimensions[1], self.dimensions[0]], [
                    self.dimensions[1], 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                img2 = cv2.polylines(
                    Webcam, [np.int32(dst)], True, (255, 0, 255), 3)

                imgWarp = cv2.warpPerspective(
                    self.imgOverlay, matrix, (Webcam.shape[1], Webcam.shape[0]))
                maskNew = np.zeros(
                    (Webcam.shape[0], Webcam.shape[1]), np.uint8)
                cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
                maskInverse = cv2.bitwise_not(maskNew)
                imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInverse)
                imgAug = cv2.bitwise_or(imgWarp, imgAug)
                if self.debug:
                    cv2.imshow('Debug window', img2)
                    cv2.imshow('Debug window', imgWarp)
                    cv2.imshow('Debug window', maskNew)
                    cv2.imshow('Debug window', mask)
                    cv2.imshow('Debug window', maskInverse)
            if self.debug:
                print('length of good matches = ' + len(good))
                cv2.imshow('Debug window', imgFeatures)

            stacked = np.concatenate((Webcam, imgAug), axis=0)
            self.cap = self.cap
            cv2.waitKey(1)
            if display:
                cv2.imshow(self.displayName, imgAug)
            return [stacked, imgAug]

        except Exception as e:
            print(e)

    def help(self):
        print('Please check the documentation at the pypi page')


class arucoar():
    def __int__(self, cap: int = 0):
        import cv2
        import cv2.aruco as aruco
        import numpy as np
        if not isinstance(cap, int):
            raise TypeError("Expected an int")
        self.cap = cap

    def findArucoMakers(self, img, draw=True, markersize=6, totoalmarkers=250):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key = getattr(aruco, f'DICT_{markersize}X{markersize}_{totoalmarkers}')
        arucoDict = aruco.Dictionary_get(key)
        arucoParam = aruco.DetectorParameters_create()
        bboxes, ids, rejected = aruco.detectMarkers(
            img_grey, arucoDict, parameters=arucoParam)
        if draw:
            aruco.drawDetectedMarkers(img, bboxes)
        return[bboxes, ids]

    def augmentAruco(self, bbox, ids, img, imgAug, draw=True):
        imgAug = imgAug.get(ids[0], False)
        if self.debug:
            print(f'ImgAug var : {imgAug}')
        if imgAug != False:
            if self.debug:
                print('Augmenting....')
            imgAug = cv2.imread(imgAug)
            topL = bbox[0][0][0], bbox[0][0][1]
            topR = bbox[0][1][0], bbox[0][1][1]
            btmR = bbox[0][2][0], bbox[0][2][1]
            btmL = bbox[0][3][0], bbox[0][3][1]
            #
            size = imgAug.shape
            #
            if self.debug:
                print('Coords : '+topL, topR, btmR, btmL)
            pts_dst = np.array([topL, topR, btmR, btmL])
            pts_src = np.array(
                [
                    [0, 0],
                    [size[1] - 1, 0],
                    [size[1] - 1, size[0] - 1],
                    [0, size[0] - 1]
                ], dtype=float
            )
            h, status = cv2.findHomography(pts_src, pts_dst)
            temp = cv2.warpPerspective(
                imgAug, h, (img.shape[1], img.shape[0]))
            cv2.fillConvexPoly(img, pts_dst.astype(int), 0, 16)
            frame = img + temp
            return frame
        else:
            if self.debug:
                print('skipped augmenting')
            return img

    def setup(self, imgAug: dict, markerSize: int = 6, totalMarkers: int = 250, debug: bool = True, cam: int = 0, displayName: str = 'Augmented by Sarang'):
        if not isinstance(cam, int):
            raise TypeError("Expected an int")
        if not isinstance(imgAug, dict):
            raise TypeError("Expected a dict")
        if not isinstance(markerSize, int) and not isinstance(totalMarkers, int):
            raise TypeError("Expected an int")
        if not isinstance(debug, bool):
            raise TypeError("Expected a bool")
        cap = cv2.VideoCapture(cam)
        self.cap = cap
        self.imgAug = imgAug
        self.displayName = displayName
        self.markersize = markerSize
        self.totalmarkers = totalMarkers
        self.debug = debug

    def start(self, display: bool = True):
        if not isinstance(display, bool):
            raise TypeError("Expected a bool")
        success, img = self.cap.read()
        arucofound = self.findArucoMakers(
            img, self.debug, self.markersize, self.totalmarkers)
        # looping through and auging each one
        if len(arucofound[0]) != 0:
            for bbox, id in zip(arucofound[0], arucofound[1]):
                if self.debug:
                    print(f'bounting box : {bbox}')
                    print(f'Marker id: {id}')
                img = self.augmentAruco(bbox, id, img, self.imgAug)
        cv2.waitKey(1)
        if display:
            cv2.imshow(self.displayName, img)
        return img
# the end
