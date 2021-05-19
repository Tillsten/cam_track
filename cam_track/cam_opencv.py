import numpy as np
import cv2 as cv

from cam_track.cam_model import Cam

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


class OpenCVCam(Cam):

    def init_cam(self):
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv.CAP_PROP_EXPOSURE, -7.0)
        cap.set(cv.CAP_PROP_GAIN, 0)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        self.cap = cap

    def read_cam(self) -> np.ndarray:
        # Capture frame-by-frame
        ret, frame = self.cap.read()
         # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return
         # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = rebin(gray, (120, 160))
        self.last_image = gray
        
        return gray