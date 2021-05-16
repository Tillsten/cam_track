from cam_track.cam_model import MockCam

def test_cam():
    cam = MockCam()
    cam.read_cam()
    cam.get_com()

def test_fit():
    cam = MockCam()
    cam.read_cam()
    cam.fit_gauss()