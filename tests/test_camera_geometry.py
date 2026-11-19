import numpy as np
from control_panel.camera_geometry import center_crop_reference


def test_hd_center_crop_preserves_the_calibrated_reference_field():
    frame = np.zeros((1080,1920,3),dtype=np.uint8)
    frame[540,960] = [17,29,41]
    cropped = center_crop_reference(frame)
    assert cropped.shape == (1080,1440,3)
    assert np.array_equal(cropped[540,720],frame[540,960])
    assert np.shares_memory(cropped,frame)
    # The original 640x480 mode matches this centered ROI at 2.25x scale.
    for x,y in ((320,240),(315.776,247.344),(0,0),(640,480)):
        assert (x*2.25/1440,y*2.25/1080) == (x/640,y/480)


def test_reference_frame_is_not_cropped_and_720p_keeps_center():
    reference=np.zeros((480,640,3),dtype=np.uint8)
    assert center_crop_reference(reference) is reference
    wide=np.zeros((720,1280,3),dtype=np.uint8)
    wide[360,640]=[1,2,3]
    cropped=center_crop_reference(wide)
    assert cropped.shape == (720,960,3)
    assert cropped[360,480].tolist()==[1,2,3]
