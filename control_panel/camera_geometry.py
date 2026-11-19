"""Preserve the calibrated 4:3 field when UVC delivers a wider HD frame."""
def center_crop_reference(frame):
    height, width = frame.shape[:2]
    if width * 3 > height * 4:
        crop_width = height * 4 // 3
        left = (width - crop_width) // 2
        return frame[:, left:left+crop_width]
    if width * 3 < height * 4:
        crop_height = width * 3 // 4
        top = (height - crop_height) // 2
        return frame[top:top+crop_height]
    return frame
