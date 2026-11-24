"""Find the center of a balloon's circular projection in the camera image."""

import numpy as np


def fit_circle(points, minimum_radius, maximum_radius):
    if len(points) < 12:
        return None
    # Compare circles through triples of boundary points so a straight edge
    # from an occluder does not pull the fit away from the curved outline.
    samples = np.random.default_rng(0).integers(len(points), size=(128, 3))
    a, b, c = points[samples].transpose(1, 0, 2)
    u, v = b - a, c - a
    determinant = 2 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    valid = abs(determinant) > 1e-6
    u, v, a, determinant = u[valid], v[valid], a[valid], determinant[valid]
    uu, vv = np.sum(u * u, axis=1), np.sum(v * v, axis=1)
    offset = np.column_stack((uu * v[:, 1] - vv * u[:, 1],
                              vv * u[:, 0] - uu * v[:, 0])) / determinant[:, None]
    radii = np.linalg.norm(offset, axis=1)
    valid = (radii >= minimum_radius) & (radii <= maximum_radius)
    centers, radii = (a + offset)[valid], radii[valid]
    if not len(radii):
        return None
    residual = abs(np.linalg.norm(points - centers[:, None, :], axis=2) - radii[:, None])
    inliers = residual <= 1.25 + 0.045 * radii[:, None]
    scores = (inliers * (1 - 0.25 * np.minimum(residual, 4))).sum(axis=1)
    for best in np.argsort(scores)[::-1]:
        arc = points[inliers[best]]
        if len(arc) < max(12, 0.5 * len(points)):
            continue
        dx, dy = (arc - centers[best]).T
        angles = np.sort(np.arctan2(dy, dx))
        if np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() <= np.pi:
            break
    else:
        return None
    # Refit over the accepted arc in translated coordinates for numerical stability.
    origin = centers[best]
    arc = arc - origin
    system = np.column_stack((2 * arc, np.ones(len(arc))))
    solution = np.linalg.lstsq(system, np.sum(arc * arc, axis=1), rcond=None)[0]
    return solution[:2] + origin


def visible_center(frame, box):
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    left, top = max(0, int(x1) - 3), max(0, int(y1) - 3)
    right, bottom = min(width, int(x2) + 4), min(height, int(y2) + 4)
    if min(x2 - x1, y2 - y1, right - left, bottom - top) < 9:
        return None
    roi = frame[top:bottom, left:right]
    scale = min(1.0, 80 / max(roi.shape[:2]))
    if scale < 1:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sx, sy = roi.shape[1] / (right - left), roi.shape[0] / (bottom - top)
    x, y = np.meshgrid((np.arange(roi.shape[1]) + 0.5) / sx + left,
                       (np.arange(roi.shape[0]) + 0.5) / sy + top)
    rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
    radius = ((x - x1 - rx) / rx) ** 2 + ((y - y1 - ry) / ry) ** 2
    inside = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
    mask = np.full(roi.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    mask[inside] = cv2.GC_PR_BGD
    mask[radius < 0.65] = cv2.GC_PR_FGD
    mask[radius < 0.07] = cv2.GC_FGD
    if not np.any(mask == cv2.GC_BGD) or not np.any(mask == cv2.GC_FGD):
        return None
    background, foreground = (np.zeros((1, 65), np.float64) for _ in range(2))
    cv2.grabCut(roi, mask, None, background, foreground, 1, cv2.GC_INIT_WITH_MASK)
    silhouette = np.uint8((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
    _, components, stats, _ = cv2.connectedComponentsWithStats(silhouette)
    component = components[int((y1 + ry - top) * sy), int((x1 + rx - left) * sx)]
    coverage = stats[component, cv2.CC_STAT_AREA] / (4 * rx * ry * sx * sy)
    # Restore equal horizontal and vertical units after resizing the crop.
    units = (scale / sx, scale / sy)
    radii = (0.7 * min(rx, ry) * scale, 1.3 * max(rx, ry) * scale)
    circle = None
    if component and 0.30 <= coverage <= 0.94:
        contours, _ = cv2.findContours(np.uint8(components == component), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        pixels = max(contours, key=cv2.contourArea).reshape(-1, 2)
        circle = fit_circle((pixels + 0.5) * units, *radii)
    if circle is None:
        # Recover the outline directly if reflections break the foreground mask.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 15, 45)
        ys, xs = np.nonzero((edges != 0) & (radius >= 0.40) & (radius <= 1.55))
        circle = fit_circle((np.column_stack((xs, ys)) + 0.5) * units, *radii)
    if circle is None:
        return None
    cx, cy = circle / scale + (left, top)
    return [float(cx), float(cy)]
