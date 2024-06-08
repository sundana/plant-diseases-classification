import cv2

def histogram(image, bins=32):
    hist_features = []
    for color_space in [cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB]:
        converted_img = cv2.cvtColor(image, color_space)
        for channel in range(3):
            hist = cv2.calcHist([converted_img], [channel], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)

    return hist_features