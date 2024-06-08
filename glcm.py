from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2

def calc_glcm_all_agls(img, label, props, distances=[1,2,3,4,5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image,
                        distances=distances,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    for dist in range(0, len(distances)):
        glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[dist]]
        for item in glcm_props:
            feature.append(item)
    
    feature.append(label)
    return feature


