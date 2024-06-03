from skimage.feature import graycomatrix, graycoprops
import numpy as np

def calc_glcm_all_agls(img, label, props, dists=[1], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    glcm = graycomatrix(
        img,
        distances=dists,
        angles=agls,
        levels=lvl,
        symmetric=sym,
        normed=norm
    )
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    feature.append(label)
    return feature


