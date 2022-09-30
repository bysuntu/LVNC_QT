import numpy as np
def normalizer01(f):
    fmin = np.min(f.astype(np.float))
    fmax = np.max(f.astype(np.float))
    Toleratance = 0.01
    if (fmax > Toleratance):
        return (f - fmin) / (fmax - fmin)
    else:
        return 0