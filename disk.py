import numpy as np
import sys

def makeMask(R, Corner):
    background = np.ones((R, R), np.uint8)
    for i in range(R):
        _r = np.min([i, R - 1 - i])
        if _r > Corner:
            continue
        for j in range(R):
            _c = np.min([j, R - 1 - j])
            if _c > Corner:
                continue
            if _r + _c < Corner:
                background[i, j] = 0
    return background

def disk(r):
    if r == 8:
        return makeMask(15, 4)
    elif r == 7:
        return makeMask(13, 2)
    elif r == 6:
        return makeMask(11, 2)
    elif r == 5:
        return makeMask(9, 2)
    elif r == 4:
        return makeMask(7, 2)
    elif r == 3:
        return makeMask(5, 0)
    elif r == 2:
        return makeMask(5, 2)
    elif r == 1:
        return makeMask(3, 1)

    else:
        sys.exit("Too large disk")