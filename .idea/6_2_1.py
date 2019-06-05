import os
import numpy as np

fileName: str = "d:\java\data.txt"

if os.path.exists(fileName):
    wholeData = np.loadtxt(fileName, delimiter=",", dtype=np.int)
    print("wholeData(size: %d): \r\n %s" % (wholeData.size, wholeData))
else:
    print("There is not [%s]." % fileName)


