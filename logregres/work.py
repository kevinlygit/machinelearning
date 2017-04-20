import random
import numpy as np
import matplotlib.pyplot as plt
import logregres

dataArr,labelMat=logregres.loadDataSet()
weights=logregres.stocGradAscent1(np.array(dataArr),labelMat)
logregres.plotBestFit(weights)

with open('output.out','w') as fw:
	fw.write(str(weights))