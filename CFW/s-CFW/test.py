# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CFW import RFWToCFW


data = pd.read_excel('实验组装数据2_150kDA SF_chazhi.xlsx',header=None)
data = data.to_numpy()
data_ori = data[:,1::].T
w_ori = data[:,0:1].T

wtrunmin1 = w_ori[0,0]
wtrunmax1 = w_ori[0,-1]
wtrunmin2 = 1315
wtrunmax2 = 1900
wtrunmin3 = 1500
wtrunmax3 = 1800
tau = 10

[data_CFW, w_CFW] = RFWToCFW(data_ori, w_ori, wtrunmin1, wtrunmax1, wtrunmin2, wtrunmax2, wtrunmin3, wtrunmax3, tau)
plt.plot(w_CFW.T, data_CFW.T)

output = pd.DataFrame(np.vstack((w_CFW, data_CFW)).T)
output.to_excel('result2实验组装数据2_150kDA SF_chazhi.xlsx', index=False, header=None)