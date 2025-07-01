# -*- coding: utf-8 -*-
'''
Input parameters
% data_ori                               Ndata x Nw_ori  original matrix
% w_ori                                  1 x Nw_ori  original vector, unit: cm^-1
% wtrunmin1                              input data lower limit, unit: cm^-1
% wtrunmax1                              input datal upper limit, unit: cm^-1
% wtrunmin2                              KKRelation-integral lower limit, unit: cm^-1
% wtrunmax2                              KKRelation-integral upper limit, unit: cm^-1
% wtrunmin3                              CFW-integral lower limit, unit: cm^-1
% wtrunmax3                              CFW-integral upper limit, unit: cm^-1
% tau                                    virtu gain, unit: cm^-1

Output parameters
% data_CFW                               Ndata x Nw_CFW  CFW result matrix
% w_CFW                                  1 x Nw_CFW  output vector, unit: cm^-1
% data_KK                                Ndata x Nw_KK  KKRelation result matrix
% w_KK                                   1 x Nw_KK  output vector, unit: cm^-1
'''

import numpy as np

def RFWToCFW(data_ori, w_ori, wtrunmin1, wtrunmax1,
             wtrunmin2, wtrunmax2, wtrunmin3, wtrunmax3, tau):
    
    bias = abs(np.min(data_ori, axis=1, keepdims=1))*1.5;
    data_bias = np.sqrt(1-(data_ori + bias));
    
    [data_KK, w_KK] = KKRelation_AmpToPhase(data_bias, w_ori, wtrunmin1, wtrunmax1, wtrunmin2, wtrunmax2)
    data_KK = (1-1/data_KK);
    
    wtrunmin3_ind = np.argmin(abs(w_KK-wtrunmin3))
    wtrunmax3_ind = np.argmin(abs(w_KK-wtrunmax3))
    dx = w_KK[0:1, 1::]-w_KK[0:1, 0:-1]
    len_dx = dx.shape[1]
    dx = np.hstack((dx[0:1, 0:1], (dx[0:1, 0:len_dx-1]+dx[0:1, 1:len_dx])/2, dx[0:1, len_dx-1:len_dx]))
    
    temp1 = w_KK[0:1, wtrunmin3_ind:wtrunmax3_ind+1]-1j*tau-w_KK.T
    t = np.linspace(10,15,11)/100*2*np.pi/4
    temp = t.reshape(1, 1, -1, order='F')
    temp2 = np.exp(1j*temp1[:, :, None]*temp)/(1j*temp1[:, :, None])
    temp2 = temp2[None, :, :, :]
    temp3 = data_KK
    
    data_CFW = temp2*temp3[:, :, None, None]*dx[:, :, None, None]/2/np.pi
    data_CFW = np.sum(data_CFW, axis=1)
    data_CFW = np.mean(data_CFW, axis=2)
    w_CFW = w_KK[0:1, wtrunmin3_ind:wtrunmax3_ind+1]
    
    data_CFW = 1-abs(1/(1-data_CFW))**2-bias
 
    return [data_CFW, w_CFW]



def KKRelation_AmpToPhase(data_bias, w_ori, wtrunmin1, wtrunmax1, 
                          wtrunmin2, wtrunmax2):
    
    wtrunmin1_ind = np.argmin(abs(w_ori-wtrunmin1))
    wtrunmax1_ind = np.argmin(abs(w_ori-wtrunmax1))
    wtrunmin2_ind = np.argmin(abs(w_ori-wtrunmin2))
    wtrunmax2_ind = np.argmin(abs(w_ori-wtrunmax2))
    
    logAbs = np.log(abs(data_bias))
    
    dx = w_ori[0:1, wtrunmin1_ind+1:wtrunmax1_ind+1]-w_ori[0:1, wtrunmin1_ind:wtrunmax1_ind]
    len_dx = dx.shape[1]
    dx = np.hstack((dx[0:1, 0:1], (dx[0:1, 0:len_dx-1]+dx[0:1, 1:len_dx])/2, dx[0:1, len_dx-1:len_dx]))
    
    temp1 = w_ori[0:1, wtrunmin2_ind:wtrunmax2_ind+1]-w_ori[0:1, wtrunmin1_ind:wtrunmax1_ind+1].T
    temp1 = temp1.reshape(1, temp1.shape[0], temp1.shape[1], order='F')     
    
    np.seterr(divide='ignore', invalid='ignore')
    phase_KK = logAbs[:, wtrunmin1_ind:wtrunmax1_ind+1, None]/temp1/np.pi*dx[:, :, None]
    phase_KK[np.isnan(phase_KK)] = 0
    phase_KK[np.isinf(phase_KK)] = 0
    phase_KK=np.sum(phase_KK, axis=1)

    data_KK = abs(data_bias[:, wtrunmin2_ind:wtrunmax2_ind+1])*np.exp(1j*phase_KK)
    w_KK = w_ori[0:1, wtrunmin2_ind:wtrunmax2_ind+1]
 
    return [data_KK, w_KK]