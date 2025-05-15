import numpy as np
import torch

def sol2res(sol_f,sys="GPS"):
    sol_log=np.load(sol_f,allow_pickle=True)
    res_mat={}
    if(sys=="GPS"):
        for i in range(1,33):
            res_mat["G{:02d}".format(i)]=[]
    if(sys=="BDS"):
        for i in range(1,66):
            res_mat["C{:02d}".format(i)]=[]
    for sol in sol_log:
        prns=list(sol.keys())
        for prn in prns:
            gweek=sol[prn]['GPSweek']
            gsec=sol[prn]['GPSsec']
            az=sol[prn]['azel'][0]
            el=sol[prn]['azel'][1]
            res_p1=sol[prn]['res_p1']
            res_p2=sol[prn]['res_p2']
            res_l1=sol[prn]['res_l1']
            res_l2=sol[prn]['res_l2']
            res_mat[prn].append([gweek,gsec,az,el,res_p1,res_p2,res_l1,res_l2])
    return res_mat

def resmat2targrt_data(resmat,target,scale=2,res_id="p1"):
    data_yes=0
    res_sign=["p1","p2","l1","l2"]
    res_t_id=res_sign.index(res_id)+4
    res_p1=np.array(resmat[target])
    #高度角归一化
    res_p1[:,3]=res_p1[:,3]/90.0
    #残差归一化(最值)
    print("Max before scaled: ",max(res_p1[:,res_t_id]),"Min before scaled",min(res_p1[:,res_t_id]))
    if(max(res_p1[:,res_t_id])>0.0):
        res_max=scale*max(res_p1[:,res_t_id])
    else:
        res_max=1/scale*max(res_p1[:,res_t_id])
    if(min(res_p1[:,res_t_id])<0.0):
        res_min=scale*min(res_p1[:,res_t_id])
    else:
        res_min=1/scale*min(res_p1[:,res_t_id])
    res_p1[:,4]=(res_p1[:,res_t_id]-res_min)/(res_max-res_min)

    data=np.array([res_p1[:,3],res_p1[:,res_t_id]]).T
    if(data.shape[0]%2!=0):
        data=np.vstack((data,data[-1]))
    datasets=torch.Tensor(data).float()
    print(target,"data shape:",data.shape," res_max:",res_max," res_min:",res_min)
    if(data.shape[0]!=0):
        data_yes=1
    return datasets,res_max,res_min,data_yes