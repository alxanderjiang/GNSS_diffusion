import numpy as np
import matplotlib.pyplot as plt
import torch
from sol2res import sol2res,resmat2targrt_data
from GNSS_DS_DDPM import *
import torch.nn as nn

#此处用于生成sol_log_sim,设置指导仿真生成的残差数据与结果输出路径
from tqdm import tqdm
target_sol_log="env_data/lh210610.25o_envonly.out.npy"
model_path="D:/GNSSrecognition/GNSS_Diffusion/LH21models"
sol_log=np.load(target_sol_log,allow_pickle=True)
sat_list=["G{:02d}".format(t) for t in range(1,33)]
out_path="LH21_WUH0610sol_log_sim.npy"


#复制仿真log
sol_log_sim=[]
for i in range(len(sol_log)):
    prns=list(sol_log[i].keys())
    sol_log_sim.append({})
    for prn in prns:
        sol_log_sim[i][prn]={}
        sol_log_sim[i][prn]['GPSweek']=sol_log[i][prn]['GPSweek']
        sol_log_sim[i][prn]['GPSsec']=sol_log[i][prn]['GPSsec']
        sol_log_sim[i][prn]['res_p1']=0.0
        sol_log_sim[i][prn]['res_p2']=0.0
        sol_log_sim[i][prn]['res_l1']=0.0
        sol_log_sim[i][prn]['res_l2']=0.0
#训练集归一化上下界
max_min={}
max_min_info=eval(str(np.load("{}/res_max_min_info.npy".format(model_path),allow_pickle=True)))
for key in list(max_min_info.keys()):
    max_min[key]={}
    max_min[key]['max']=max_min_info[key][0]
    max_min[key]['min']=max_min_info[key][1]

for target in tqdm(sat_list):
    #加载仿真器模型
    try:
        model=torch.load("{}/{}.pth".format(model_path,target))
    except:
        print("No valid data and model for",target)
        continue
    #目标源数据
    resmat=sol2res(target_sol_log)[target]
    res_p1=np.array(resmat)
    data=np.array([res_p1[:,3],res_p1[:,4]]).T

    samples=get_samples_full_2(model,max_min[target]['min'],max_min[target]['max'],data[:,0])
    sim_count=0
    for i in range(len(sol_log)):
        try:
            sol_log_sim[i][target]['res_p1']=samples[sim_count]
            sim_count+=1
        except:
            pass

np.save(out_path,sol_log_sim,allow_pickle=True)