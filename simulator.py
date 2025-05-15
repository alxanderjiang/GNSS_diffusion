import numpy as np
import sys
sys.path.append('src')
from src.RINEX import decode_epoch_record,RINEX3_to_obsmat
from tqdm import tqdm

def find_target_res(PRN,gweek,gsec,sol_log,mode='plus'):
    target=[0.0,0.0,0.0,0.0]
    for i in range(len(sol_log)):
        try:
            d=sol_log[i][PRN]
            if(d['GPSweek']==gweek and d['GPSsec']==gsec):
                if(mode=='plus'):
                    target=[d['res_p1'],d['res_l1'],d['res_p2'],d['res_l2']]
                elif(mode=='mius'):
                    target=[-d['res_p1'],-d['res_l1'],-d['res_p2'],-d['res_l2']]
                break
        except:
            pass
    return target
def find_prns(gweek,gsec,sol_log):
    for i in range(len(sol_log)):
        prns=list(sol_log[i].keys())
        d=sol_log[i][prns[0]]
        if(d['GPSweek']==gweek and d['GPSsec']==gsec):
            break
    return prns

#反向策略仿真
target_prns=[]
sol_log=np.load("sim_data/LH21_WUH0610sol_log_sim.npy",allow_pickle=True)
obs_f="IGS_data/wuh20610.25o"

for i in range(len(sol_log)):
    prns_epoch=list(sol_log[i].keys())
    gweek=sol_log[i][prns_epoch[0]]['GPSweek']
    gsec=sol_log[i][prns_epoch[0]]['GPSsec']
    target_prns.append([gweek,gsec,prns_epoch.copy()])
    for prn in prns_epoch:
        sol_log[i][prn]['res_p2']=0.0
        sol_log[i][prn]['res_l1']=0.0
        sol_log[i][prn]['res_l2']=0.0
target_prns=np.array(target_prns,dtype=object)


header_copy=[]
data_simu=[]
prns_epoch=[]
rp_ids=[0,12,3,15]#此处输入双频伪距和载波相位在观测文件中的序号
#rp_ids=[0,1,3,4]
out_file=''
lambda_1=299792458.0/1575.42*1e6
lambda_2=299792458.0/1227.60*1e6
sim_sat=["G{:02d}".format(t) for t in range(33)]
print(sim_sat)
#sim_sat=["G07","G08","G22","G19"]
with open(obs_f,'r') as fobs:
    lines=fobs.readlines()
    header=0        
    for i in range(len(lines)):
        header_copy.append(lines[i])
        if("END OF HEADER" in lines[i]):
            header=i+1
            break
    index=0#初始化index
    for i in tqdm(range(header,len(lines))):
        if '>' in lines[i]:
            now=decode_epoch_record(lines[i])
            gweek=now['GPSweek']
            gsec=now['GPSsec']
            
            try:
                index=list(target_prns[:,1]).index(gsec)
            except:
                print(gsec,"Not in target_prns. Set gsec={}".format(target_prns[index,1]),)
            prns_epoch=target_prns[index,2]  
            #修改观测卫星数量
            s_num=decode_epoch_record(lines[i])['s_num']
            #print(lines[i][:32]+"{:3d}".format(len(prns_epoch))+lines[i][35:])
            #lines[i]=lines[i][:32]+lines[i][32:35].replace(str(s_num),str(len(prns_epoch)))+lines[i][35:]
            lines[i]=lines[i][:32]+"{:3d}".format(len(prns_epoch))+lines[i][35:]
            data_simu.append(lines[i])#首先拷贝引导行内容
        if '>' not in lines[i]:
            prn=lines[i][:3]
            res_target=find_target_res(prn,gweek,gsec,sol_log)
            if(prn not in prns_epoch):
                continue#未在目标卫星列表中,跳过
            try:
                C1=float(lines[i][3+rp_ids[0]*16:3+rp_ids[0]*16+14])
                C1_new=C1+float(res_target[0])
                if(prn not in sim_sat):
                    C1_new=C1_new-res_target[0]
                #print(lines[i],res_target,C1,C1_new)
                lines[i]=lines[i].replace('{:.3f}'.format(C1),'{:.3f}'.format(C1_new))
                #print(lines[i]) 
            except:
                pass
                
            try:
                L1=float(lines[i][3+rp_ids[1]*16:3+rp_ids[1]*16+14])
                L1_new=L1+float(res_target[1])/lambda_1
                lines[i]=lines[i].replace('{:.3f}'.format(L1),'{:.3f}'.format(L1_new))
            except:
                pass
                
            try:
                C2=float(lines[i][3+rp_ids[2]*16:3+rp_ids[2]*16+14])
                C2_new=C2+float(res_target[2])
                lines[i]=lines[i].replace('{:.3f}'.format(C2),'{:.3f}'.format(C2_new))
            except:
                pass

            try:
                L2=float(lines[i][3+rp_ids[3]*16:3+rp_ids[3]*16+14])
                L2_new=L2+float(res_target[3])/lambda_2
                lines[i]=lines[i].replace('{:.3f}'.format(L2),'{:.3f}'.format(L2_new))
            except:
                pass
            data_simu.append(lines[i])

#观测卫星数量校正
s_num_count=0
s_num_old=0
for i in range(len(data_simu)):
    if('>' in data_simu[i]):
        #更新卫星数量标识
        if(s_num_count!=s_num_old):
            #print(data_simu[id_snum])
            data_simu[id_snum]=data_simu[id_snum][:32]+data_simu[id_snum][32:35].replace(str(s_num_old),str(s_num_count))+data_simu[id_snum][35:]
            # print(data_simu[id_snum])
        #重置检验量
        id_snum=i
        s_num_count=0
        s_num_old=int(data_simu[i][32:35])
    if('>' not in data_simu[i]):
        s_num_count+=1
#文件写入
if(out_file==''):
    out_file=obs_f+'.sim'
with open(out_file,'w') as f:
    for h in header_copy:
        f.write(h)
    for d in data_simu:
        f.write(d)

print("反向策略仿真完成, 下面进行双反向策略处理原始环境误差观测数据")

# 双反向策略环境误差数据源重整
# 此处需要环境误差的原始观测值文件支持
# 误差源数据对齐, 请注意, 由于相关法律要求, 不直接开源环境误差数据的原始观测值,
# 如有科研需要, 请依据论文提供的联系方式联系作者
obs_f=obs_f+".sim"
obs_type=['C1C','L1C','D1C','S1C','C2W','L2W','D2W','S2W']
sol_log=RINEX3_to_obsmat(obs_f,obs_type)
target_prns=[]
for i in range(len(sol_log)):
    prns_epoch=[t['PRN'] for t in sol_log[i][1]]
    gweek=sol_log[i][0]['GPSweek']
    gsec=sol_log[i][0]['GPSsec']
    target_prns.append([gweek,gsec,prns_epoch.copy()])
target_prns=np.array(target_prns,dtype=object)

obs_f="lh210610.25o" #此处路径设置为环境误差的原始观测值
header_copy=[]
data_simu=[]
prns_epoch=[]
out_file=''
epoch_in=0
lambda_1=299792458.0/1575.42*1e6
lambda_2=299792458.0/1227.60*1e6
with open(obs_f,'r') as fobs:
    lines=fobs.readlines()
    header=0        
    for i in range(len(lines)):
        header_copy.append(lines[i])
        if("END OF HEADER" in lines[i]):
            header=i+1
            break
    
    for i in tqdm(range(header,len(lines))):
        if '>' in lines[i]:
            now=decode_epoch_record(lines[i])
            gweek=now['GPSweek']
            gsec=now['GPSsec']
            if(gsec not in list(target_prns[:,1])):
                epoch_in=0
                continue
            
            epoch_in=1
            index=list(target_prns[:,1]).index(gsec)
            prns_epoch=target_prns[index,2]  
            #修改观测卫星数量
            s_num=decode_epoch_record(lines[i])['s_num']
            lines[i]=lines[i][:32]+lines[i][32:35].replace(str(s_num),str(len(prns_epoch)))+lines[i][35:]
            data_simu.append(lines[i])#首先拷贝引导行内容
        if '>' not in lines[i]:
            if(not epoch_in):
                continue
            prn=lines[i][:3]
            res_target=find_target_res(prn,gweek,gsec,sol_log)
            if(prn not in prns_epoch):
                continue#未在目标卫星列表中,跳过
            data_simu.append(lines[i])

#观测卫星数量校正
s_num_count=0
s_num_old=0
for i in range(len(data_simu)):
    if('>' in data_simu[i]):
        #更新卫星数量标识
        if(s_num_count!=s_num_old):
            #print(data_simu[id_snum])
            data_simu[id_snum]=data_simu[id_snum][:32]+data_simu[id_snum][32:35].replace(str(s_num_old),str(s_num_count))+data_simu[id_snum][35:]
            # print(data_simu[id_snum])
        #重置检验量
        id_snum=i
        s_num_count=0
        s_num_old=int(data_simu[i][32:35])
    if('>' not in data_simu[i]):
        s_num_count+=1
#文件写入
if(out_file==''):
    out_file=obs_f+'.sim'
with open(out_file,'w') as f:
    for h in header_copy:
        f.write(h)
    for d in data_simu:
        f.write(d)

print("双反向策略环境误差数据处理完成")







