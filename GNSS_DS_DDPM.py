import numpy as np
import matplotlib.pyplot as plt
import torch
from sol2res import sol2res,resmat2targrt_data
import torch.nn as nn

num_steps=100 #设定扩散步长T

#扩散超参数预定义
betas=torch.linspace(-6,6,num_steps)
betas=torch.sigmoid(betas)*(0.5e-2 - 1e-5) + 1e-5

#计算alpha及其相关超参
alphas=1-betas
alphas_prod=torch.cumprod(alphas,0)#alphas 0-T 的全部乘积序列
alphas_prod_p=torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)#alphas 0-(T-1)的全部乘积序列

alphas_bar_sqrt=torch.sqrt(alphas_prod)
one_minus_alphas_bar_log=torch.log(1-alphas_prod)
one_minus_alphas_bar_sqrt=torch.sqrt(1-alphas_prod)

def qx(x_0,t):
    '''基于x[0]得到任意时刻t的x[t]'''

    noise=torch.randn_like(x_0)      #正态分布随机噪声
    alphas_t=alphas_bar_sqrt[t]     #参数读取 
    alphas_1_m_t=one_minus_alphas_bar_sqrt[t]

    return (alphas_t*x_0+alphas_1_m_t*noise)  #输出t时刻前向加噪结果



#定义一个编码-解码器网络(MLP)
class MLPDiffusion(nn.Module):
    
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion,self).__init__()

        self.liners=nn.ModuleList(
            [
                nn.Linear(2,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,2),
            ]
        )

        self.step_embeddings=nn.ModuleList(
            [
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        )
    
    #带时间步位置编码的MLP
    def forward(self,x,t):
        for idx,embedding_layer in enumerate(self.step_embeddings):
            t_embedding=embedding_layer(t)
            x=self.liners[2*idx](x)
            x+=t_embedding
            x=self.liners[2*idx+1](x)
        x=self.liners[-1](x)
        
        return x

#GNSS_DS_DDPM训练loss函数    
def diffusion_loss(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps,device='cpu'):
    '''对任意时刻t进行采样并计算loss'''
    batch_size=x_0.shape[0]

    #随机采样一个时刻t, 假设为batch_size个, 用二分采样减少重复
    t=torch.randint(0,n_steps,size=(batch_size//2,))#在0-采样步上限内采样batch_size/2个t
    t=torch.cat([t,n_steps-1-t],dim=0)              #采样另一半
    t=t.unsqueeze(-1)

    #x0系数
    a=alphas_bar_sqrt[t]
    #eps系数
    aml=one_minus_alphas_bar_sqrt[t]

    #随机噪音eps:
    e=torch.randn_like(x_0)

    #构造模型输入
    x=x_0*a+e*aml

    #数据转移至GPU
    if(device!='cpu'):
        e=e.to(device)
        x=x.to(device)
        t=t.to(device)
    
    #送入网络模型, 得到t时刻随机噪声预测值
    output=model(x,t.squeeze(-1))

    #计算loss
    loss=(e-output).square().mean()
    
    #清除变量
    del e,x,t

    #与真实噪声一起计算误差, 求均值（MSE）
    return loss


#GNSS_DS_DDPM第一阶段采样
def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt,device='cpu'):

    t=torch.tensor([t])

    coeff=betas[t]/one_minus_alphas_bar_sqrt[t]

    if(device!='cpu'):
        x=x.to(device)
        t=t.to(device)
    eps_theta = model(x,t)
    
    eps_theta =eps_theta.to('cpu')
    x=x.to('cpu')
    t=t.to('cpu')

    mean=(1/(1-betas[t]).sqrt()) * (x-(coeff*eps_theta))

    z=torch.randn_like(x)
    sigma_t=betas[t].sqrt()

    sample=mean+sigma_t*z
    
    return (sample)

def p_sample_loop(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt,device='cpu'):

    cur_x=torch.randn(shape)
    x_seq=[cur_x]
    for i in reversed(range(n_steps)):
        cur_x=p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt,device=device)
        x_seq.append(cur_x)
    return x_seq


#GNSS_diffision采样法2
def get_samples_full_2(model,min,max,ele_s,N=1024,time_step=100):
    x_seq=p_sample_loop(model,(5*N,2),time_step,betas,one_minus_alphas_bar_sqrt)
    x_seq_eles=x_seq[100].T[0].detach()*90
    x_seq_eles_int=(np.array(x_seq_eles)*100).astype(int)  #构建样本高度角列表为快速采样做准备
    samples=x_seq[100].T[1].detach()*(max-min)+min
    
    ele_s_int=np.array(ele_s,dtype=np.float64)
    ele_s_int=ele_s_int*100                      #构建目标列表快速采样
    sample_results=np.zeros(len(ele_s),dtype=np.float64)
    f_count=0
    f_counts=[]
    #进行第一次采样, 得到基础数据列表
    for i in range(len(ele_s)):
        ele=int(ele_s_int[i])#目标高度角
        try:
            id=list(x_seq_eles_int).index(ele)
            sample_results[i]=float(samples[id])
        except:
            f_count+=1
            f_counts.append(i)
    
    #print("初次循环采样失败数量",f_count)
    
    #开始循环,直至采样失败列表置空
    while(len(f_counts)):
        f_counts_old=f_counts.copy()
        f_counts=[]
        f_count=0
        x_seq=p_sample_loop(model,(5*N,2),time_step,betas,one_minus_alphas_bar_sqrt)
        x_seq_eles=x_seq[100].T[0].detach()*90
        x_seq_eles_int=(np.array(x_seq_eles)*100).astype(int)  #构建样本高度角列表为快速采样做准备
        samples=x_seq[100].T[1].detach()*(max-min)+min
        
        for i in f_counts_old:
            ele=int(ele_s_int[i])#目标高度角
            try:
                id=list(x_seq_eles_int).index(ele)
                if(sample_results[i]==0.0):
                    sample_results[i]=float(samples[id])
            except:
                f_count+=1
                f_counts.append(i)
        #print("次循环采样失败数量",f_count)
    
    return sample_results