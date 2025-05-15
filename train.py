from GNSS_DS_DDPM import *

##在这里设置模型训练的数据文件路径
res_path="env_data/lh210600.25o_envonly.out.npy"
##在这里设置模型训练结果和权重文件的保存路径
save_path="models/LH21models"
##在这里设置需要训练的卫星列表(默认对GPS全星座进行训练)
sat_list=["G{:02d}".format(t) for t in range(1,33)]

#在这里设置训练参数
batch_size=1024
num_epoch=10000
seed=1234#随机数种子

'''配置完成后, 控制台运行train.py执行训练'''

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Trainning Model")

plt.rc('text',color='blue')

resmat=sol2res(res_path)
tag_prns=list(resmat.keys())

loss_info={}
res_max_min_info={}

for target in sat_list:
    print("Now Processing:",target)
    try:
        datasets,res_max,res_min,_=resmat2targrt_data(resmat,target,scale=2)
    except:
        print("No vailed data for ",target)
        continue
    
    loss_info[target]=[]
    res_max_min_info[target]=[res_max,res_min]
    
    dataloader=torch.utils.data.DataLoader(datasets,batch_size=batch_size,shuffle=True)
    model=MLPDiffusion(num_steps)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

    #模型搬移到GPU
    model=model.to(device)

    for t in range(num_epoch):
        for idx,batch_x in enumerate(dataloader):
            loss=diffusion_loss(model,batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps,device=device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
    
        #显示loss并绘制去噪采样图像
        loss_info[target].append(float(loss))
        if((t+1)%1000==0):
            #print(loss)
            x_seq=p_sample_loop(model,datasets.shape,num_steps,betas,one_minus_alphas_bar_sqrt,device=device)
            fig,axs=plt.subplots(1,10,figsize=(28,3))
            for i in range(1,11):
                cur_x=x_seq[i*10].detach()
                axs[i-1].scatter(cur_x[:,0],cur_x[:,1],color='red',edgecolor='white')
                axs[i-1].set_axis_off()
                axs[i-1].set_title("q{}".format(i*10))
            fig.savefig("{}/training_figs/{}_training_{}.png".format(save_path,target,int(t/1000)))
    #保存验证图片
    model=model.to('cpu')
    x_seq=p_sample_loop(model,(1024,2),100,betas,one_minus_alphas_bar_sqrt)
    fig=plt.figure(dpi=400,facecolor="white",figsize=(6,3))
    plt.title("Residuals Simulation of {}".format(target))
    plt.scatter(x_seq[100].T[0].detach(),x_seq[100].T[1].detach(),s=2)
    plt.axis('equal')
    plt.scatter(datasets[:,0],datasets[:,1],s=2)
    plt.legend(['Simu','True'])
    fig.savefig("{}/{}_evaluation.png".format(save_path,target))
    #保存模型权重
    torch.save(model,"{}/{}.pth".format(save_path,target))

np.save("{}/res_max_min_info.npy".format(save_path),res_max_min_info,allow_pickle=True)
np.save("{}/loss_info.npy".format(save_path),loss_info,allow_pickle=True)