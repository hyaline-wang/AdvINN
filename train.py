import torch.nn.functional as F
import sys
from util.dataset import trainloader
from util import viz
import config as c
import modules.Unet_common as common
import warnings
from torchvision import models
import torchvision.transforms as transforms
from util.vgg_loss import VGGLoss
import time
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
from args import get_args_parser
from util.utils import *
from model.model import *

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################
# Model initialize: #
#####################
args = get_args_parser()
INN_net = Model().to(device)
init_model(INN_net)

INN_net = torch.nn.DataParallel(INN_net,device_ids=[0])
para = get_parameter_number(INN_net)
print(para)



params_trainable = (list(filter(lambda p: p.requires_grad, INN_net.parameters())))
optim1 = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
optim_init =optim1.state_dict()
dwt = common.DWT()
iwt = common.IWT()

class_idx = json.load(open("./util/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if args.models == 'Resnet50':
    model = nn.Sequential(
        norm_layer,
        models.resnet50(pretrained=True)
    ).to(device)
elif args.models == 'Inception_v3':
    model = nn.Sequential(
        norm_layer,
        models.inception_v3(pretrained=True)
    ).to(device)
elif args.models == 'Densenet121':
    model = nn.Sequential(
        norm_layer,
        models.densenet121(pretrained=True)
    ).to(device)
else:
    sys.exit("Please choose Resnet50 or Inception_v3 or Densenet121")
model = model.eval()

try:
    totalTime = time.time()
    vgg_loss = VGGLoss(3, 1, False)
    vgg_loss.to(device)
    failnum = 0
    count = 0.0
    print(f"trainloader length {len(trainloader)}")
    for i_batch, mydata in enumerate(trainloader):
        # 这里 i_batch 是一个 index 可以这样测试 ，print(f"i_batch {i_batch}")
        # mydata 是一个list，里面包裹了3 个 图像,可用以下语句测试
        print(f"mydata type:{type(mydata)} len:{len(mydata)}  item_0_shape:{mydata[0].shape}  item_1_shape:{mydata[1].shape} item_2_shape:{mydata[2].shape}")
        # mydata[0] 中是一个 torch.Size([1, 1, 3, 224, 224])，也就是3通道，244*244的尺寸，至于为什么有两个1需要往下看
        # 其中的一个1指的是 batch size 为1
        # mydata[1] 中是一个 torch.Size([1]) 后续查看中，发现含义为这个图片真实的class
        # mydata[2] 中是一个 torch.Size([1]) 后续查看中，发现含义为这个图片最不可能是 哪个 class

        start_time = time.time()

        X_1 = torch.full((1, 3, 224, 224), 0.5).to(device) #创建一个全为0.5的tensor，这张空图最后会变成
        X_ori = X_1.to(device)
        X_ori = Variable(X_ori, requires_grad=True) # 为了启用torch的自动求导，新版本已经取消
        optim2 = torch.optim.Adam([X_ori], lr=c.lr2)

        if c.pretrain:
            load(args.pre_model, INN_net) # 为网络加载预训练参数
            optim1.load_state_dict(optim_init) # ?


        data = mydata[0].to(device) 



        data = data.squeeze(0) 
        # squeeze 去掉一个维度，假设 data 是一个形状为 (1, 1, 3, 244, 244) 的张量，表示一个大小为 244x244 的 RGB 图片，
        # 你可以使用 data.squeeze(0) 来将其转换为一个形状为 (1,3, 244 244) 的张量。可用以下语句测试
        # print(f"after data.squeeze shape = {data.shape}")

        lablist1 = mydata[1]
        lablist2 = mydata[2]

        n1 = int(lablist1)
        n2 = int(lablist2)
        i1 = np.array([n1])
        i2 = np.array([n2])
        source_name = index(n1) # 真实class
        target_name = index(n2) # 希望变成的class

        labels = torch.from_numpy(i2).to(device)
        labels = labels.to(torch.int64).to(device) # 希望变成的class

        # print(f"data shape {data.shape}")
        cover = data.to(device)  # channels = 3
        

        cover_dwt_1 = dwt(cover).to(device)  # channels = 12
        # print(f"cover_dwt_1 shape {cover_dwt_1.shape}")
        # 原图像3个channels 都经过 dwt 后变为 12 个channels 分别为
        # 近似系数（low-low），是四个子图像的加和，代表图像的低频分量。  (3 channel)
        # 水平细节系数（high-low），是水平方向上的高频分量。  (3 channel)
        # 垂直细节系数（low-high），是垂直方向上的高频分量。 (3 channel)
        # 对角线细节系数（high-high），是水平和垂直方向上的高频分量。 (3 channel)


        # 提取dwt的低频分量 
        cover_dwt_low = cover_dwt_1.narrow(1, 0, c.channels_in).to(device)  # channels = 3
        
        if not os.path.exists(args.outputpath + source_name + "-" + target_name):
            os.makedirs(args.outputpath + source_name + "-" + target_name)

        # 保存原图
        save_image(cover, args.outputpath + source_name + "-" + target_name + '/cover.png')

        # 现在开始训练
        for i_epoch in range(c.epochs):
            #################
            #    train:   #
            #################

            # 将之前的空图 也做一次dwt，并提取低频分量 
            CGT = X_ori.to(device)
            CGT_dwt_1 = dwt(CGT).to(device)# channels =12
            CGT_dwt_low_1 = CGT_dwt_1.narrow(1, 0, c.channels_in).to(device)# channels =3

            # 将原图与 空图 的 dwt结果接到一起
            input_dwt_1 = torch.cat((cover_dwt_1, CGT_dwt_1), 1).to(device)  # channels = 12*2

            # 过网络
            # 输入 24 channel
            output_dwt_1 = INN_net(input_dwt_1).to(device)  # channels = 24


            output_steg_dwt_2 = output_dwt_1.narrow(1, 0, 4 * c.channels_in).to(device)  # channels = 12

            output_step_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in ).to(device)  # channels = 3
            output_steg_dwt_low_1 = output_steg_dwt_2.narrow(1, 0, c.channels_in).to(device)  # channels = 3
            # 将输出结果提取低频分量 

            # 取后12 chennel
            output_r_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in).to(device)
            # get steg ，逆变换，得到生成图
            output_steg_1 = iwt(output_steg_dwt_2).to(device)  # channels = 3

            # 逆变换
            output_r = iwt(output_r_dwt_1).to(device) # 用于最后的存储，特征

            output_steg_1 = torch.clamp(output_steg_1, min=0, max=1).to(device)

            # eta = torch.clamp(output_steg_1 - cover, min=-args.eps, max=args.eps)# 通过上下限现在生成图与原图的差异范围
            
            # output_steg_1 = torch.clamp(cover + eta, min=0, max=1)
            # 我们希望 output_steg_1 和原图一样，但是会被检测成完全不同的class

            #################
            #     loss:     #
            #################
            # 比较原图与生成结果是否相似
            g_loss = guide_loss(output_steg_1.cuda(), cover.cuda()).to(device)


            # 分别把原图和生成的图 扔到被截了一部分的 vgg中
            vgg_on_cov = vgg_loss(cover).to(device)
            vgg_on_steg_1 = vgg_loss(output_steg_1).to(device)
            # 计算感知差异，具体看lei Feifei 的论文
            perc_loss = guide_loss(vgg_on_cov, vgg_on_steg_1).to(device)
            # 计算 dwt低频分量 和生成的图的低频分量的差异
            l_loss = guide_loss(output_step_low_2.cuda(), cover_dwt_low.cuda()).to(device)

            out = model(output_steg_1 * 255.0).to(device) 
            _, pre = torch.max(out.data, 1)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 # pre 最可能是什么 ，percentage是概率
            
            _, indices = torch.sort(out, descending=True) # 按概率高低排序
            adv_cost = nn.CrossEntropyLoss().to(device)
            adv_loss = adv_cost(out, labels).to(device) # 计算预测值差异

            suc_rate = ((pre == labels).sum()).cpu().detach().numpy() #  suc_rate 只有 0和1 .

            total_loss = c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss + args.lamda_adv * adv_loss + c.lamda_per * perc_loss
            ii = int(pre)
            state = "img" + str(i_batch) + ":" + str(suc_rate)
            #################
            #     Exit:     #
            #################
            if suc_rate == 1:
                if (int(percentage[indices[0]][0]) >= 85):
                    print(state)
                    print([(class2label[idx], percentage[idx].item()) for idx in indices[0][:5]])
                    save_image(output_steg_1, args.outputpath + source_name + "-" + target_name +'/'+ str(i_epoch) + 'result.png')
                    output_r = normal_r(output_r)
                    save_image(output_r, args.outputpath + source_name + "-" + target_name + '/r.png')
                    count +=1
                    break
                if (i_epoch >= 2000):
                    print([(class2label[idx], percentage[idx].item()) for idx in indices[0][:5]])
                    save_image(output_steg_1, args.outputpath + source_name + "-" + target_name + '/'+
                               str(i_epoch) + "_" + str(int(percentage[indices[0]][0])) +'d_result.png')
                    output_r = normal_r(output_r)
                    save_image(output_r , args.outputpath + source_name + "-" + target_name + '/r.png')
                    count +=1
                    break
            if (i_epoch >= 5000):
                failnum += 1
                print([(class2label[idx], percentage[idx].item()) for idx in indices[0][:5]])
                save_image(output_steg_1 , args.outputpath + source_name + "-" + target_name + '/' +
                           str(i_epoch) + 'dw_result.png')
                output_r = normal_r(output_r)
                save_image(output_r , args.outputpath + source_name + "-" + target_name + '/r.png')
                count +=1
                break

            #################
            #   Backward:   #
            #################
            optim1.zero_grad()
            optim2.zero_grad()
            total_loss.backward()
            optim1.step()

            C_out = model(CGT * 255.0).to(device)
            C_adv_loss = adv_cost(C_out, labels).to(device)
            C_adv_loss.backward()
            optim2.step()

            weight_scheduler.step()
            lr_min = c.lr_min
            lr_now = optim1.param_groups[0]['lr']
            if lr_now < lr_min:
                optim1.param_groups[0]['lr'] = lr_min
        save_image(CGT , args.outputpath + source_name + "-" + target_name + '/CGT.png')
    totalstop_time = time.time()
    time_cost = totalstop_time - totalTime
    Total_suc_rate = (count-failnum)/count
    print("Total cost time :" + str(time_cost))
    print("Total suc rate :" + str(Total_suc_rate))
except:
    raise

finally:
    viz.signal_stop()
