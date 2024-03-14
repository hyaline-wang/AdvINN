import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch
import json
import config as c
from args import get_args_parser
args = get_args_parser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.models == 'Resnet50':
    model = models.resnet50(pretrained=True)
if args.models == 'Inception_v3':
    model = models.inception_v3(pretrained=True)
if args.models == 'Densenet121':
    model = models.densenet121(pretrained=True)

with open('./util/imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

def index(i):
    class_idx = json.load(open("./util/imagenet_class_index.json"))
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    return class2label[i]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

def choose_target(number):
    # 本函数的model 要从这个文件的顶部看，通过加载已有的预训练网络，推理这个图片最可能是哪个class
    # 最不可能是哪个 class
    # 难道这个项目的目标是 让最不可能的变成最可能的
    open_dir = args.inputpath
    pic_dir = os.path.join(open_dir, number)
    pic = os.listdir(pic_dir)[0]
    path = os.path.join(pic_dir, pic)
    image = Image.open(path, 'r')
    image_t = transform(image).to(device)
    batch_t = torch.unsqueeze(image_t, 0)

    model.eval().to(device)
    out = model(batch_t)
    _, index = torch.max(out, 1)
    _, target = torch.min(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print("recent_label:", [classes[index[0]], percentage[index[0]].item()])
    print("target_label:", [classes[target[0]], percentage[target[0]].item()])

    class_idx = json.load(open("./util/imagenet_class_index.json"))
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    target = class2label[target[0]]
    print(f"target  {target}")
    return target











