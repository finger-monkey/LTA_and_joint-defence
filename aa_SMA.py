# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import yaml
import math
from model import ft_net, ft_net_dense
import os.path as osp
import torchattacks
from PIL import Image
import cv2
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../data/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')#To save the adversarial sample code only supports batchsize=1
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()

save_root_path = './adv_data/SMA_to_attack_' + opt.name

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
mkdir_if_missing(save_root_path)
toImage = transforms.ToPILImage()
def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.squeeze()
    for i in range(len(mean)):
        input_tensor[i] = input_tensor[i] * std[i] + mean[i]
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(
        0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_tensor = input_tensor.astype(np.uint8)
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)
    return input_tensor

###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)

opt.use_dense = config['use_dense']

opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
])

data_dir = test_dir

image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                         shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network
######################################################################
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature2(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    total = 0
    _, query_label,name = get_id2(query_path)
    for data in dataloaders:
        img, label = data
        real_label =  name[total][0:4]
        # ------------------Generating adversarial examples----------------------------------
        img = SMA(img, label)
        path = save_root_path + '/' + real_label + '/'
        mkdir_if_missing(path)
        save_path = path + name[total]
        img = img.cpu()
        img = toImage(img[0])
        img.save(save_path)
        print('{} --> adversarial example have been saved'.format(save_path))
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        )
        img = trans(img)
        img = img.reshape(1,3,256,128)
        #------------------------------------------------------------------------
        total += 1
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

    features = torch.cat((features,ff.data.cpu()), 0)
    print('features.size():',features.size())

    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        # print('filename:',filename)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def get_id2(img_path):
    camera_id = []
    labels = []
    name = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        # print('filename:',filename)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
        name.append(filename)
    return camera_id, labels,name
#-----------------------------------------------------------------------------------------------------------

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)


######################################################################
# Load Collected data Trained model
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()
##new add
for p in model.parameters():
    p.requires_grad = False

print('================>Init Attack...<===================')

SMA = torchattacks.PGD(model, eps=5/255, alpha=1/255,steps=15, random_start=True)

# Extract feature
query_feature = extract_feature2(model,dataloaders['query'])

