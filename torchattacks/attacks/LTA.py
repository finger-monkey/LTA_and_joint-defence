import torch
import torch.nn as nn
from ..attack import Attack
from collections import OrderedDict
from torchvision import transforms
from multimodal import *
class LoadFromFloder(torch.utils.data.Dataset):
    def __init__(self, name_list, transform=None):
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.name_list[idx])
        if self.transform != None:
            image = self.transform(image)
        sample = image

        return sample

def extract_feature_img(model, data, flip=False):
    img = data
    # Resize and  Normalize
    img = torch.nn.functional.interpolate(img, size=(256, 128), mode='bilinear', align_corners=False)
    img -= torch.cuda.FloatTensor([[[0.485]], [[0.456]], [[0.406]]])
    img /= torch.cuda.FloatTensor([[[0.229]], [[0.224]], [[0.225]]])

    f1 = model(img)
    if flip:
        flip_img = fliplr(img)
        f2 = model(flip_img, False)
        ff = f1 + f2
    else:
        ff = f1
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff / fnorm
    return ff


def criterion(f1s, f2s):
    ret = 0
    loss = torch.nn.MSELoss()
    for f1 in f1s:
        for f2 in f2s:
            ret += loss(f1, f2)
    return ret

def get_id(img_path):
    camera_id = []
    labels = []
    data_list = OrderedDict()
    for path, v in img_path:
        label, filename = path.split('/')[-2:]
        camera = filename.split('c')[1]
        labels.append(int(label))
        camera_id.append(int(camera[0]))
        if label in data_list:
            data_list[label].append(path)
        else:
            data_list[label] = [path]
    return camera_id, labels, data_list


class LTA(Attack):
    def __init__(self, model, eps=5/255.0, alpha = 1/255.0, steps = 15, momentum = 1.0 ):
        super(LTA, self).__init__("LTA", model)
        self.eps = eps
        self.model = model
        self.alpha = alpha
        self.max_iter = steps
        self.momentum = momentum

    def forward(self, images, labels):
        # print('max_iter=',max_iter,'epsilon=',self.eps)
        images = images.to(self.device)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            LGPR(1),
            transforms.ToTensor()
        ])
        lower_bound = images.data.cuda() - self.eps
        lower_bound[lower_bound < 0.0] = 0.0
        upper_bound = images.data.cuda() + self.eps
        upper_bound[upper_bound > 1.0] = 1.0
        x_adv = images

        x_adv.requires_grad = True
        grad = None
        for _ in range(self.max_iter):
            img_gray = images.clone().detach()
            img_gray = img_gray.cpu()
            img_gray[0] = transform(img_gray[0])
            img_gray = img_gray.to(self.device)
            q_feature = extract_feature_img(self.model, x_adv)
            g_feature = extract_feature_img(self.model, img_gray)
            g_feature.detach_()
            loss = criterion(q_feature, g_feature).to(self.device)
            loss.backward()
            # get normed x_grad
            x_grad = x_adv.grad.data
            norm = torch.mean(torch.abs(x_grad).view((x_grad.shape[0], -1)), dim=1).view((-1, 1, 1, 1))
            norm[norm < 1e-12] = 1e-12
            x_grad /= norm

            grad = x_grad if grad is None else self.momentum * grad + x_grad
            x_adv = x_adv.data + self.alpha * torch.sign(grad)

            x_adv = torch.max(x_adv, lower_bound)
            x_adv = torch.min(x_adv, upper_bound)
            x_adv.requires_grad = True
        return x_adv