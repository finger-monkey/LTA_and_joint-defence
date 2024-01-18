import torch
import torch.nn as nn
from torchvision import transforms
from multimodal import *
from ..attack import Attack
from defendAugment import *
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

class PGD(Attack):
    def __init__(self, model, eps=5/255.0, alpha=1/255.0, steps=15, random_start=False, targeted=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.sign = 1
        if targeted:
            self.sign = -1

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=-1, max=1)

        for i in range(self.steps):
            adv_images.requires_grad = True

            q_feature1 = extract_feature_img(self.model, adv_images)
            q_feature2 = extract_feature_img(self.model, images)
            cost = criterion(q_feature1, q_feature2).to(self.device)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        return adv_images
