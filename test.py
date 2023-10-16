import simplenet
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import cv2
from PIL import Image
import os
import ripplit as rp
import matplotlib.pyplot as plt
from datasets.mvtec import IMAGENET_MEAN, IMAGENET_STD
from main import net

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model(model_path, device):
    backbone_names = "wideresnet50",
    layers_to_extract = ["layer2", "layer3"]
    pretrain_embed_dimension = 1536
    target_embed_dimension = 1536
    patchsize = 3
    embedding_size= 256
    meta_epochs = 1
    aed_meta_epochs = 1
    gan_epochs = 1
    nosie_std = 0.015
    dsc_layers = 2
    dsc_hidden = 1024
    dsc_margin = 0.5
    dsc_lr = 0.0001
    auto_noise = 0
    train_backbone = False
    cos_lr = True
    pre_proj = 1
    proj_layer_type = 0
    mix_noise = 1

    model_function = net(backbone_names, layers_to_extract, pretrain_embed_dimension, target_embed_dimension, patchsize, embedding_size, meta_epochs, aed_meta_epochs, gan_epochs, nosie_std, dsc_layers, dsc_hidden, dsc_margin, dsc_lr, auto_noise, train_backbone, cos_lr, pre_proj, proj_layer_type, mix_noise)[1]
    model = model_function((3, 288, 288), device)[0]

    state_dict = torch.load(model_path, map_location=device)
    if 'discriminator' in state_dict:
        model.discriminator.load_state_dict(state_dict['discriminator'])
        model.discriminator.eval()
        if "pre_projection" in state_dict:
            model.pre_projection.load_state_dict(state_dict["pre_projection"])
            model.pre_projection.eval()
    else:
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    return model

def get_transforms(resize, imagesize):
    data_transforms = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return data_transforms

# Load the data
def load_data(data_dir):
    data_transforms = get_transforms(329, 288)
    images = []
    data = []
    files = os.listdir(data_dir)
    for file in files:
        img = Image.open(os.path.join(data_dir, file)).convert("RGB")
        img_trans = data_transforms(img)
        images.append(img)
        data.append(img_trans)
    data = torch.stack(data, dim=0)
    return data, images

# eval data
def eval_data(model_path, data_dir):
    model = load_model(model_path, device)
    data, images = load_data(data_dir)
    with torch.no_grad():
        image_scores, masks, features = model.predict(data)
    return image_scores, masks, features, images

def main():
    model_path = "/data/work/docker/SimpleNet/results/MVTecAD_Results/simplenet_mvtec/run/models/0/mvtec_bottle/ckpt.pth"
    data_dir = "/mnt/dataset/WY/mvtec_anomaly_detection/bottle/test/contamination"
    image_scores, masks, features, images = eval_data(model_path, data_dir)
    for i, (mask, img) in enumerate(zip(masks, images)):
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.savefig("test_{}.png".format(i))
    # plt.show()

if __name__ == "__main__":
    main()
    # import matplotlib
    # from matplotlib import pyplot as plt
    # # matplotlib.use('TkAgg')
    # plt.plot([1,2,3],[2,3,4])
    # plt.savefig("test.png")
    # plt.show()



