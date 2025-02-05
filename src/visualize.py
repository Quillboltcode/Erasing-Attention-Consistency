import re
import pprint
from model import Model
import torch
from torchvision import transforms
from utils import adapt_first_layer
from PIL import Image
from dataset import RafDataset
import utils
from tqdm import tqdm
import os
import cv2
# Mean: 0.5073918104171753, Std: 0.2121405303478241
MEAN = 0.5073918104171753
STD = 0.2121405303478241
file_names = []
with open('../log/18-01-24.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        match = re.search(r'Path: (.+)/(.+\.jpg)', line)
        if match:
            label = match.group(1).split('/')[-1]
            file_name = match.group(2)
            file_names.append(f"{label}/{file_name}")

pprint.pprint(file_names[0])

net = Model(None,pretrained=False)
net = adapt_first_layer(net)
net.load_state_dict(torch.load('../log/res_50_13_list_patition_label.txt.pth'))
net.eval()
# load the first image to model to get the attention map
# create a dataloader


img_name = file_names[0]
print(img_name)
for i, img_name in tqdm(enumerate(file_names)):
    utils.process_image(img_name, net, MEAN, STD, utils)