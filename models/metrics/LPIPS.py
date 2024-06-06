import lpips

import torch
import os
import cv2
import numpy as np
from PIL import Image


# 加载预训练的LPIPS模型
lpips_model = lpips.LPIPS(net="alex").cuda()
# 假设已经有了要计算LPIPS距离的两张图片 image1 和 image2
# 加载图像文件
folder_path_gt = './/RSSCN7//RSSCN7_test_HR//'
folder_path_res = './/result//De-SR_RSSCN7//'

lpips_total = 0
num = 0

for i in os.listdir(folder_path_res):
    # Load images
    img_res = lpips.im2tensor(lpips.load_image(folder_path_res + i))  # RGB image from [-1,1]
    img_gt = lpips.im2tensor(lpips.load_image(folder_path_gt + i))

    img_res = img_res.cuda()
    img_gt = img_gt.cuda()
    #     img_res = Image.open(folder_path_res+i)
    #     img_gt = Image.open(folder_path_gt+i)
    #     # 将图像转换为PyTorch的Tensor格式

    #     img_res = torch.tensor(np.array(img_res)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    #     img_gt = torch.tensor(np.array(img_gt)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 使用LPIPS模型计算距离
    distance = lpips_model(img_res, img_gt)
    lpips_total += distance.item()
    num += 1
    print("lpips:", distance.item())

this_lpips = lpips_total / num
print("LPIPS distance:", this_lpips)
