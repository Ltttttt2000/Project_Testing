"""
Deblur_L1*1 without SSIM
blur, real
psnr = 19.187013288793754
ssim = 0.49781219905939533
deblur, real
psnr = 16.729033935306944
ssim = 0.29266666319447676


Deblur_L1*1 with SSIM
real, deblur
psnr = 16.986884386203258
ssim = 0.3017874094478562

real, blur
psnr = 19.30605358818903
ssim = 0.5041748153711625


Inpainting_withSSIM
real, recon
psnr = 26.94019802066839
ssim = 0.9168741451765728

real, cropped
psnr = 7.5986315941259015
ssim = 0.7479973958115127

"""
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as sk_cpt_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
path1 = '/Users/tl/Downloads/Inpainting_withSSIM/CelebA/real'
path2 = '/Users/tl/Downloads/Inpainting_withSSIM/CelebA/cropped'

ssim_list = []
psnr_list = []
i = 0
for img in os.listdir(path2):
    output_path = path2 + "/" + img
    input_path = path1 + "/" + img
    output = Image.open(output_path).convert('RGB')
    input = Image.open(input_path).convert('RGB')
    img1 = np.array(input)
    img2 = np.array(output)  # (256, 256, 3)

    psnr_list.append(psnr(img1, img2))
    ssim_list.append(sk_cpt_ssim(img1, img2, multichannel=True))
    i += 1

    # print(psnr(img1, img2))  # 52.26561389555605 numpy.ndarray
    # print(sk_cpt_ssim(img1, img2, multichannel=True))

print(sum(ssim_list)/len(ssim_list))
print(sum(psnr_list)/len(psnr_list))