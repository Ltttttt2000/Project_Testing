"""
make the result folder to save results
"""
import os
try:
    os.mkdir("result")
    os.mkdir("result/inpainting")
    os.mkdir("result/inpainting/cropped")
    os.mkdir("result/inpainting/real")
    os.mkdir("result/inpainting/recon")
    os.mkdir("result/deblur")
    os.mkdir("result/deblur/blur")
    os.mkdir("result/deblur/real")
    os.mkdir("result/deblur/deblur")
except Exception as e:
    print(e)