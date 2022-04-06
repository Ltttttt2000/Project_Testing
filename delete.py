"""
.DS_Store
"""
import os

# path = '/Users/tl/Downloads/Inpainting_withSSIM/recon'
path = '/Users/tl/Downloads/Deblur_100/real'
imageSize = 256
for image in os.listdir(path):
    file = os.path.join(path, str(image))
    # print(image.endswith(".jpg"))
    if image.endswith('.png') == False:
        print(file)
        os.remove(file)
