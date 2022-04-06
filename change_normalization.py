"""
I forgot unnormalize the images before save it as first.
So I Unnormalize photos in this file to show more clear results.
"""
import os

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])
toImage = transforms.ToPILImage()
normalize_path = '/Users/tl/Downloads/Inpainting_withoutSSIM/real'
for img in os.listdir(normalize_path):
    img_path = normalize_path + "/" + img
    input = Image.open(img_path).convert('RGB')
    output = transform(input)
    output = output * 0.5 + 0.5
    good = toImage(output)
    good.save(os.path.join(normalize_path, str(img)), "PNG", quality=100)



