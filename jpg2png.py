import os
from PIL import Image

d_path = "simple_pics"
for i in os.listdir(d_path):
    if i.endswith(".jpg"):
        img = Image.open(os.path.join(d_path, i))
        img.save(os.path.join(d_path, i[:-4] + ".png"), "PNG")
