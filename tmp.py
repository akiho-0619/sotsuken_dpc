from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

import cv2

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
import sys

import pandas as pd

color_samples = pd.DataFrame(
    columns=[
        "x",
        "y",
        "hue",
        "saturation",
        "value",
        "std_hue",
        "std_saturation",
        "std_value",
    ]
)

color_data_list = []

print(datetime.now())
s = datetime.now()
# pic_path = r"E:\sotsuken\pictures\IMG20240717181328.jpg"
pic_path = r"E:\sotsuken\pictures\pic20250709.jpg"
# 画像をエッジ検出してエッジ画像を使う
img = cv2.imread(pic_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
edges.show()