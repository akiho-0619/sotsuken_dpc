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
# エッジ画像を3チャンネルに変換（SAMはRGB画像を想定）
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
pic = edges_rgb
pic = cv2.imread(pic_path)
# 縦長なら90°回転
if pic.shape[0] > pic.shape[1]:
    pic = cv2.rotate(pic, cv2.ROTATE_90_CLOCKWISE)

resized_img = cv2.resize(pic, (2000, 1500))

image_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
img_hsv_shape = image_hsv.shape
masks = mask_generator.generate(image_rgb)
# print(masks)
print(datetime.now() - s)


pic_x, pic_y = masks[0]["segmentation"].shape
print(masks[0]["segmentation"].shape)

import matplotlib.pyplot as plt
import numpy as np

# 元画像のコピーを作成
masked_image = image_rgb.copy()


import numpy as np


def is_point_in_ellipse(px, py, ellipse):  # 楕円中心が別の楕円内にあるか判定する関数
    (cx, cy), (a, b), angle = ellipse
    angle = np.deg2rad(angle)
    # 点を楕円の中心基準に平行移動
    dx = px - cx
    dy = py - cy
    # 回転を考慮して座標変換
    x_rot = dx * np.cos(angle) + dy * np.sin(angle)
    y_rot = -dx * np.sin(angle) + dy * np.cos(angle)
    # 標準化して楕円内か判定
    return (x_rot / (a / 2)) ** 2 + (y_rot / (b / 2)) ** 2 <= 1


ellipses = []  # 楕円のリストを初期化
# 各マスクを色付きで重ねる
ellipses_list = []

# def is_ellipse_inside_another(ellipse1, ellipse2):
#     """楕円1が楕円2の内側にあるかどうかを判定"""
#     (cx1, cy1), (a1, b1), angle1 = ellipse1
#     (cx2, cy2), (a2, b2), angle2 = ellipse2
#     
#     # 楕円1の中心が楕円2の内側にあるかチェック
#     if not is_point_in_ellipse(cx1, cy1, ellipse2):
#         return False
#     
#     # 楕円1の大きさが楕円2より小さいかチェック
#     area1 = np.pi * (a1/2) * (b1/2)
#     area2 = np.pi * (a2/2) * (b2/2)
#     
#     return area1 < area2

# def filter_nested_ellipses(ellipses_list):
#     """内側にある楕円を除去する"""
#     filtered_ellipses = []
#     
#     for i, ellipse1 in enumerate(ellipses_list):
#         is_inside = False
#         for j, ellipse2 in enumerate(ellipses_list):
#             if i != j and is_ellipse_inside_another(ellipse1, ellipse2):
#                 is_inside = True
#                 break
#         
#         if not is_inside:
#             filtered_ellipses.append(ellipse1)
#     
#     return filtered_ellipses


def main(num, mask):
    global masked_image, ellipses, color_data_list
    segmentation = mask["segmentation"]
    color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
    masked_image[segmentation] = masked_image[segmentation] * 0.5 + color * 0.5

    # --- 楕円近似の追加 ---
    # segmentationをuint8型に変換
    seg_uint8 = segmentation.astype(np.uint8) * 255

    # 前処理（例：膨張→収縮でノイズ除去）
    kernel = np.ones((3, 3), np.uint8)
    seg_uint8 = cv2.morphologyEx(seg_uint8, cv2.MORPH_OPEN, kernel)

    # --- 前処理の強化 ---
    # 1. 平滑化でノイズ除去
    seg_uint8 = cv2.GaussianBlur(seg_uint8, (5, 5), 0)

    # 2. モルフォロジー処理（開閉処理で小さなノイズ除去＆穴埋め）
    kernel = np.ones((5, 5), np.uint8)
    seg_uint8 = cv2.morphologyEx(seg_uint8, cv2.MORPH_OPEN, kernel)
    seg_uint8 = cv2.morphologyEx(seg_uint8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE  # 点を間引かず全て使う
    )
    if len(contours) > 0:
        # 面積が最大の輪郭を選択
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            ellipse = [
                [int(ellipse[0][0]), int(ellipse[0][1])],
                [int(ellipse[1][0]), int(ellipse[1][1])],
                int(ellipse[2]),
            ]
            if pic_x * 0.5 > ellipse[1][0] or pic_y * 0.5 > ellipse[1][1]:
                ellipses.append(ellipse)
            else:
                return

            # ---------------色データのサンプリング--------------------
            """
            positions = []
            this_color_samples = []
            min_width = min(ellipse[1][0], ellipse[1][1])
            while len(positions) < 10:
                x = min(
                    np.random.randint(
                        ellipse[0][0] - min_width // 2,
                        ellipse[0][0] + min_width // 2,
                    ),
                    img_hsv_shape[0] - 1,
                )
                y = min(
                    np.random.randint(
                        ellipse[0][1] - min_width // 2,
                        ellipse[0][1] + min_width // 2,
                    ),
                    img_hsv_shape[1] - 1,
                )
                if (
                    sqrt((x - ellipse[0][0]) ** 2 + (y - ellipse[0][1]) ** 2)
                    < min(ellipse[1]) // 2
                    or True
                ):
                    positions.append((x, y))
                    try:
                        this_color_samples.append(image_hsv[y, x])
                    except IndexError:
                        print(f"IndexError at position ({x}, {y})")

                if len(positions) >= 10:
                    break
            

            # ----------------サンプリングした色の平均と標準偏差を計算----------------
            average_hue = np.mean([col[0] for col in this_color_samples])
            average_saturation = np.mean([col[1] for col in this_color_samples])
            average_value = np.mean([col[2] for col in this_color_samples])
            std_hue = np.std([col[0] for col in this_color_samples])
            std_saturation = np.std([col[1] for col in this_color_samples])
            std_value = np.std([col[2] for col in this_color_samples])

            # ----------------データフレームに追加----------------
            color_data_list.append(
                {
                    "x": ellipse[0][0],
                    "y": ellipse[0][1],
                    "hue": average_hue,
                    "saturation": average_saturation,
                    "value": average_value,
                    "std_hue": std_hue,
                    "std_saturation": std_saturation,
                    "std_value": std_value,
                }
            )
            # color_samples = pd.concat([color_samples, tmp_df], ignore_index=True)

            # print(positions)
            # print(this_color_samples)
            # sys.exit()
            """

            # try:

            # for i in range(len(mask_sorted[num + 1 :])):
            #     print(*mask_sorted[i]["segmentation"][1])
            #     result = is_point_in_ellipse(
            #         *mask_sorted[i]["segmentation"][1], ellipse
            #     )
            #     if result:
            #         continue
            # ellipses.append(
            #     f"{ellipse[0][0]},{ellipse[0][1]},{ellipse[1][0]},{ellipse[1][1]},{ellipse[2]}\n"
            # )

            # except Exception as e:
            #     print(e, ellipse)

            # 楕円の描画は後でまとめて行う（重複チェック後）


with ThreadPoolExecutor(max_workers=20) as executor:
    mask_sorted = sorted(masks, key=lambda x: max(x["segmentation"][1]), reverse=True)
    futures = []
    for num, mask in enumerate(mask_sorted):
        future = executor.submit(main, num, mask)
        futures.append(future)
    
    # すべてのタスクの完了を待つ
    for future in futures:
        future.result()

# 内側にある楕円を除去
# print(f"フィルタリング前の楕円数: {len(ellipses)}")
# filtered_ellipses = filter_nested_ellipses(ellipses)
# print(f"フィルタリング後の楕円数: {len(filtered_ellipses)}")

# フィルタリングされた楕円を描画
# for ellipse in filtered_ellipses:
#     cv2.ellipse(masked_image, ellipse, (255, 0, 0), 2)

# フィルタリングなしで楕円を描画
for ellipse in ellipses:
    cv2.ellipse(masked_image, ellipse, (255, 0, 0), 2)


# with open("ellipses.csv", "w") as f:
#     f.writelines(ellipses)
plt.imshow(masked_image)
plt.axis("off")
plt.show()

zu_v = []
zu_s = []
with open(r"E:\sotsuken\color_samples.csv", "w") as f:
    for i in color_data_list:
        f.writelines(str(i) + "\n")
        zu_v.append(int(i["hue"]))
        zu_s.append(int(i["std_hue"]))

plt.scatter(zu_v, zu_s, c="red", s=1)
plt.show()
# print(color_data_list)
color_samples = pd.DataFrame(color_data_list)
color_samples.to_csv(r"E:\sotsuken\color_samples.csv", index=False)
