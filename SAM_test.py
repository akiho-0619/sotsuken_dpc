from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
# mask_generator = SamAutomaticMaskGenerator(sam)
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
sam = sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,  # デフォルトは32、必要に応じて調整
    points_per_batch=64,  # バッチサイズを調整
    pred_iou_thresh=0.98,  # IOU閾値を調整
    stability_score_thresh=0.95,  # 安定性スコアの閾値を調整
    crop_n_layers=0,  # クロップのレイヤー数を調整
    crop_n_points_downscale_factor=1,  # クロップのポイントダウンスケール係数を調整
    min_mask_region_area=3000,  # 小さい領域を無視
)

"""

better parameters for SAM
    sam,
    points_per_side=32,  # デフォルトは32、必要に応じて調整
    points_per_batch=64,  # バッチサイズを調整
    pred_iou_thresh=0.98,  # IOU閾値を調整
    stability_score_thresh=0.98,  # 安定性スコアの閾値を調整
    crop_n_layers=0,  # クロップのレイヤー数を調整
    crop_n_points_downscale_factor=1,  # クロップのポイントダウンスケール係数を調整
    min_mask_region_area=3000,  # 小さい領域を無視

pred_iou_thresh
説明: 予測マスク同士のIoU（重なり度）しきい値。これ以上重なるマスクは除外される。
効果: 値を下げると重複マスクが増える。上げるとユニークなマスクだけ残る。
デフォルト: 0.88
stability_score_thresh
説明: マスクの安定性スコアのしきい値。低いと不安定なマスクも残る。
効果: 値を下げるとノイズ的なマスクも残る。上げると信頼性の高いマスクだけ残る。
デフォルト: 0.95
crop_n_layers
説明: 画像を何段階でクロップ（分割）して細かい物体を検出するか。
効果: 0ならクロップなし。1以上で画像を分割し、細かい物体も検出しやすくなるが、計算コスト増。
デフォルト: 0
crop_n_points_downscale_factor
説明: クロップ時のグリッド縮小率。クロップ画像でのサンプリング密度を調整。
効果: 値を大きくするとクロップ画像でのグリッドが粗くなる。
デフォルト: 2
min_mask_region_area
説明: このピクセル数未満のマスクは除外される。
効果: 小さいノイズや細かい領域を除外できる。
デフォルト: 0


"""
import cv2

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
import sys

import pandas as pd
import os

import matplotlib.pyplot as plt
import numpy as np

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

simple_pics_pathes = [
    r"E:\sotsuken\simple_pics\carrot_marked.png",
    r"E:\sotsuken\simple_pics\tomato_marked.png",
    r"E:\sotsuken\simple_pics\nasu_marked.png",
]

print(datetime.now())
start = datetime.now()
# pic_path = r"E:\sotsuken\pictures\IMG20240717181328.jpg"
# pic_path = r"E:\sotsuken\Classification\Vegetable Images\train\Bean\0035.jpg"
pic_path = simple_pics_pathes[1]
print(os.path.basename(pic_path))
pic = cv2.imread(pic_path)
# pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)  # RGBに変換
hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = cv2.add(v, 1)  # 40だけ明るく（値は調整可、255を超えると255になる）
pic = cv2.cvtColor((cv2.merge((h, s, v))), cv2.COLOR_HSV2RGB)

# ---前処理---
# pic = cv2.GaussianBlur(pic, (5, 5), 0)  # ノイズ除去のための平滑化-ガウシアンブラー
# pic = cv2.medianBlur(pic, 5)  # メディアンブラーでさらにノイズ除去
# pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)  # グレースケール変換
# pic = cv2.equalizeHist(pic)  # ヒストグラム均等化でコントラスト調整 --多分ダメ
# pic = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(pic)  # コントラスト制御
# _, pic = cv2.threshold(pic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二値化
# pic = cv2.adaptiveThreshold(  # アダプティブ二値化
#     pic, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# )
# kernel = np.ones((3, 3), np.uint8)  #
# dilated = cv2.dilate(pic, kernel, iterations=1)
# pic = cv2.erode(thresh, kernel, iterations=1)
plt.imshow(pic)
plt.axis("off")
plt.show()
# 縦長なら90°回転
if pic.shape[0] > pic.shape[1]:
    pic = cv2.rotate(pic, cv2.ROTATE_90_CLOCKWISE)

resized_img = cv2.resize(pic, (2000, 1500))

# image_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
# image_hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
# img_hsv_shape = image_hsv.shape
# masks = mask_generator.generate(image_rgb)
# resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
masks = mask_generator.generate(resized_img)
# print(masks)
print(datetime.now() - start)


pic_x, pic_y = masks[0]["segmentation"].shape
print(masks[0]["segmentation"].shape)


# 元画像のコピーを作成
masked_image = resized_img.copy()


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
def is_mask_rectangle(segmentation):
    # seg_uint8 = segmentation.astype(np.uint8) * 255
    # contours, _ = cv2.findContours(
    #     seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )
    # for cnt in contours:
    #     epsilon = 0.02 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     if len(approx) == 4:  # 頂点が4つ
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         area_cnt = cv2.contourArea(cnt)
    #         area_rect = w * h
    #         if area_cnt / area_rect > 0.9:  # 面積比が高い
    #             return True
    x, w = 0, 0
    is_rect = False
    is_split = False
    for i in segmentation:
        if sum(i) == 0:
            continue

        if (x, w) == (0, 0):
            for num, j in enumerate(i):
                if j:
                    x = num if x == 0 else x
                    w += 1

        else:
            if False:
                pass

    return False


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
            """
            for cnt in contours:
                # 輪郭を多角形近似
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) == 4:  # 頂点が4つなら矩形
                    x, y, w, h = cv2.boundingRect(cnt)
                    area_cnt = cv2.contourArea(cnt)
                    area_rect = w * h
                    # 面積比が高い（矩形に近い）ものだけ抽出
                    if area_cnt / area_rect > 0.8:
                        print(f"矩形: 位置=({x},{y}), サイズ=({w}x{h})")
            """


print(len(masks))
rect_masks = [
    m for m in masks if m["area"] == (m["bbox"][2] + 1) * (m["bbox"][3] + 1)
]  # is_mask_rectangle(m["segmentation"])
print(f"矩形マスクの数: {len(rect_masks)}")
for m in rect_masks:
    print("masks: ", m["bbox"], m["area"])


with ThreadPoolExecutor(max_workers=20) as executor:
    mask_sorted = sorted(masks, key=lambda x: max(x["segmentation"][1]), reverse=False)
    futures = []
    # print(mask_sorted[0])
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
