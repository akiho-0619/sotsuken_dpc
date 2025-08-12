from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# SAMモデルの準備
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
sam = sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,  # デフォルトは32、必要に応じて調整
    min_mask_region_area=100,  # 小さい領域を無視
)

img_dir = r"E:\sotsuken\Classification\Vegetable Images"
out_dir = r"E:\sotsuken\Classification\Vegetable Images Coco"
os.makedirs(out_dir, exist_ok=True)


def coco(dir_path):
    attr_dir = os.path.join(out_dir, *dir_path[1:])
    os.makedirs(attr_dir, exist_ok=True)

    # print(f"Processing directory: {os.path.join(*dir_path)}")

    for img_name in os.listdir(os.path.join(*dir_path)):
        if os.path.splitext(img_name)[1].lower() not in [".jpg", ".jpeg", ".png"]:
            print(f"Skipping non-image file: {img_name}")
            continue

        img_path = os.path.join(*dir_path, img_name)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img_rgb)
        label_lines = []

        print(f"Processing {img_path} with {len(masks)} masks")

        for mask in masks:
            # マスクからバウンディングボックス取得
            y_indices, x_indices = np.where(mask["segmentation"])
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            x1, x2 = x_indices.min(), x_indices.max()
            y1, y2 = y_indices.min(), y_indices.max()
            # YOLO形式に変換
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            class_id = 0  # クラスIDは0（全て同じ）や手動で割り当て
            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n"
            )
        # ラベルファイル保存
        label_path = os.path.join(
            out_dir, *dir_path[1:], os.path.splitext(img_name)[0] + ".txt"
        )
        with open(label_path, "w") as f:
            f.writelines(label_lines)


if __name__ == "__main__":
    # with ProcessPoolExecutor(max_workers=3) as executor:
    for attr in ["train", "validation", "test"]:
        for sub_attr in os.listdir(os.path.join(img_dir, attr)):  # class name
            # executor.submit(coco, [img_dir, attr, sub_attr])
            coco([img_dir, attr, sub_attr])
    # executor.submit(coco, [img_dir, "train", "Bean"])
    # coco([img_dir, "train", "Bean"])  # デバッグ用に特定のクラスだけ処理

# for img_name in os.listdir(img_dir):
#     if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
#         continue
#     img_path = os.path.join(img_dir, img_name)
#     img = cv2.imread(img_path)
#     img_h, img_w = img.shape[:2]
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     masks = mask_generator.generate(img_rgb)
#     label_lines = []
#     for mask in masks:
#         # マスクからバウンディングボックス取得
#         y_indices, x_indices = np.where(mask["segmentation"])
#         if len(x_indices) == 0 or len(y_indices) == 0:
#             continue
#         x1, x2 = x_indices.min(), x_indices.max()
#         y1, y2 = y_indices.min(), y_indices.max()
#         # YOLO形式に変換
#         x_center = ((x1 + x2) / 2) / img_w
#         y_center = ((y1 + y2) / 2) / img_h
#         bw = (x2 - x1) / img_w
#         bh = (y2 - y1) / img_h
#         class_id = 0  # クラスIDは0（全て同じ）や手動で割り当て
#         label_lines.append(
#             f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n"
#         )
#     # ラベルファイル保存
#     label_path = os.path.join(out_dir, os.path.splitext(img_name)[0] + ".txt")
#     with open(label_path, "w") as f:
#         f.writelines(label_lines)
