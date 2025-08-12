import cv2  # OpenCVをインポート
from PIL import Image, ImageTk
import os

edge_image_cv = None  # エッジ検出後の画像を保持する変数


# filepath: e:\卒研関連\label_add.py
def set_picture_with_edges(file_name_entry, pic_canvas):
    """画像をセットし、エッジ検出結果を描画"""
    global edge_image, edge_image_cv
    file_path = file_name_entry.get()
    print(f"絶対パス: {file_path}")  # デバッグ用

    if file_path:
        # 絶対パスを相対パスに変換
        file_path = os.path.relpath(file_path)
        print(f"相対パス: {file_path}")  # デバッグ用

        pic_canvas.delete("all")

        # OpenCVで画像を読み込む
        img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            print(f"画像を読み込めません: {file_path}")
            return

        # エッジ検出
        edges = cv2.Canny(img_cv, 100, 200)  # Cannyエッジ検出
        edge_image_cv = edges  # エッジ検出後の画像をグローバル変数に保存
        edge_image = ImageTk.PhotoImage(image=Image.fromarray(edges))  # Tkinter用に変換

        # キャンバスのサイズを取得
        canvas_width = pic_canvas.winfo_width() or 1
        canvas_height = pic_canvas.winfo_height() or 1

        # エッジ画像をキャンバスに描画
        pic_canvas.create_image(
            canvas_width // 2, canvas_height // 2, anchor="center", image=edge_image
        )
