from tkinter import ttk, Tk, Label, Button, Canvas, Frame, Entry
from tkinter.ttk import Combobox
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2  # OpenCVをインポート


class_list = [os.path.splitext(i)[0] for i in os.listdir("classes")]
write_file = "classes/"
# クリックした座標を保持するリスト（ポリゴン用）
polygon_points = []

# 画像の拡大縮小と移動用の変数
scale_factor = 1.0
image_offset = [0, 0]  # [x_offset, y_offset]

dragging_point_index = None  # ドラッグ中の頂点のインデックス

is_edge_displayed = False  # エッジ画像が表示されているかどうかを保持するフラグ
original_image_cv = None  # 元の画像を保持する変数
edge_image_cv = None  # エッジ検出後の画像を保持する変数


def start_drag_point(event):
    """左クリックでドラッグ開始時に頂点を選択"""
    global dragging_point_index
    dragging_point_index = find_nearest_point(event.x, event.y)


def drag_point(event):
    """左クリックを押しながらドラッグで頂点を移動"""
    global dragging_point_index
    if dragging_point_index is not None:
        # 新しい座標を逆変換して保存
        x = (event.x - image_offset[0]) / scale_factor
        y = (event.y - image_offset[1]) / scale_factor
        polygon_points[dragging_point_index] = (x, y)
        print(f"Moved point to: ({x}, {y})")  # デバッグ用
        redraw_image()


def end_drag_point(event):
    """左クリックを離してドラッグ終了"""
    global dragging_point_index
    dragging_point_index = None


def delete_first_point(event):
    """BackSpaceキーでインデックス0の座標を削除"""
    global polygon_points
    if polygon_points:
        deleted_point = polygon_points.pop(0)  # インデックス0の座標を削除
        print(f"Deleted point: {deleted_point}")  # デバッグ用
        # 再描画
        pic_canvas.delete("polygon")
        if len(polygon_points) > 1:
            pic_canvas.create_polygon(
                *[coord for point in polygon_points for coord in point],
                outline="blue",
                fill="",
                tags="polygon",
            )


def zoom(event):
    """マウスホイールで画像を拡大縮小"""
    global scale_factor
    if event.delta > 0:  # ホイールを上に回すと拡大
        scale_factor *= 1.1
    elif event.delta < 0:  # ホイールを下に回すと縮小
        scale_factor /= 1.1
    redraw_image()


def key_zoom(event):
    """+/-キーで画像を拡大縮小"""
    global scale_factor
    if event.keysym == "plus":  # +キーで拡大
        scale_factor *= 1.1
    elif event.keysym == "minus":  # -キーで縮小
        scale_factor /= 1.1
    redraw_image()


def key_move(event):
    """矢印キーで画像を移動"""
    global image_offset
    if event.keysym == "Up":  # 上矢印キー
        image_offset[1] += 30
    elif event.keysym == "Down":  # 下矢印キー
        image_offset[1] -= 30
    elif event.keysym == "Left":  # 左矢印キー
        image_offset[0] += 30
    elif event.keysym == "Right":  # 右矢印キー
        image_offset[0] -= 30
    redraw_image()


def start_drag(event):
    """ドラッグ開始時の座標を記録"""
    pic_canvas.scan_mark(event.x, event.y)


def drag(event):
    """ドラッグ中の画像移動"""
    pic_canvas.scan_dragto(event.x, event.y, gain=1)


def redraw_polygon():
    # ポリゴンを再描画
    if len(polygon_points) > 1:
        transformed_points = [
            (
                x * scale_factor + canvas_center_x + image_offset[0],
                y * scale_factor + canvas_center_y + image_offset[1],
            )
            for x, y in polygon_points
        ]
        pic_canvas.create_polygon(
            *[coord for point in transformed_points for coord in point],
            outline="blue",
            fill="",
            tags="polygon",
        )

        # ポリゴンを最前面に移動
        pic_canvas.tag_raise("polygon")


def redraw_image():
    """画像を再描画（拡大縮小や移動を反映）"""
    global edge_image_cv, original_image_cv, scale_factor, image_offset, tk_img, is_edge_displayed, canvas_center_x, canvas_center_y
    if edge_image_cv is None or original_image_cv is None:
        print("画像がロードされていません。")
        return

    pic_canvas.delete("all")

    # 表示する画像を選択
    if is_edge_displayed:
        img_to_display = edge_image_cv
    else:
        img_to_display = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)

    # 画像をリサイズ
    height, width = img_to_display.shape[:2]
    new_width = max(1, int(width * scale_factor))
    new_height = max(1, int(height * scale_factor))
    resized_img = cv2.resize(
        img_to_display, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # Tkinter用に変換
    tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized_img))

    # 画像をキャンバスに描画
    canvas_center_x = pic_canvas.winfo_width() // 2
    canvas_center_y = pic_canvas.winfo_height() // 2
    pic_canvas.create_image(
        canvas_center_x + image_offset[0],
        canvas_center_y + image_offset[1],
        anchor="center",
        image=tk_img,
    )
    pic_canvas.image = tk_img  # 参照を保持してガベージコレクションを防ぐ

    # ポリゴンを再描画
    redraw_polygon()
    # if len(polygon_points) > 1:
    #     transformed_points = [
    #         (
    #             x * scale_factor + canvas_center_x + image_offset[0],
    #             y * scale_factor + canvas_center_y + image_offset[1],
    #         )
    #         for x, y in polygon_points
    #     ]
    #     pic_canvas.create_polygon(
    #         *[coord for point in transformed_points for coord in point],
    #         outline="blue",
    #         fill="",
    #         tags="polygon",
    #     )

    #     # ポリゴンを最前面に移動
    #     pic_canvas.tag_raise("polygon")


def save_segmentation():
    """YOLO-Seg形式でポリゴン座標を保存"""
    if len(polygon_points) < 3:
        print("ポリゴンは少なくとも3点必要です")
        return

    class_name = class_combo.get()
    if not class_name:
        print("クラスを選択してください")
        return

    file_path = write_file + class_name + ".txt"
    with open(file_path, "a") as f:
        # クラスID（仮に0とする）とポリゴン座標を保存
        try:
            class_id = class_list.index(class_name)

        except ValueError:
            class_list.append(class_name)
            class_id = class_list.index(class_name)
        polygon_str = " ".join([f"{x},{y}" for x, y in polygon_points])
        f.write(f"{class_id} {polygon_str}\n")
    print(f"保存しました: {file_path}")

    # ポリゴンをリセット
    polygon_points.clear()
    pic_canvas.delete("polygon")


def find_nearest_point(x, y, threshold=10):
    """クリック位置に最も近い頂点を探す"""
    for i, (px, py) in enumerate(polygon_points):
        if (
            abs(px * scale_factor + image_offset[0] - x) <= threshold
            and abs(py * scale_factor + image_offset[1] - y) <= threshold
        ):
            return i
    return None


def toggle_image(event):
    """エッジ画像と元の画像を切り替える"""
    global is_edge_displayed
    is_edge_displayed = not is_edge_displayed
    redraw_image()


def on_click(event):
    """左Ctrlキーでポリゴンの頂点を追加"""
    global polygon_points

    # マウスカーソルの現在位置を取得
    x, y = (
        pic_canvas.winfo_pointerx() - pic_canvas.winfo_rootx(),
        pic_canvas.winfo_pointery() - pic_canvas.winfo_rooty(),
    )

    # クリック位置を逆変換して元の座標系に保存
    x = (x - image_offset[0]) / scale_factor
    y = (y - image_offset[1]) / scale_factor

    # 既存の頂点が近ければ選択、それ以外は新しい頂点を追加
    nearest_index = find_nearest_point(x, y)
    if nearest_index is not None:
        print(f"Selected point: {polygon_points[nearest_index]}")  # デバッグ用
    else:
        polygon_points.append((x, y))
        print(f"Added point: ({x}, {y})")  # デバッグ用

    # ポリゴンを再描画
    redraw_image()


def set_picture_with_edges(file_name_entry, pic_canvas):
    """画像をセットし、エッジ検出結果を描画"""
    global edge_image_cv, original_image_cv, is_edge_displayed
    file_path = file_name_entry.get()
    print(f"絶対パス: {file_path}")  # デバッグ用

    if file_path:
        # 絶対パスを相対パスに変換
        file_path = os.path.relpath(file_path)
        print(f"相対パス: {file_path}")  # デバッグ用

        pic_canvas.delete("all")

        # OpenCVで画像を読み込む
        img_cv = cv2.imread(file_path)
        if img_cv is None:
            print(f"画像を読み込めません: {file_path}")
            return

        original_image_cv = img_cv  # 元の画像を保存
        gray_img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # グレースケール変換

        # エッジ検出
        edges = cv2.Canny(gray_img_cv, 100, 200)  # Cannyエッジ検出
        edge_image_cv = edges  # エッジ検出後の画像を保存

        # 初期状態ではエッジ画像を表示
        is_edge_displayed = True
        redraw_image()


def set_picture():
    file_path = file_name_entry.get()
    if file_path:
        pic_canvas.delete("all")

        # 画像をPillowで開く
        img = Image.open(file_path)

        # キャンバスのサイズを取得
        canvas_width = pic_canvas.winfo_width() or 1  # 幅が0の場合に1を設定
        canvas_height = pic_canvas.winfo_height() or 1  # 高さが0の場合に1を設定

        # リサイズ比率を計算
        scale = min(canvas_width / img.width, canvas_height / img.height)
        new_width = max(1, int(img.width * scale))  # 幅が0以下にならないようにする
        new_height = max(1, int(img.height * scale))  # 高さが0以下にならないようにする

        # 画像をリサイズ
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)

        # 画像をキャンバスに描画
        pic_canvas.create_image(
            canvas_width // 2, canvas_height // 2, anchor="center", image=tk_img
        )
        pic_canvas.image = tk_img  # 参照を保持してガベージコレクションを防ぐ


def delete_nearest_point(event):
    """右クリックで近くの頂点を削除"""
    nearest_index = find_nearest_point(event.x, event.y)
    if nearest_index is not None:
        deleted_point = polygon_points.pop(nearest_index)
        print(f"Deleted point: {deleted_point}")  # デバッグ用
        redraw_image()


def file_choose():
    file = filedialog.askopenfilename()
    if file:
        file_name_entry.delete(0, "end")
        file_name_entry.insert(0, file)


root = Tk()
root.state("zoomed")

file_frame = Frame(root)
file_frame.grid(row=0, column=0, sticky="nsew", columnspan=2)

pic_canvas = Canvas(root, bg="white")
pic_canvas.grid(row=1, column=0, sticky="nsew")
pic_canvas.bind("<Button-1>", on_click)

data_frame = Frame(root, bg="red")
data_frame.grid(row=1, column=1, sticky="nsew", rowspan=2)

added_frame = Frame(root, bg="blue")
added_frame.grid(row=2, column=0, sticky="nsew")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=100)
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=100)
root.grid_columnconfigure(1, weight=1)


file_name_entry = Entry(file_frame, width=50)
file_name_entry.grid(row=0, column=0, sticky="nsew")
file_name_entry.insert(
    0, r"C:\Users\aki19\Pictures\Screenshots\スクリーンショット 2024-07-25 133604.png"
)  # debug用
# root.after(1, set_picture)  # debug用

file_choose_button = Button(file_frame, text="Choose File", command=file_choose)
file_choose_button.grid(row=0, column=1, sticky="nsew")

# file_set_buttonのコールバック関数を変更
file_set_button = Button(
    file_frame,
    text="Set Picture",
    command=lambda file_entry=file_name_entry, pic_ca=pic_canvas: set_picture_with_edges(
        file_entry, pic_ca
    ),
)
file_set_button.grid(row=0, column=2, sticky="nsew")


class_label = Label(data_frame, text="Class")
class_label.pack(anchor="nw", padx=5, pady=5)
class_combo = Combobox = Combobox(
    data_frame, width=50, value=class_list, takefocus=False
)
class_combo.pack(anchor="ne", padx=5, pady=5)

get_class_button = Button(data_frame, text="Get Class", command=save_segmentation)
get_class_button.pack(anchor="ne", padx=5, pady=5)

# 保存ボタンを追加
# save_button = Button(data_frame, text="Save Segmentation", command=save_segmentation)
# save_button.pack(anchor="ne", padx=5, pady=5)
# イベントバインド
# pic_canvas.bind("<MouseWheel>", zoom)  # マウスホイールでズーム
# pic_canvas.bind("<ButtonPress-2>", start_drag)  # 中ボタン押下でドラッグ開始
# pic_canvas.bind("<B2-Motion>", drag)  # 中ボタンを押しながら移動
# イベントバインド

pic_canvas.bind("<MouseWheel>", zoom)  # マウスホイールでズーム
pic_canvas.bind("<ButtonPress-2>", start_drag)  # 中ボタン押下でドラッグ開始
pic_canvas.bind("<B2-Motion>", drag)  # 中ボタンを押しながら移動
pic_canvas.bind("<ButtonPress-1>", start_drag_point)  # 左クリックでドラッグ開始
pic_canvas.bind("<B1-Motion>", drag_point)  # 左クリックを押しながら移動
pic_canvas.bind("<ButtonRelease-1>", end_drag_point)  # 左クリックを離してドラッグ終了
pic_canvas.bind("<Button-3>", delete_nearest_point)  # 右クリックで削除
root.bind("<KeyPress-plus>", key_zoom)  # +キーで拡大
root.bind("<KeyPress-minus>", key_zoom)  # -キーで縮小
root.bind("<KeyPress-Up>", key_move)  # 上矢印キーで移動
root.bind("<KeyPress-Down>", key_move)  # 下矢印キーで移動
root.bind("<KeyPress-Left>", key_move)  # 左矢印キーで移動
root.bind("<KeyPress-Right>", key_move)  # 右矢印キーで移動
root.bind("<BackSpace>", delete_first_point)  # BackSpaceキーで最初の座標を削除
root.bind("<space>", toggle_image)  # スペースキーで画像を切り替え
root.bind("<Control_L>", on_click)  # 左Ctrlキーで頂点を追加
root.mainloop()
