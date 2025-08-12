from tkinter import ttk, Tk, Label, Button, Canvas, Frame, Entry
from tkinter.ttk import Combobox
from tkinter import filedialog
from PIL import Image, ImageTk
import os

class_list = [os.path.splitext(i)[0] for i in os.listdir("classes") ]
write_file = "classes/"
# クリックした座標を保持するリスト（ポリゴン用）
polygon_points = []

# 画像の拡大縮小と移動用の変数
scale_factor = 1.0
image_offset = [0, 0]  # [x_offset, y_offset]

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
                outline="blue", fill="", tags="polygon"
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

def redraw_image():
    """画像を再描画（拡大縮小や移動を反映）"""
    global tk_img, scale_factor, image_offset
    file_path = file_name_entry.get()
    if file_path:
        pic_canvas.delete("all")
        
        # 画像をPillowで開く
        img = Image.open(file_path)
        
        # リサイズ
        new_width = max(1, int(img.width * scale_factor))
        new_height = max(1, int(img.height * scale_factor))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        
        # 画像をキャンバスに描画
        pic_canvas.create_image(image_offset[0], image_offset[1], anchor="center", image=tk_img)
        pic_canvas.image = tk_img  # 参照を保持してガベージコレクションを防ぐ


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

def on_click(event):
    """クリックイベントのコールバック関数（ポリゴン用）"""
    global polygon_points
    x, y = event.x, event.y
    polygon_points.append((x, y))
    print(f"Clicked at: ({x}, {y})")  # デバッグ用

    # ポリゴンを描画
    if len(polygon_points) > 1:
        pic_canvas.delete("polygon")
        pic_canvas.create_polygon(
            *[coord for point in polygon_points for coord in point],
            outline="blue", fill="", tags="polygon"
        )


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
        pic_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor="center", image=tk_img)
        pic_canvas.image = tk_img  # 参照を保持してガベージコレクションを防ぐ

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

data_frame = Frame(root, bg = "red")
data_frame.grid(row=1, column=1, sticky="nsew", rowspan=2)

added_frame = Frame(root, bg = "blue")
added_frame.grid(row=2, column=0, sticky="nsew")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=100)
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=100)
root.grid_columnconfigure(1, weight=1)


file_name_entry = Entry(file_frame, width=50)
file_name_entry.grid(row=0, column=0, sticky="nsew")
file_name_entry.insert(0, r"C:\Users\aki19\Pictures\Screenshots\スクリーンショット 2024-07-25 133604.png")  #debug用
root.after(1, set_picture)  #debug用

file_choose_button = Button(file_frame, text="Choose File", command=file_choose)
file_choose_button.grid(row=0, column=1, sticky="nsew")

file_set_button = Button(file_frame, text="Set Picture", command=set_picture)
file_set_button.grid(row=0, column=2, sticky="nsew")


class_label = Label(data_frame, text="Class")
class_label.pack(anchor="nw", padx=5, pady=5)
class_combo = Combobox = Combobox(data_frame, width=50, value = class_list, takefocus=False)
class_combo.pack(anchor="ne", padx=5, pady=5)

get_class_button = Button(data_frame, text="Get Class", command=save_segmentation)
get_class_button.pack(anchor="ne", padx=5, pady=5)

# 保存ボタンを追加
# save_button = Button(data_frame, text="Save Segmentation", command=save_segmentation)
# save_button.pack(anchor="ne", padx=5, pady=5)
# イベントバインド
pic_canvas.bind("<MouseWheel>", zoom)  # マウスホイールでズーム
pic_canvas.bind("<ButtonPress-2>", start_drag)  # 中ボタン押下でドラッグ開始
pic_canvas.bind("<B2-Motion>", drag)  # 中ボタンを押しながら移動
# イベントバインド
pic_canvas.bind("<MouseWheel>", zoom)  # マウスホイールでズーム
pic_canvas.bind("<ButtonPress-2>", start_drag)  # 中ボタン押下でドラッグ開始
pic_canvas.bind("<B2-Motion>", drag)  # 中ボタンを押しながら移動
root.bind("<KeyPress-plus>", key_zoom)  # +キーで拡大
root.bind("<KeyPress-minus>", key_zoom)  # -キーで縮小
root.bind("<KeyPress-Up>", key_move)  # 上矢印キーで移動
root.bind("<KeyPress-Down>", key_move)  # 下矢印キーで移動
root.bind("<KeyPress-Left>", key_move)  # 左矢印キーで移動
root.bind("<KeyPress-Right>", key_move)  # 右矢印キーで移動
root.bind("<BackSpace>", delete_first_point)  # BackSpaceキーで最初の座標を削除
root.mainloop()