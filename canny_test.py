import cv2
import numpy as np
from tkinter import (
    Tk,
    Canvas,
    PhotoImage,
    Scale,
    HORIZONTAL,
    Button,
    Frame,
    Entry,
    StringVar,
)
from PIL import Image, ImageTk
import threading

# 入力画像の読み込み
image_path = r"pictures\IMG20240711155650.jpg"  # ここに画像ファイルのパスを指定
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("画像を読み込めませんでした。パスを確認してください。")
    exit()

# 初期のCannyエッジ検出パラメータ
low_threshold = 180
high_threshold = 220


# Cannyエッジ検出
def apply_canny(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = [contour.reshape(-1, 2).tolist() for contour in contours]
    return edges, contour_points


edges, contour_points = apply_canny(image, low_threshold, high_threshold)


# Entryの値を監視してScaleを更新
def on_low_entry_change(*args):
    try:
        value = int(low_var.get())
        if 0 <= value <= 255:  # 値が範囲内の場合のみ更新
            low_scale.set(value)
    except ValueError:
        pass  # 無効な値の場合は無視


def on_high_entry_change(*args):
    try:
        value = int(high_var.get())
        if 0 <= value <= 255:  # 値が範囲内の場合のみ更新
            high_scale.set(value)
    except ValueError:
        pass  # 無効な値の場合は無視


# Tkinterで画像を表示
def display_image_with_points(image, points, canvas_size=(1333, 1000)):
    global low_var, high_var, low_scale, high_scale, edge_image
    # Tkinterの初期化
    root = Tk()
    root.title("Image with Contour Points")
    root.state("zoomed")  # ウィンドウを最大化

    # 画像のスケーリング
    height, width = image.shape
    if height > width:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height

    scale_x = canvas_size[0] / width
    scale_y = canvas_size[1] / height
    scale = min(scale_x, scale_y)  # アスペクト比を維持するためのスケール
    new_width = int(width * scale)
    new_height = int(height * scale)

    # OpenCV画像をリサイズ
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    cv2.imwrite("tmp.png", resized_image)  # デバッグ用

    # Tkinter用に変換
    # photo = PhotoImage(width=new_width, height=new_height)
    # for y in range(new_height):
    #     for x in range(new_width):
    #         color = resized_image[y, x]
    #         photo.put(f"#{color:02x}{color:02x}{color:02x}", (x, y))

    photo = ImageTk.PhotoImage(
        Image.open("tmp.png")
    )  # PILを使用して画像をTkinterに変換

    # Canvasに画像を描画
    canvas = Canvas(root, width=canvas_size[0], height=canvas_size[1])
    canvas.pack()

    # 画像をキャンバスの中央に配置
    offset_x = (canvas_size[0] - new_width) // 2
    offset_y = (canvas_size[1] - new_height) // 2
    canvas_image = canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)

    # 保存する座標リスト
    saved_points = []

    def delet_points(event):
        nonlocal point_items, canvas_image, photo
        click_x = event.x - offset_x
        click_y = event.y - offset_y

        # スケールを元に戻した座標で計算
        original_x = int(click_x / scale)
        original_y = int(click_y / scale)

        # 保存された座標から削除
        N = 5
        # 半径N内の座標を検索
        for num, contour in enumerate(points):
            for x, y in contour:
                distance = ((x - original_x) ** 2 + (y - original_y) ** 2) ** 0.5
                if distance <= N:
                    del points[num]
                    # saved_points.remove((x, y))

        # ドットを削除
        canvas.delete("all")
        canvas_image = canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)
        for contour in points:
            for x, y in contour:
                scaled_x = int(x * scale) + offset_x
                scaled_y = int(y * scale) + offset_y
                canvas.create_oval(
                    scaled_x - 1,
                    scaled_y - 1,
                    scaled_x + 1,
                    scaled_y + 1,
                    fill="red",
                    outline="red",
                )

    # 半径N内の座標を保存
    def save_points(event):
        nonlocal saved_points
        N = 10  # 半径の設定
        click_x = event.x - offset_x
        click_y = event.y - offset_y

        # スケールを元に戻した座標で計算
        original_x = int(click_x / scale)
        original_y = int(click_y / scale)

        # 半径N内の座標を検索
        for contour in points:
            for x, y in contour:
                distance = ((x - original_x) ** 2 + (y - original_y) ** 2) ** 0.5
                if distance <= N:
                    saved_points.append((x, y))
                    # スケーリング後の座標を計算
                    scaled_x = int(x * scale) + offset_x
                    scaled_y = int(y * scale) + offset_y
                    # ドットの色を青色に変更
                    canvas.create_oval(
                        scaled_x - 1,
                        scaled_y - 1,
                        scaled_x + 1,
                        scaled_y + 1,
                        fill="blue",
                        outline="blue",
                    )

        print(f"クリック位置: ({original_x}, {original_y})")
        print(f"保存された座標: {saved_points}")

    # Canvasクリックイベントをバインド
    canvas.bind("<B1-Motion>", save_points)  # 左クリックで座標を保存
    zoom_scale = 1.0  # ズームスケール

    # マウスホイールで拡大縮小
    def zoom(event):
        nonlocal zoom_scale, canvas_image, photo, offset_x, offset_y
        if event.delta > 0:  # スクロールアップで拡大
            zoom_scale *= 1.1
        elif event.delta < 0:  # スクロールダウンで縮小
            zoom_scale /= 1.1

        # 新しいサイズを計算
        new_width = int(width * zoom_scale)
        new_height = int(height * zoom_scale)

        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite("tmp.png", resized_image)  # デバッグ用

        # Tkinter用に変換
        # photo = PhotoImage(width=new_width, height=new_height)
        # for y in range(new_height):
        #     for x in range(new_width):
        #         color = resized_image[y, x]
        #         photo.put(f"#{color:02x}{color:02x}{color:02x}", (x, y))

        photo = ImageTk.PhotoImage(
            Image.open("tmp.png")
        )  # PILを使用して画像をTkinterに変換

        # 画像を再描画
        # canvas.delete(canvas_image)
        # offset_x = (canvas_size[0] - new_width) // 2
        # offset_y = (canvas_size[1] - new_height) // 2
        # canvas_image = canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)

        canvas.delete("all")
        # for item in point_items:
        #     canvas.delete(item)
        offset_x = abs(canvas_size[0] - new_width) // 2
        offset_y = abs(canvas_size[1] - new_height) // 2
        canvas_image = canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)

        # ドットを再描画
        point_items = []
        for x, y in saved_points:
            scaled_x = int(x * zoom_scale) + offset_x
            scaled_y = int(y * zoom_scale) + offset_y
            point_items.append(
                canvas.create_oval(
                    scaled_x - 2,
                    scaled_y - 2,
                    scaled_x + 2,
                    scaled_y + 2,
                    fill="blue",
                    outline="blue",
                )
            )

    def zoom_func(e):
        threading.Thread(target=zoom, args=(e,)).start()

    # ミドルクリックでドラッグ移動
    drag_data = {"x": 0, "y": 0}

    def start_drag(event):
        drag_data["x"] = event.x
        drag_data["y"] = event.y

    def drag(event):
        nonlocal offset_x, offset_y
        dx = event.x - drag_data["x"]
        dy = event.y - drag_data["y"]
        offset_x += dx
        offset_y += dy
        canvas.move(canvas_image, dx, dy)
        drag_data["x"] = event.x
        drag_data["y"] = event.y

    # Contour Pointsをプロット（スケーリング後の座標に変換）
    point_items = []
    for contour in points:
        for x, y in contour:
            scaled_x = int(x * scale) + offset_x
            scaled_y = int(y * scale) + offset_y
            point_items.append(
                canvas.create_oval(
                    scaled_x - 1,
                    scaled_y - 1,
                    scaled_x + 1,
                    scaled_y + 1,
                    fill="red",
                    outline="red",
                )
            )

    # フレームを作成してスライダーを配置
    control_frame = Frame(root, width=canvas_size[0], height=100)
    control_frame.pack()

    low_scale = Scale(
        control_frame, from_=0, to=255, orient=HORIZONTAL, label="Low Threshold"
    )
    low_scale.set(low_threshold)
    low_scale.pack(side="left", padx=10)

    high_scale = Scale(
        control_frame, from_=0, to=255, orient=HORIZONTAL, label="High Threshold"
    )
    high_scale.set(high_threshold)
    high_scale.pack(side="left", padx=10)

    def update_image():
        # 新しいパラメータを取得
        new_low_threshold = low_scale.get()
        new_high_threshold = high_scale.get()

        # Cannyエッジ検出の再実行
        new_edges = cv2.Canny(image, new_low_threshold, new_high_threshold)

        # 輪郭の再抽出
        new_contours, _ = cv2.findContours(
            new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        new_contour_points = [
            contour.reshape(-1, 2).tolist() for contour in new_contours
        ]

        # Canvasをクリア
        canvas.delete("all")
        # for item in point_items:
        #     canvas.delete(item)

        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        # 画像を再描画
        cv2.imwrite("tmp.png", resized_image)  # デバッグ用

        # Tkinter用に変換
        # photo = PhotoImage(width=new_width, height=new_height)
        # for y in range(new_height):
        #     for x in range(new_width):
        #         color = resized_image[y, x]
        #         photo.put(f"#{color:02x}{color:02x}{color:02x}", (x, y))

        photo = ImageTk.PhotoImage(
            Image.open("tmp.png")
        )  # PILを使用して画像をTkinterに変換
        canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)

        # 輪郭点を再描画
        for contour in new_contour_points:
            for x, y in contour:
                scaled_x = int(x * scale) + offset_x
                scaled_y = int(y * scale) + offset_y
                point_items.append(
                    canvas.create_oval(
                        scaled_x - 1,
                        scaled_y - 1,
                        scaled_x + 1,
                        scaled_y + 1,
                        fill="red",
                        outline="red",
                    )
                )

    # 更新ボタンを作成
    update_button = Button(control_frame, text="Update", command=update_image)
    update_button.pack(side="left", padx=10)

    # Entryの値を監視するためのStringVarを作成
    low_var = StringVar()
    high_var = StringVar()

    low_var.trace_add("write", on_low_entry_change)
    high_var.trace_add("write", on_high_entry_change)

    low_entry = Entry(control_frame, textvariable=low_var, width=5)
    high_entry = Entry(control_frame, textvariable=high_var, width=5)
    low_entry.pack()
    high_entry.pack()

    low_var.set(str(low_threshold))
    high_var.set(str(high_threshold))

    # イベントバインド
    canvas.bind("<MouseWheel>", zoom_func)  # マウスホイールで拡大縮小
    canvas.bind("<ButtonPress-2>", start_drag)  # ミドルクリックでドラッグ開始
    canvas.bind("<B2-Motion>", drag)  # ミドルクリックを押しながら移動

    canvas.bind("<Button-3>", delet_points)  # 右クリックでドットを削除
    root.mainloop()


deleted_points = []


# 距離が近い座標を削除する関数
def remove_close_points(points, min_distance=1):
    filtered_points = []
    for contour in points:
        filtered_contour = []
        for i, (x1, y1) in enumerate(contour):
            keep = True
            for j, (x2, y2) in enumerate(contour):
                if i != j:
                    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    if distance < min_distance:
                        keep = False
                        break
            if keep:
                filtered_contour.append((x1, y1))
        filtered_points.append(filtered_contour)
    return filtered_points


deleted_points = remove_close_points(contour_points)
# # 距離が近い座標を削除
# contour_points = remove_close_points(contour_points)
# for contour in range(len(contour_points)):
#     for num in range(len(contour_points[contour])):
#         this_x, this_y = contour_points[contour][num]
#         for contour_ in range(contour, len(contour_points)):
#             for num_ in range(len(contour_points[contour_])):
#                 distance = (
#                     (this_x - contour_points[contour_][num_][0]) ** 2
#                     + (this_y - contour_points[contour_][num_][1]) ** 2
#                 ) ** 0.5
#                 if distance >= 5:  # and contour != contour_:
#                     # print(f"({this_x}, {this_y})と({contour_points[contour_][num_][0]}, {contour_points[contour_][num_][1]})の距離: {distance}")
#                     deleted_points.append((this_x, this_y))

# グレースケール画像を表示し、輪郭点をプロット
display_image_with_points(image, deleted_points)
