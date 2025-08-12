from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # 学習済みモデル（COCOデータセット）
results = model(r"E:\sotsuken\pictures\IMG20240520173806.jpg")  # 画像ファイルで推論

for r in results:
    boxes = r.boxes.xyxy  # [x1, y1, x2, y2]
    classes = r.boxes.cls  # クラスID
    confs = r.boxes.conf  # 信頼度
    print(boxes, classes, confs)
    r.show()  # バウンディングボックスを描画して表示
