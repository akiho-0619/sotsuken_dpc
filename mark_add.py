from PIL import Image, ImageDraw

# 画像を読み込む（例: 'input.jpg'）
image = Image.open(r"E:\sotsuken\simple_pics\nasu.png").convert("RGB")

# 描画用オブジェクトを作成
draw = ImageDraw.Draw(image)

# 左上から5pxずつ余白を空けて、6px×6pxの黒色矩形を描画
x_offset = 5
y_offset = 5
rect_size = 6

# 矩形の左上と右下の座標を指定
top_left = (x_offset, y_offset)
bottom_right = (x_offset + rect_size, y_offset + rect_size)

# 黒色で塗りつぶし
draw.rectangle([top_left, bottom_right], fill=(0, 0, 0))

# 結果を保存（例: 'output.jpg'）
image.save(r"E:\sotsuken\simple_pics\nasu_marked.png", "PNG")
