import cv2

# 画像を読み込む
img = cv2.imread(r"E:\sotsuken\simple_pics\carrot_marked.jpg")

# グレースケール化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二値化
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# 輪郭抽出
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # 矩形の位置とサイズを取得
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"位置: ({x}, {y}), サイズ: ({w} x {h})")
