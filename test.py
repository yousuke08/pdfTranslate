import cv2
import numpy as np
import sys

img = cv2.imread('./dat/doc.jpg')

# グレースケール化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 大津の二値化
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# 白黒反転
gray = cv2.bitwise_not(gray)
kernel = np.ones((4,4),np.uint8)
pan = cv2.dilate(gray,kernel,iterations = 5)
cv2.imwrite('pan.jpg', pan)

# ラベリング処理(詳細版)
label = cv2.connectedComponentsWithStats(pan)

# オブジェクト情報を項目別に抽出
n = label[0] - 1
data = np.delete(label[2], 0, 0)
center = np.delete(label[3], 0, 0)

# ラベリング結果書き出し用に二値画像をカラー変換
color_src = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# オブジェクト情報を利用してラベリング結果を表示
for i in range(n):
    # 各オブジェクトの外接矩形を赤枠で表示
    x0 = data[i][0]
    y0 = data[i][1]
    x1 = data[i][0] + data[i][2]
    y1 = data[i][1] + data[i][3]
    cv2.rectangle(color_src, (x0, y0), (x1, y1), (0, 0, 255))

    # 各オブジェクトのラベル番号と面積に黄文字で表示
    cv2.putText(color_src, "ID: " +str(i + 1), (x0, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
    cv2.putText(color_src, "S: " +str(data[i][4]), (x0, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    # 各オブジェクトの重心座標をに黄文字で表示
    cv2.putText(color_src, "X: " + str(int(center[i][0])), (x1 - 10, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
    cv2.putText(color_src, "Y: " + str(int(center[i][1])), (x1 - 10, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

# 画像の保存
cv2.imwrite('sample_label2.jpg', color_src)
