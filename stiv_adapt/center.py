import cv2
import os

video_path = r"D:\Programs\Python\stiv1\CRR.MP4"
pic_path =r"D:\Programs\Python\stiv1\CRR_calibration_image.jpg"
cap = cv2.VideoCapture(video_path)
img = cv2.imread(pic_path,cv2.IMREAD_COLOR)

H, W = img.shape[:2]
scale = 0.3
small = cv2.resize(img, (int(W*scale), int(H*scale)))
disp = cv2.resize(img,
                  (int(W * scale),int(H * scale)),
                  interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                  )

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        # 转换回原图坐标
        orig_x = int(round(x / scale))
        orig_y = int(round(y / scale))
        print(f"你点击的位置: ({orig_x}, {orig_y})")
        cv2.destroyAllWindows()

cv2.imshow("Select Center (缩小显示)", small)
cv2.setMouseCallback("Select Center (缩小显示)", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()

