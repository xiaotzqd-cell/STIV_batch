import cv2
import os

video_path = r"D:\Programs\Python\stiv1\CRR.MP4"
#pic_path =r"D:\Programs\Python\stiv1\CRR_calibration_image.jpg"
pic_path =r"D:\Programs\Python\stiv\stiv_adapt\ACS_calibration_image.jpg"
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
        param.append((orig_x, orig_y))


def redraw(window_name: str, pts):
    canvas = disp.copy()
    for i, (ox, oy) in enumerate(pts, 1):
        dx = int(round(ox * scale))
        dy = int(round(oy * scale))
        cv2.circle(canvas, (dx, dy), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (dx + 6, dy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_name, canvas)


window_name = "Select Centers (缩小显示，回车确认)"
selected_points = []
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback, selected_points)

print("左键点击选择任意多个点，按回车键输出所有坐标并退出。")

while True:
    redraw(window_name, selected_points)
    key = cv2.waitKey(30) & 0xFF
    if key in (13, 10):  # Enter 键
        break

cv2.destroyAllWindows()

if selected_points:
    print(f"共选择 {len(selected_points)} 个点：")
    for idx, (ox, oy) in enumerate(selected_points, 1):
        print(f"  #{idx}: ({ox}, {oy})")
else:
    print("未选择任何点")

