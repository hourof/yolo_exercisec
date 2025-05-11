import cv2
img = cv2.imread("bus.jpg")
height, width, _ = img.shape

#读取坐标文件
with open("bus.txt", "r", encoding="utf-8") as file:
    yolo_data = file.readlines()
object = [item.strip().split() for item in yolo_data]
print(object[0])

class_id, x_center, y_center, w, h = map(float, object[0])
#转换为像素坐标
x_center, y_center = int(x_center* width), int(y_center * height)

cv2.circle(img,(x_center, y_center),4,(0, 255, 0), -1)
w, h = int(w * width), int(h * height)
cv2.line(img, (x_center-(w //2), y_center), (x_center-(w //2) + w,y_center), color=(0, 255, 0), thickness=3)
cv2.line(img, (x_center, y_center- (h //2)),(x_center,y_center-(h //2) + h), color= (0,255,0), thickness=3 )
#计算左上角和右下角坐标
x1, y1 = x_center- w//2, y_center -h//2
x2, y2 = x_center + w //2, y_center + h //2
cv2.rectangle(img,(x1,y1),(x2, y2), color=(0 ,255 ,0), thickness=2 )
cv2.circle(img,(x1, y1),4, color=(0, 255, 0), thickness= -1)
cv2.circle(img,(x2, y2),4, color=(0, 255, 0), thickness= -1)
#给图像加标签类别
cv2.putText(img, f"class:{int(class_id)}",(x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
print(w,h)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyWindow()