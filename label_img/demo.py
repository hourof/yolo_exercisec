import cv2
img = cv2.imread("1.jpg")
#获取图像得高度和宽和和通道数
height, width, _ = img.shape

#读取图像得标签信息
with open("1.txt", "r", encoding="utf-8") as f:
    #图像得id(标签)
    #0.358025 0.587963 图像中心的归一化后得x,y坐标
    #0.113580 0.425926 这两个数据是图像归一化后得宽高
    #'0 0.358025 0.587963 0.113580 0.425926\n'
    yolo_data = f.readlines()
    #print(yolo_data)
    object = [item.strip().split() for item in yolo_data]
    #print(object)
#读取图像得标签类别
with open("best.names", "r", encoding="utf-8") as f:
    class_list = f.readlines()
    class_list = [item.strip() for item in class_list]
    print(class_list)
#画出所有图像得位置信息
for i in object:
    class_id , x_center , y_center , w_center , h_center = map(float, i)
    #print(class_id , x_center , y_center , w_center , h_center)

    x_center, y_center = int(x_center *width), int(y_center *height)
    #画出图像标签得中心点
    cv2.circle(img, (x_center, y_center), 5, (0, 0, 255), -1)
    #获取图像得宽高
    w,h  = int(w_center * width),int(h_center * height)
    print(w,  h)
    #还原到图像得正确宽度位置
    #cv2.line(img, (x_center- (w //2), y_center), (x_center - (w//2) +w, y_center), (0, 255, 0), 2)
    #得到图像正确得高度位置
    #cv2.line(img,(x_center, y_center - (h //2)), (x_center, y_center - (h //2)+h) , color=(0,255,0), thickness=2)
    #得到图像左上角和又下角坐标
    x1, y1 = x_center - (w//2), y_center - (h//2)
    x2, y2 = x_center + (w//2), y_center + (h//2)
    #cv2.circle(img, (x2, y2), 5, (0, 0, 255), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #给图像加上类别标签
    id = class_list[int(class_id)]
    cv2.putText(img, f"{id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()