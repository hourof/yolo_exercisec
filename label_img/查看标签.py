import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage

class ImageLabelingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.index = 0  # 初始化索引
        self.images_list = [os.path.join("images", i) for i in os.listdir("images")]
        self.label_list = [os.path.join("label", i) for i in os.listdir("label")]

        self.initUI()

    def initUI(self):
        self.setWindowTitle("YOLO 图像标注查看器")
        self.setGeometry(100, 100, 800, 600)

        # 创建 QLabel 用于显示图像
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 500)

        # 创建按钮
        self.button = QPushButton("下一张", self)
        self.button.clicked.connect(self.next_image)

        # 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.button)
        self.setLayout(layout)

        # 初次加载图像
        self.load_image()

    def load_image(self):
        if self.index >= len(self.images_list):
            self.index = 0  # 如果超出范围，循环回到第一张

        img = cv2.imread(self.images_list[self.index])
        height, width, _ = img.shape

        # 读取 YOLO 标签
        with open(self.label_list[self.index], "r", encoding="utf-8") as f:
            yolo_data = f.readlines()
            objects = [item.strip().split() for item in yolo_data]

        # 读取类别列表
        with open("best.names", "r", encoding="utf-8") as f:
            class_list = [item.strip() for item in f.readlines()]

        # 绘制目标框
        for obj in objects:
            class_id, x_center, y_center, w_center, h_center = map(float, obj)
            x_center, y_center = int(x_center * width), int(y_center * height)
            w, h = int(w_center * width), int(h_center * height)

            x1, y1 = x_center - w // 2, y_center - h // 2
            x2, y2 = x_center + w // 2, y_center + h // 2

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_list[int(class_id)], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 转换为 Qt 格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimage = QImage(img.data, width, height, width * 3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))

    def next_image(self):
        self.index += 1  # 索引递增
        self.load_image()  # 更新图像

# 启动应用
app = QApplication(sys.argv)
window = ImageLabelingApp()
window.show()
sys.exit(app.exec_())
