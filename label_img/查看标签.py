#这个文件主要是把没有检测到的图片保存下来
import sys
import cv2
import  numpy as np
import  mss
import os
import onnxruntime
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from  PyQt5.QtCore import  QTimer
class YOLOv7:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        #self.class_names = list(map(lambda x: x.strip(), open('best.names', 'r').readlines()))
        self.class_names = ['person', 'para', 'monster', 'jabe', 'alter', 'boos']
        print(self.class_names)
        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.has_postprocess = 'score' in self.output_names

    def detect(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        if self.has_postprocess:
            boxes, scores, class_ids = self.parse_processed_output(outputs)

        else:
            # Process output data
            boxes, scores, class_ids = self.process_output(outputs)

        return boxes, scores, class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def process_output(self, output):
        predictions = np.squeeze(output[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def parse_processed_output(self, outputs):

        scores = np.squeeze(outputs[self.output_names.index('score')])
        predictions = outputs[self.output_names.index('batchno_classid_x1y1x2y2')]

        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        # Extract the boxes and class ids
        # TODO: Separate based on batch number
        batch_number = predictions[:, 0]
        class_ids = predictions[:, 1]
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xywh format
        boxes_ = np.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        return boxes_

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box.astype(int)
            # Draw rectangle
            cv2.circle(image,(x,y),5, (0,255,0), 1)
            #当检测到图像时,就把图像写成yolo格式的文件截屏保存

            #然后定时器2秒,之后在开启新的截屏
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            label = self.class_names[class_id]
            label = f'{label} {int(score * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # top = max(y1, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return image

model_path = "dnf.onnx"
#置信度阈值,当预测图像高于0.8时,才显示
confThreshold = 0.7
confNmsThreshold =0.5
yolov7_detector = YOLOv7(model_path, confThreshold, confNmsThreshold)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
class ScreenCapture(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyqt视频窗口")
        self.setGeometry(100, 100 ,100, 100)
        self.label =QLabel(self)
        self.label.setGeometry(0,0, 100,100)

        #创建qpushbutton
        self.sava_button = QPushButton("Save_image", self)
        self.sava_button.clicked.connect(self.save_image)
        self.cropped_frame_ = None
        self.next = 0
        self.sct = mss.mss()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)
    def  update_frame(self):
        # screenshot = self.sct.grab(self.sct.monitors[1])
        # frame = np.array(screenshot)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #
        # height, width, channel = frame.shape
        # bytes_pre_line = channel * width
        # q_image = QImage(frame.data, width, height, bytes_pre_line, QImage.Format_RGB888)
        # self.label.setPixmap(QPixmap.fromImage(q_image))
        ret, frame = cap.read()
        self.cropped_frame = frame[0:600, 0:800]
        self.cropped_frame_ = np.array(self.cropped_frame)
        #cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        boxes,scores, class_ids = yolov7_detector.detect(self.cropped_frame)
        # 绘制检测框
        self.cropped_frame = yolov7_detector.draw_detections(self.cropped_frame, boxes, scores, class_ids)
        # height, width, change = cropped_frame.shape
        # bytes_pre_line = change * width
        # q_image = QImage(dstimg.data, width, height, bytes_pre_line, QImage.Format_RGB888)
        # self.label.setPixmap(QPixmap.fromImage(q_image))
        cv2.imshow("img", self.cropped_frame)
        cv2.waitKey(10)
    def save_image(self):
        if self.cropped_frame_ is not None:
            # #确保文件夹存在
            save_dir = "img"
            # os.makedirs(save_dir,exist_ok= True)
            # #获取目录里的所有文件,并查找最大编号
            # existing_files = [f for f  in os.listdir(save_dir) if f.endswidth(".jpg")]
            # existing_numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
            # next_number = max(existing_numbers, default=0) + 1  # ✅ 计算下一张图片编号
            # 生成文件名
            self.next += 1
            filename = os.path.join(save_dir, f"{self.next}.jpg")

            cv2.imwrite(filename, cv2.cvtColor(self.cropped_frame_, cv2.COLOR_RGBA2RGB))
            print(f"图像以保存{filename}")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScreenCapture()
    window.show()
    sys.exit(app.exec_())

