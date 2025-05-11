import cv2
import numpy as np
import onnxruntime
import argparse
import os

class YOLOv7:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = list(map(lambda x: x.strip(), open('best.names', 'r').readlines()))
        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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
        img_list  = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, w, h = box.astype(int)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), thickness=2)
            # x1,y1是图像左上角坐标,x2,y2是图像右下角坐标
            x2, y2 = x1 + w, y1 +h
            img_list.append([class_id,x1,y1, x2, y2])
            label = self.class_names[class_id]
            label = f'{label} {int(score * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # top = max(y1, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        return image, img_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/1.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='dnf.onnx',
                        choices=["models/yolov7_640x640.onnx", "models/yolov7-tiny_640x640.onnx",
                                 "models/yolov7_736x1280.onnx", "models/yolov7-tiny_384x640.onnx",
                                 "models/yolov7_480x640.onnx", "models/yolov7_384x640.onnx",
                                 "models/yolov7-tiny_256x480.onnx", "models/yolov7-tiny_256x320.onnx",
                                 "models/yolov7_256x320.onnx", "models/yolov7-tiny_256x640.onnx",
                                 "models/yolov7_256x640.onnx", "models/yolov7-tiny_480x640.onnx",
                                 "models/yolov7-tiny_736x1280.onnx", "models/yolov7_256x480.onnx"],
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    # Initialize YOLOv7 object detector
    yolov7_detector = YOLOv7(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    # 把图像中的坐标信息保存到label文件夹里面
    floder_path = "images"
    # 读取文件夹里的图片
    images_list = []
    for index,filename in enumerate(os.listdir(floder_path)):
        # 确保是图像文件
        if filename.lower().endswith((".png", ".jpg")):
            img_path = os.path.join(floder_path, filename)
            images_list.append(filename)
    # 确保有label文件夹,如果没有就创建,并且把images_list的文件夹里的元素都把后缀改名为txt
    # 确保有label文件存在,如果没有就创建
    print(images_list)
    os.makedirs("label", exist_ok=True)
    for images_name in images_list:
        txt_path = os.path.join("label", images_name.replace("jpg", "txt"))
        # 创建空的字符串
        with open(txt_path, "w", ) as f:
            f.write("")

    # 为每张图像创建一个txt文件
    label_list = []
    #依次读取图像
    for index,item in enumerate(images_list):
        #print(index)
        img_path = os.path.join("images", images_list[index])
        srcimg = cv2.imread(img_path)
        # Detect Objects
        boxes, scores, class_ids = yolov7_detector.detect(srcimg)
        # Draw detections2
        dstimg, images = yolov7_detector.draw_detections(srcimg, boxes, scores, class_ids)
        height, width, _ = dstimg.shape
        for item in images:
            print("数据",item)
            class_id , x1, y1, x2, y2 = item
            # 归一化后数据
            x_center = round(((x1 + x2) / 2) / width, 6)  # 归一化 X 轴中心点
            y_center = round(((y1 + y2) / 2) / height, 6)  # 归一化 Y 轴中心点
            w_norm = round((x2 - x1) / width, 6)  # 归一化目标宽度
            h_norm = round((y2 - y1) / height, 6)  # 归一化目标高度

            # for images_name in images_list:
            #     txt_path = os.path.join("label", images_name.replace("jpg", "txt"))
            #     # 创建空的字符串
            #     with open(txt_path, "w", ) as f:
            #         f.write("")
            #         # 去掉图像路径
            #         label_list.append(f.name)
            for a in os.listdir("label"):
                label_list.append(a)
            #print(label_list[index])
            label_txt = os.path.join("label", label_list[index])
            with open(label_txt, 'a', encoding='utf-8') as f:
                f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")



    cv2.imshow("img", dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()