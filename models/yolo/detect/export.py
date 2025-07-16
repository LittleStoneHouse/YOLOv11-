import os
import cv2
import numpy as np
from ultralytics import YOLO


def export_train_map():
    model = YOLO('y16n.pt')  # 或者你训练好的权重
    data = r'F:\PythonProjects\YOLOv11\ultralytics\models\yolo\detect\PF_dataset\data.yaml'
    args = dict(model=model, data=data, epochs=0, resume=True, pretrained=True, workers=0)
    model.train(**args)
    # results = model.val(
    #     data=r"F:\PythonProjects\YOLOv11\ultralytics\models\yolo\detect\ABS\data.yaml",
    #     imgsz=640
    # )
    # print(results.box.map)


def export_pred_and_true_label():
    root_inner = './runs/detect/外部验证集结果'
    inner = [os.path.join(root_inner, path) for path in os.listdir(root_inner)]
    inner_path_info = [os.path.splitext(path) for path in inner]
    path_dict = {k: [] for k, _ in inner_path_info}
    # Yellow, Dark-Green
    label_dict = {0.0: 'NPCOS-label', 1.0: 'PCOS-label'}
    color_dict = {0.0: (0, 255, 255), 1.0: (0, 66, 0)}
    for path in inner_path_info:
        path_dict[path[0]].append(path[1])
    # 0074FF NPCOS
    # 00FFFF PCOS
    for k, v in path_dict.items():
        t_img, t_lab = v
        image = cv2.imread(k + t_img)
        img_h, img_w = image.shape[:2]
        label = np.loadtxt(k + t_lab)
        if label.ndim == 1:
            t, cx, cy, w, h = label
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            canvas = image.copy()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=color_dict[t], thickness=4)
            cv2.putText(canvas, label_dict[t], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness=2)
            print('./result', k + t_img)
            cv2.imwrite(os.path.join('./result', os.path.basename(k + t_img)), canvas)


if __name__ == "__main__":
    export_train_map()
