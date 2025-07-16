import os, shutil
from sklearn.model_selection import train_test_split

val_size = 0.1
test_size = 0.0
postfix = 'jpg'
imgpath = r'F:\PythonProjects\YOLOv11\ultralytics\models\yolo\detect\ThyCar\train\images'
txtpath = r'F:\PythonProjects\YOLOv11\ultralytics\models\yolo\detect\ThyCar\train\labels'

listdir = [i for i in os.listdir(txtpath) if 'txt' in i]
train, val = train_test_split(listdir, test_size=val_size, shuffle=True, random_state=0)
print(f'train set size:{len(train)} val set size:{len(val)}')

for i in val:
    shutil.move('{}/{}.{}'.format(imgpath, i[:-4], postfix), r'F:\PythonProjects\YOLOv11\ultralytics\models\yolo\detect\ThyCar\val\images\{}.{}'.format(i[:-4], postfix))
    shutil.move('{}/{}'.format(txtpath, i), r'F:\PythonProjects\YOLOv11\ultralytics\models\yolo\detect\ThyCar\val\labels\{}'.format(i))
