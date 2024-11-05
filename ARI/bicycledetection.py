import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import torch
import os


def main():
    path_train_im = 'KITTI/data_tsinghua/train/images/'
    path_train_bb = 'KITTI/data_tsinghua/train/labels/'

    path_val_im = 'KITTI/data_tsinghua/val/images/'
    path_val_bb = 'KITTI/data_tsinghua/val/labels/'

    path_test_im = 'KITTI/data_tsinghua/test/images/'
    path_val_bb = 'KITTI/data_tsinghua/test/labels/'


    res = []
    for file in os.listdir(path_train_im):
        if file.endswith('.jpg'):
            res.append(path_train_im+file)
    with open('train.txt', 'w') as f:
        for line in res:
            f.write(line)
            f.write('\n')
    f.close()

    res=[]
    for file in os.listdir(path_val_im):
        if file.endswith('.jpg'):
            res.append(path_val_im+file)
    with open('val.txt', 'w') as f:
        for line in res:
            f.write(line)
            f.write('\n')
    f.close()

    res=[]
    for file in os.listdir(path_test_im):
        if file.endswith('.jpg'):
            res.append(path_test_im+file)
    with open('test.txt', 'w') as f:
        for line in res:
            f.write(line)
            f.write('\n')
    f.close()



    with open('CARCLA_bicycle.yaml', 'w') as f:

        f.write("path:  \ntrain: train.txt \nval: val.txt \ntest: test.txt \n")
        f.write("nc: 1 \n")
        f.write("names:\n  0: Bicycle")
        f.write('\n')
    f.close()

    model = YOLO('yolov8n.yaml')
    results=model.train(data="CARCLA_bicycle.yaml", epochs=10, batch=16, project="CARLA", name="Bicycle")


if __name__ == "__main__":
    main()