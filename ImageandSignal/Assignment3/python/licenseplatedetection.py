import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import torch
import os


def main():
    path_train_im = 'train/images/'

    path_val_im = 'valid/images/'

    path_test_im = 'test/images/'


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



    with open('LicensePlate.yaml', 'w') as f:

        f.write("path:  \ntrain: train.txt \nval: val.txt \ntest: test.txt \n")
        f.write("nc: 1 \n")
        f.write("names:\n  0: LicensePlate")
        f.write('\n')
    f.close()

    model = YOLO('yolov4.yaml')
    results=model.train(data="LicensePlate.yaml", epochs=10, batch=16, project="LicensePlateDetectionMatlab", name="LicensePlate")


if __name__ == "__main__":
    main()