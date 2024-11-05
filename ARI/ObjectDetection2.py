import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import torch
import os


def main():
    path_train_im = 'Combined/train/images/'

    path_val_im = 'Combined/val/images/'

    path_test_im = 'Combined/test/images/'


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



    with open('CARCLA_bicycle_newer.yaml', 'w') as f:

        f.write("path:  \ntrain: train.txt \nval: val.txt \ntest: test.txt \n")
        f.write("nc: 1 \n")
        f.write("names:\n  0: Bicycle")
        f.write('\n')
    f.close()

    model = YOLO('CARLA\\Bicycle12\\weights\\best.pt')
    results=model.train(data="CARCLA_bicycle_newer.yaml", epochs=10, batch=16, project="CARLA_newer", name="Bicycle")


if __name__ == "__main__":
    main()