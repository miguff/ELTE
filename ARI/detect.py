from ultralytics import YOLO

# Load the trained model
modelnew = YOLO(r'CARLA_newer\Bicycle\weights\best.pt')
modelold = YOLO(r'CARLA\Bicycle12\weights\best.pt')


source = r'KITTI\Other\test\img\006524.png'

results = modelnew.predict(source, conf=0.5)
print(results)
for result in results:
    print(result.boxes.xyxy)
    result.show()

results = modelold.predict(source, conf=0.5)
print(results)
for result in results:
    print(result.boxes.xyxy)
    result.show()