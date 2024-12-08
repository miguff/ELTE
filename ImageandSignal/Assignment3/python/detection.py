from ultralytics import YOLO

# Load the trained model

def detect(source):
    modelold = YOLO(r'LicensePlateDetection\LicensePlate5\weights\best.pt')
    #path = modelold.export(format='onnx', dynamic=True, opset=14)
    #print(path)

    source = r"C:\Users\msi\Documents\ELTE\ImageandSignal\Assignment3\P9170012.jpg"

    results = modelold.predict(source, conf=0.5)
    print(results)
    for result in results:
        x1 = int(result.boxes.xyxy[0][0].item())
        y1 = int(result.boxes.xyxy[0][1].item())
        x2 = int(result.boxes.xyxy[0][2].item())
        y2 = int(result.boxes.xyxy[0][3].item())
        print(x1, y1, x2, y2)
        x2 = x2-x1
        y2 = y2-y1

        print(x1, y1, x2, y2)
        result.show()

    return x1, y1, x2, y2

source = r"C:\Users\msi\Documents\ELTE\ImageandSignal\Assignment3\P9170012.jpg"
detect(source)
