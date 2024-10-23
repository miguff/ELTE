import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import cv2
import numpy as np
import sys





class ImageDataset(Dataset):
    """
    A custom dataset for loading images and their corresponding YOLO-formatted labels from a directory.

    Args:
        root_dir (str): Path to the root directory containing 'images' and 'labels' subdirectories.
                        The 'images' directory should contain image files (e.g., .jpg), and the 'labels'
                        directory should contain corresponding YOLO-format label files (e.g., .txt).
        transform (callable, optional): Optional transformation to be applied on an image. 
                                        If None, it defaults to ToTensor.

    Attributes:
        root_dir (str): The root directory path.
        image_dir (str): Path to the 'images' subdirectory within root_dir.
        label_dir (str): Path to the 'labels' subdirectory within root_dir.
        image_filenames (list): A list of filenames from the 'images' directory.
        transform (callable, optional): A transformation to apply to each image (default is None).

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns the image and its corresponding YOLO-formatted label (class and bounding box)
                          for a given index.
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialize the ImageDataset object by setting the root directory and loading image filenames.

        Args:
            root_dir (str): Path to the root directory containing 'images' and 'labels' folders.
            transform (callable, optional): A function/transform to apply to the image (e.g., a torchvision transform).
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.image_filenames = os.listdir(self.image_dir)
        self.transform = transform  # Default to ToTensor if no transform

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            int: The number of image files in the 'images' directory.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding YOLO-formatted label (bounding box).

        Args:
            idx (int): The index of the image and label to retrieve.

        Returns:
            dict: A dictionary with the following keys:
                  - 'image' (Tensor): The image loaded and transformed (if a transform is provided).
                  - 'label' (Tensor): The class ID of the object in the image.
                  - 'bbox' (Tensor): The bounding box in YOLO format (x_center, y_center, width, height),
                                     all normalized to the range [0, 1].

        Notes:
            - The image is loaded using PIL and converted to RGB.
            - YOLO label files must be in .txt format with class ID and bounding box details (in YOLO format).
            - If there's an error loading an image, it prints an error message and returns None.
        """
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load the image safely using PIL and convert to RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        # This is for YOLO
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, label_name)

        # Read the YOLO label data
        with open(label_path, 'r') as YoloData:
            line = YoloData.readline().strip()
            values = line.split()
            class_id = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

        # Construct the bounding box
        bbox = [
            x_center,
            y_center,
            width,
            height
        ]

        # Apply any transformations on the image (if specified)
        if self.transform:
            image = self.transform(image)

        # Return image and label as a dictionary
        return {
            'image': image,
            'label': torch.tensor(class_id, dtype=torch.long),
            'bbox': torch.tensor(bbox, dtype=torch.float32)  # normalized bbox
        }

class CustomObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomObjectDetectionModel, self).__init__()
        # Define layers (same as before)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.bbox_regressor = nn.Sequential(
            
            nn.Linear(320, 128),
            nn.BatchNorm2d(3),
            nn.Flatten(),
            nn.Linear(122880, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 4)  # Output: (x_center, y_center, width, height)
        )

    def forward(self, x):
        features = self.features(x)
        class_logits = self.classifier(features)
        bbox_regression = self.bbox_regressor(x)
        return class_logits, bbox_regression
    

    def trainfuction(self, train_loader, val_loader, num_epochs=20, lr=0.001, device='cude'):
        # Move the model to the correct device (GPU/CPU)
        self.to(device)
        
        # Define optimizer and loss functions
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        classification_loss_fn = nn.CrossEntropyLoss()
        bbox_loss_fn = nn.SmoothL1Loss()

        train_losses = []
        val_losses = []
        train_classification_losses = []
        train_regression_losses = []

        for epoch in range(num_epochs):

            running_loss = 0.0
            train_classification_loss = 0.0
            train_regression_loss = 0.0
            self.train()  # Set the model to training mode
            total_accuracy = 0.0
            

            all_preds = []
            all_labels = []
            all_t_iou = []

            # Iterate over the training data
            for batch in tqdm(train_loader):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                bboxes = batch['bbox'].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                class_logits, bbox_regression = self(images)

                # Calculate losses
                classification_loss = classification_loss_fn(class_logits, labels)
                # Remove extra dimension from bbox if present
                bboxes = torch.squeeze(bboxes, dim=1)

                
                
                #bbox_loss = bbox_loss_fn(bbox_regression, bboxes)
                #bbox_loss = bbox_loss_fn(bbox_regression, bboxes)

                regression_loss = bbox_loss_fn(bbox_regression, bboxes)
                iou_loss_value = iou_loss(bbox_regression, bboxes)

                # Total loss is a weighted sum of classification and bbox regression loss
                total_regression_loss = 0.1 * regression_loss + 0.9 * iou_loss_value
                loss = classification_loss*0.1 + total_regression_loss*0.9

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy (this function should be defined elsewhere)
                batch_accuracy = self.calculate_accuracy(class_logits, labels)
                total_accuracy += batch_accuracy

                running_loss += loss.item() * images.size(0)
                train_classification_loss += classification_loss.item() * images.size(0)
                train_regression_loss += regression_loss.item() * images.size(0)

                preds = torch.argmax(class_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_t_iou.append(iou_loss_value.cpu().detach().numpy())
            
            avg_loss = running_loss / len(train_loader)
            avg_accuracy = total_accuracy / len(train_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy*100:.2f}%")
            self.validating(val_loader, device)
            #self.predict(r"root\dataset\test\images\all_images00107.jpg", device)

    def calculate_accuracy(self, class_logits, labels):
        # Assuming a simple accuracy function: percentage of correct classifications
        _, predicted = torch.max(class_logits, 1)
        correct = (predicted == labels).sum().item()
        return correct / len(labels)

    def validating(self, val_loader, device):
        self.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                bboxes = batch['bbox'].to(device)

                # Forward pass
                class_logits, bbox_regression = self(images)

                classification_loss = nn.CrossEntropyLoss()(class_logits, labels)
                bbox_loss = nn.SmoothL1Loss()(bbox_regression, bboxes)

                loss = classification_loss + bbox_loss
                total_loss += loss.item()

                batch_accuracy = self.calculate_accuracy(class_logits, labels)
                total_accuracy += batch_accuracy

        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy*100:.2f}%")
    
    def predict(self, img_path:str, device='cpu'):
        image  = Image.open(img_path).convert('RGB')
        transformsaug = transforms.ToTensor()
        imagetobeused = transformsaug(image).unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            class_logits, bbox = self.forward(imagetobeused)

        class_probs = torch.softmax(class_logits, dim=1)
        predicted_class = torch.argmax(class_probs, dim=1).item()
        redicted_class_prob = class_probs[0, predicted_class].item()
        predicted_bbox = bbox[0].cpu().numpy()

        x_center, y_center, width, height = predicted_bbox

        x_center = x_center*image.size[0]
        y_center = y_center*image.size[1]
        width = width*image.size[0]
        height = height*image.size[1] 

        #get the start Coordinates
        x_min = x_center - (width/2)
        y_min = y_center - (height/2)
        x_max = x_center + (width/2)
        y_max = y_center + (height/2)

        label_name = img_path.replace("/images", "/labels")  
        label_name = label_name.replace(".jpg", ".txt")      

        label_name = r"root\dataset\test\labels\all_images02764.txt"

        with open(label_name, 'r') as YoloData:
            line = YoloData.readline().strip()
            values = line.split()
            class_id = int(values[0])
            x_centerorig = float(values[1])
            y_centerorig = float(values[2])
            widthorig = float(values[3])
            heightorig = float(values[4])

        x_centerorig = x_centerorig*image.size[0]
        y_centerorig = y_centerorig*image.size[1]
        widthorig = widthorig*image.size[0]
        heightorig = heightorig*image.size[1] 

        #get the start Coordinates
        x_minorig = x_centerorig - (widthorig/2)
        y_minorig = y_centerorig - (heightorig/2)
        x_maxorig = x_centerorig + (widthorig/2)
        y_maxorig = y_centerorig + (heightorig/2)

        #Intersection Over Union
        x1 = max(x_min, x_minorig)
        x2 = min(x_max, x_maxorig)
        y1 = max(y_min, y_minorig)
        y2 = min(y_max, y_maxorig)
        
        # Compute the area of the intersection
        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)
        intersection_area = intersection_width * intersection_height

        box1_area = (x_max - x_min) * (y_max - y_min)
        box2_area = (x_maxorig - x_minorig) * (y_maxorig - y_minorig)

        union_area = box1_area + box2_area - intersection_area

        # Compute the IoU
        if union_area != 0:
            iou = intersection_area / (union_area + 1e-6)  # Add small epsilon to avoid division by zero
        else:
            iou = 0
        # print(iou)


        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        rectorig = Rectangle((x_minorig, y_minorig), widthorig, heightorig, linewidth=2, edgecolor='g', facecolor='none')

#Add the patch to the Axes
        ax.add_patch(rect)
        ax.add_patch(rectorig)
        plt.title(f"Class: {predicted_class}")
        plt.show()

        return {
            'class_id': predicted_class,
            'bbox': [x_min, y_min, x_max, y_max]
        }
            




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),  # Convert image to tensor
    ])

    train_dataset = ImageDataset(root_dir='root\\dataset\\train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=False)

    val_dataset = ImageDataset(root_dir='root\\dataset\\val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=False)

    #visualize_dataset(train_dataset, 2)

    #model preparation
    num_classes = 3  # Update based on your number of classes
    model = CustomObjectDetectionModel(num_classes)
    model.trainfuction(train_loader,val_loader, num_epochs=40, device=device)
    torch.save(model.state_dict(), 'model_weights_new_coordinate6.pth')


    # datapath = r"root\dataset\test\images"
    # imagespath = os.listdir(datapath)
    # print(imagespath)

    model.predict(r"root\dataset\test\images\all_images02764.jpg", device)
    sys.exit()

    model = CustomObjectDetectionModel(3)
    model.load_state_dict(torch.load('model_weights_new_coordinate2.pth'))

    TestPath = "root\\dataset\\test\\images"
    TestPathBB = "root\\dataset\\test\\labels"

    TestImages = os.listdir(TestPath)
    TestBB = os.listdir(TestPathBB)
# Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

# Set the model to evaluation mode
    model.eval()


    correctimages = 0

    all_targets = []
    all_predictions = []
# Predict on a new image
    for i in range(len(TestImages)):
        image_path = f'root/dataset/test/images/{TestImages[i]}'

    #This is YOLO format
        with open(f'root/dataset/test/labels/{TestBB[i]}', 'r') as YoloData:
            line = YoloData.readline().strip()
            values = line.split()
            realclass = int(values[0])
    

        prediction = model.predict(image_path, device=device)
    
        all_predictions.append(prediction["class_id"])
        all_targets.append(realclass)
        if prediction["class_id"] == realclass:
            correctimages += 1
    # Output the prediction
    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=1)
    recall = recall_score(all_targets, all_predictions, average='macro')
    all_targets_binary = label_binarize(all_targets, classes=np.unique(all_targets))
    all_predictions_binary = label_binarize(all_predictions, classes=np.unique(all_predictions))
    mAP = roc_auc_score(all_targets_binary, all_predictions_binary, average='macro', multi_class='ovr')
    testaccuracy = correctimages/len(TestImages)
    print(f"Accuracy on test set: {testaccuracy:.2f} ")
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, mAP: {mAP:.4f}')


    for i in range(5):
            image = random.choice(TestImages)
            image_path = f'root/dataset/test/images/{image}'
            imagecv2 = cv2.imread(image_path)
            prediction = model.predict(image_path, device=device)
            x_center, y_center, width, height = prediction['bbox']

            # x_center = x_center*imagecv2.shape[0]
            # y_center = y_center*imagecv2.shape[1]
            # width = width*imagecv2.shape[0]
            # height = height*imagecv2.shape[1] 

            x_min = x_center - (width/2)
            y_min = y_center - (height/2)
    
            fig, ax = plt.subplots(1)
            ax.imshow(imagecv2)
            rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
            ax.add_patch(rect)
            plt.title(f"Sample: {image}, Class: {prediction['class_id']}")
            plt.show()

    

def iou_loss(pred_boxes, true_boxes, img_width=640, img_height=640):
    """
    IoU Loss: 1 - IoU (since IoU is a value between 0 and 1, minimizing 1 - IoU will maximize IoU).
    Both pred_boxes and true_boxes should be in absolute coordinates [xmin, ymin, xmax, ymax].
    """
    # Create copies of the boxes to avoid in-place modification
    pred_boxes_scaled = pred_boxes.clone()
    true_boxes_scaled = true_boxes.clone()

    # Convert normalized bounding boxes to absolute pixel values (if applicable)
    pred_boxes_scaled[:, [0, 2]] *= img_width  # Convert x coordinates
    pred_boxes_scaled[:, [1, 3]] *= img_height  # Convert y coordinates
    true_boxes_scaled[:, [0, 2]] *= img_width  # Convert x coordinates
    true_boxes_scaled[:, [1, 3]] *= img_height  # Convert y coordinates

    # Ensure tensor shapes are correct before IoU calculation
    #print(f"Pred Boxes Shape: {pred_boxes_scaled.shape}, True Boxes Shape: {true_boxes_scaled.shape}")
    #print(f"Predicted Boxes: {pred_boxes_scaled}")
    #print(f"True Boxes: {true_boxes_scaled}")

    # Convert boxes from [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax]
    def convert_to_corners(boxes):
        """
        Convert boxes from [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax].
        """
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # xmax = xmin + width
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # ymax = ymin + height
        return boxes

    # Apply the conversion to both predicted and true boxes
    pred_boxes_converted = convert_to_corners(pred_boxes_scaled)
    true_boxes_converted = convert_to_corners(true_boxes_scaled)

    # Manual IoU computation for each box pair
    def calculate_iou(box1, box2):
        """
        Manually calculates the IoU (Intersection over Union) between two bounding boxes.
        Each box is represented by [xmin, ymin, xmax, ymax].
        """
        # Determine the coordinates of the intersection rectangle
        inter_xmin = max(box1[0], box2[0])
        inter_ymin = max(box1[1], box2[1])
        inter_xmax = min(box1[2], box2[2])
        inter_ymax = min(box1[3], box2[3])

        # Compute the area of the intersection rectangle
        inter_width = max(inter_xmax - inter_xmin, 0)
        inter_height = max(inter_ymax - inter_ymin, 0)
        inter_area = inter_width * inter_height

        # Compute the area of both the predicted and true rectangles
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute the union area
        union_area = box1_area + box2_area - inter_area

        # Compute IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    # Loop through each predicted and true box to compute the IoU
    ious = []
    for i in range(len(pred_boxes_converted)):
        pred_box = pred_boxes_converted[i]
        true_box = true_boxes_converted[i]
        iou = calculate_iou(pred_box, true_box)
        ious.append(iou)
        #print(f"IoU for predicted box {i}: {iou:.4f}")

    # Convert list of IoUs to a tensor and compute the mean for IoU loss
    iou_tensor = torch.tensor(ious, device=pred_boxes.device)
    iou_loss_value = 1 - iou_tensor.mean()

    return iou_loss_value




def visualize_dataset(dataset, numberofimages):
    ObjectDirectory = 'Object'  
    #Gather all the names of the objects
    ObjectNames = os.listdir(ObjectDirectory)



    ObjectID = {
        0: ObjectNames[0][:-4],
        1: ObjectNames[1][:-4],
        2: ObjectNames[2][:-4]
    }

    

    for i in range(numberofimages):
        imagenumber = random.randint(0, len(dataset)-1)
        sample = dataset[imagenumber]
        image = sample['image']
        classid = sample['label'].item()
        classname = ObjectID[classid]

        image = image.permute(1, 2, 0).numpy()
        x_center, y_center, width, height = sample['bbox']


        x_center = x_center*image.shape[0]
        y_center = y_center*image.shape[1]
        width = width*image.shape[0]
        height = height*image.shape[1] 

        #get the start Coordinates
        x_min = x_center - (width/2)
        y_min = y_center - (height/2)
        
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')

# Add the patch to the Axes
        ax.add_patch(rect)
        plt.title(f"Sample: {imagenumber}, Class: {classname}")
        plt.show()



if __name__ == "__main__":
    main()