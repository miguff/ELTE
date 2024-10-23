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
from torchvision.ops import generalized_box_iou_loss





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
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss = 0.0
            total_accuracy = 0.0

            # Iterate over the training data
            for batch in tqdm(train_loader):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                bboxes = batch['bbox'].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                class_logits, bbox_regression = self(images)

                # Remove extra dimension from bbox if present
                bboxes = torch.squeeze(bboxes, dim=1)

                # Calculate losses
                classification_loss = classification_loss_fn(class_logits, labels)
                
                #bbox_loss = bbox_loss_fn(bbox_regression, bboxes)
                bbox_loss = bbox_loss_fn(bbox_regression, bboxes)

                # Total loss is a weighted sum of classification and bbox regression loss
                loss = classification_loss + bbox_loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy (this function should be defined elsewhere)
                batch_accuracy = self.calculate_accuracy(class_logits, labels)
                total_accuracy += batch_accuracy
            print(intersection_over_union(bbox_regression, bboxes))
            avg_loss = running_loss / len(train_loader)
            avg_accuracy = total_accuracy / len(train_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy*100:.2f}%")
            self.validating(val_loader, device)
            self.predict(r"root\dataset\test\images\all_images00107.jpg", device)

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
    
    def predict(self, img_path, device='cpu'):
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

        
        label_name = r"root\dataset\test\labels\all_images00107.txt"
       

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
        print(iou)


        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        rectorig = Rectangle((x_minorig, y_minorig), widthorig, heightorig, linewidth=2, edgecolor='g', facecolor='none')

# Add the patch to the Axes
        ax.add_patch(rect)
        ax.add_patch(rectorig)
        plt.title(f"Class: {predicted_class}")
        plt.show()
            




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
    model.trainfuction(train_loader,val_loader, num_epochs=20, device=device)
    torch.save(model.state_dict(), 'model_weights_new_coordinate2.pth')
    model.predict(r"root\dataset\test\images\all_images00107.jpg", device)
    

def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculate Intersection over Union (IoU) for bounding boxes.
    
    Args:
    - boxes_preds (tensor): Predicted bounding boxes (B, 4) -> (x_center, y_center, width, height)
    - boxes_labels (tensor): Ground truth bounding boxes (B, 4) -> (x_center, y_center, width, height)

    Returns:
    - IoU (tensor): IoU for each example in the batch.
    """
    # Convert from center (x, y, w, h) to corners (x1, y1, x2, y2)
    box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2
    
    box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    # Find the intersection box
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Clamp the values at 0 to prevent negative areas
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate the area of both the prediction and ground-truth boxes
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Union area
    union = box1_area + box2_area - intersection

    # IoU
    iou = intersection / (union + 1e-6)  # Add a small value to avoid division by zero
    return iou


def iou_loss(pred_boxes, target_boxes):
    """
    Calculate IoU loss.
    
    Args:
    - pred_boxes (tensor): Predicted bounding boxes (B, 4) -> (x_center, y_center, width, height)
    - target_boxes (tensor): Ground truth bounding boxes (B, 4) -> (x_center, y_center, width, height)

    Returns:
    - IoU loss (tensor): The IoU loss for each prediction.
    """
    iou = intersection_over_union(pred_boxes, target_boxes)
    return (1 - iou).min()  # IoU Loss is 1 - IoU




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