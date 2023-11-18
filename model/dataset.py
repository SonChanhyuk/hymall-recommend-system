from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import os

class ClothImageDataset(Dataset):
    def __init__(self, csv_name, train, transform=transforms.Resize(224), target_transform=None):
        csv = pd.read_csv(csv_name)
        self.img_labels = csv[["path","id"]]
        
        self.images, self.labels = [], []
        for i in range(len(self.img_labels)):
            self.images.append(self.img_labels["path"][i])
            self.labels.append(int(self.img_labels["id"][i][3:]))
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transform = self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        label = self.labels[idx]
        
        sample = (image, label, img_path)
        return sample

class CommerceImageDataset(Dataset):
    def __init__(self, transform=transforms.Resize(224), target_transform=None):
        self.path_name = "commerce_imgs"
        self.commerce_data = os.listdir(self.path_name)
                
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transform = self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.commerce_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_name, self.commerce_data[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        label = img_path[:-4]
        
        sample = (image, label)
        return sample