from src.common.lib import *
from src.common.dir import *
from src.common.var import *
from src.utils.utils import *
from src.pipeline.image_processing import ImageProcessing

class StrepDataset(Dataset):
    def __init__(self, csv, img_dir, transform = None):
        self.data = read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        features = torch.tensor(self.data.iloc[idx, 2:9])

        label = torch.tensor(self.data.iloc[idx, 1])

        return image, features, label

#################################################################################

class CreateDataloader:
    def __init__(self, train_idx, test_idx, val_idx, dataset):
        self.transforms_train = transforms.Compose([])
        self.transforms_test = transforms.Compose([])
        self.ip = ImageProcessing()
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.val_idx = val_idx
        self.dataset = dataset
    
    def create_transforms(self):
        self.transforms_train = transforms.Compose([
        transforms.Lambda(lambda x: self.ip.white_balance(x)),
        transforms.Lambda(lambda x: self.ip.apply_clahe(x)),
        transforms.RandomResizedCrop(IMG_RESIZE_DIM, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transforms_test = transforms.Compose([
            transforms.Lambda(lambda x: self.ip.white_balance(x)),       ## Fix the blur
            transforms.Lambda(lambda x: self.ip.apply_clahe(x)),         ## Brighten dark areas
            transforms.Resize(IMG_RESIZE_DIM),
            transforms.CenterCrop(IMG_CROP_DIM),                         ## Crop to focus only on the tonsil area, disregard tongue / teeth
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def create_dataloader(self):
        
        self.create_transforms() 

        ## Created datasets for the train and test data with individual transforms
        if self.dataset == "cnh":
            train_dataset = StrepDataset(csv=UPDATED_CSV_FILE, img_dir=IMG_DIR, transform=self.transforms_train)
            test_dataset = StrepDataset(csv=UPDATED_CSV_FILE, img_dir=IMG_DIR, transform=self.transforms_test)

        else:
            train_dataset = StrepDataset(csv=KAGGLE_CSV_FILE, img_dir=KAGGLE_IMG_DIR, transform=self.transforms_train)
            test_dataset = StrepDataset(csv=KAGGLE_CSV_FILE, img_dir=KAGGLE_IMG_DIR, transform=self.transforms_test)

        ## Created Subsets for the train and test dataset indices
        train_subset = Subset(train_dataset, self.train_idx)
        test_subset = Subset(test_dataset, self.test_idx)
        val_subset = Subset(test_dataset, self.val_idx)

        ## Created Dataloaders for the train and test dataset 
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=True)


        return train_loader, test_loader, val_loader