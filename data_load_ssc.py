import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_ssc_augmentation():
    """
    Returns a torchvision.transforms Compose object for SSC augmentation.
    """
    ssc_augmentor = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return ssc_augmentor

def get_base_transform():
    """
    Returns the base transform for evaluation/test data.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class TwoViewImageFolder(datasets.ImageFolder):
    """
    A custom ImageFolder dataset that returns two augmented views of the same image.
    """
    def __init__(self, root, transform=None):
        # Initialize with no transform since we'll apply our own
        super().__init__(root, transform=None)
        self.augmentation = transform if transform else get_ssc_augmentation()

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # Apply two different augmentations to the same image
        img1 = self.augmentation(sample)
        img2 = self.augmentation(sample)
        
        return (img1, img2), target

def get_loader(name_dataset, batch_size, train=True, is_target=False):
   
    dataset_paths = {
        'amazon': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\amazon',
        'dslr': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\dslr',
        'webcam': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\webcam'
    }
    
    dataset_path = dataset_paths.get(name_dataset)
    if dataset_path is None:
        raise ValueError(f"Dataset {name_dataset} not found in defined paths")
    
    if train and is_target:
        # Use TwoViewImageFolder for target training data (SSC)
        transform = get_ssc_augmentation()
        dataset = TwoViewImageFolder(root=dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, drop_last=True)
    elif train:
        # Use standard ImageFolder for source training data
        transform = get_ssc_augmentation()
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, drop_last=False)
    else:
        # For evaluation/test data
        transform = get_base_transform()
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, drop_last=False)

    return loader


def get_source_loaders(source_names, target_name, batch_size):
    
    source_loaders = []
    for source_name in source_names:
        # Source training data
        source_loader = get_loader(source_name, batch_size, train=True, is_target=False)
        source_loaders.append(source_loader)
    
    # Target data - both labeled (if any) and unlabeled
    target_train_loader = get_loader(target_name, batch_size, train=True, is_target=True)
    target_test_loader = get_loader(target_name, batch_size, train=False, is_target=False)
    
    return source_loaders, target_train_loader, target_test_loader

source_names = ['amazon']
target_name = 'dslr'
batch_size = 32

source_loaders, target_train_loader, target_test_loader = get_source_loaders(
    source_names, target_name, batch_size
    
)

