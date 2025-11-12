import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
def get_loader(name_dataset, batch_size, train=True):
    dataset_paths = {
        'amazon': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\amazon',
        'dslr': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\dslr',
        'webcam': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\webcam'
    }
    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    dataset = ImageFolder(root=dataset_paths[name_dataset], transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)
    
    return loader