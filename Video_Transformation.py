# Andrew Aquino
# DS681 - Assingment 1

import os
from PIL import Image
import torch  
from torch.utils.data import Dataset
from torchvision import transforms

class VideoFramesDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        # images sorted
        self.image_paths = sorted(
            [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        
        if self.transform:
            image = self.transform(image)

        return image

def main():
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames_directory = "video_frames"
    video_dataset = VideoFramesDataset(root_dir=frames_directory, transform=data_transforms)

    from torch.utils.data import DataLoader
    
    #load the data
    video_loader = DataLoader(video_dataset, batch_size=4, shuffle=True)

    print(f"Dataset has {len(video_dataset)} images.")
    first_batch = next(iter(video_loader))
    print(f"Shape of the first batch: {first_batch.shape}")
    print(f"Type of the first batch: {type(first_batch)}")

main()