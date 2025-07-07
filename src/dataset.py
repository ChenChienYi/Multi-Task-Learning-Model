
import torch
from torchvision import transforms
from PIL import Image
import os

class CelebAMultiTaskDataset(Dataset):
    def __init__(self, img_dir, attrs_df, landmarks_df, partition_df, partition_type='train', n_sample = 1000):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # ImageNet stats
                                 ])
        
        self.file_list = partition_df[partition_df['partition'] == {'train': 0, 'val': 1, 'test': 2}[partition_type]]['image_id'].tolist()[:n_sample]
        self.attrs_df = attrs_df.set_index('image_id').loc[self.file_list]
        self.landmarks_df = landmarks_df.set_index('image_id').loc[self.file_list]

        # Convert attribute labels from -1/1 to 0/1
        self.attrs_df = (self.attrs_df + 1) / 2

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        attributes = torch.tensor(self.attrs_df.loc[img_name].values, dtype=torch.float32)
        landmarks = torch.tensor(self.landmarks_df.loc[img_name].values, dtype=torch.float32)
        original_width, original_height = 178, 218

        if self.transform:
            image = self.transform(image)

            landmarks[0::2] = landmarks[0::2] / original_width * 224.0
            landmarks[1::2] = landmarks[1::2] / original_height * 224.0
            landmarks = landmarks / 224.0

        return image, attributes, landmarks