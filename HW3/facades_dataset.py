import torch
from torch.utils.data import Dataset
import cv2
import os  

'''
该类用于读取图像文件，并将它们转换为 PyTorch 张量
'''
class FacadesDataset(Dataset):
    def __init__(self, path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')  
    
        # 使用列表推导式获取所有图片文件  
        images = [f for f in os.listdir(path) if f.lower().endswith(image_extensions)]  
        
        # 生成完整路径  
        self.image_filenames = [os.path.join(path, image) for image in images]  
        
    def __len__(self): #获取数据集大小
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx): # 获取特定图像
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        # Convert the image to a PyTorch tensor
        # image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0
        image_rgb = image[:, :, :256]
        image_semantic = image[:, :, 256:]
        return image_rgb, image_semantic