import os
from PIL import Image
import random
from glob import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]



class FSAD_Dataset_train(Dataset):
    def __init__(self,
                 config,
                 is_train=True):
        
        self.category = config.data.category
        assert self.category in CLASS_NAMES, 'class_name: {}, should be in {}'.format(self.category, CLASS_NAMES)

        self.config = config
        self.is_train = is_train
        self.shot = config.data.shot
        
        self.batch_size = config.data.batch_size
        self.image_size = config.data.image_size
        self.query_dir, self.support_dir = self.load_dataset_folder()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size, Image.LANCZOS),
                transforms.ToTensor(), # Scales data into [0,1] 
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )


    def __len__(self):
        return len(self.query_dir)
    

    def __getitem__(self, index):
        
        query_img = None
        support_img = None
        support_sub_img = None
        query_list, support_list = self.query_dir[index], self.support_dir[index]


        for i in range(len(query_list)):
            image = Image.open(query_list[i]).convert('RGB')
            image = self.image_transform(image) # [3, 224, 224]
            image = image.unsqueeze(dim = 0) # [1, 3, 224, 224]

            if query_img is None:
                query_img = image
            else:
                query_img = torch.cat([query_img, image],dim=0)

            for k in range(self.shot):
                image = Image.open(support_list[i][k]).convert('RGB')
                image = self.image_transform(image) # [3, 224, 224]
                image = image.unsqueeze(dim = 0) # [1, 3, 224, 224]

                if support_sub_img is None:
                    support_sub_img = image
                else:
                    support_sub_img = torch.cat([support_sub_img, image], dim=0)

            support_sub_img = support_sub_img.unsqueeze(dim=0) # [1, shot, 3, 224, 224]
            if support_img is None:
                support_img = support_sub_img
            else:
                support_img = torch.cat([support_img, support_sub_img], dim=0)

            support_sub_img = None

        mask = torch.zeros([self.batch_size,  self.image_size,  self.image_size])
        return query_img, support_img, mask

    
    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'

        data_img = {}

        for class_name_one in CLASS_NAMES:
            if class_name_one != self.category:
                data_img[class_name_one] = []

                img_dir = os.path.join(self.config.data.data_dir, self.category, phase, 'good')
                sorted_imgs = sorted(os.listdir(img_dir))

                for img in sorted_imgs:
                    img_path = os.path.join(img_dir, img)
                    data_img[class_name_one].append(img_path)
                
                random.shuffle(data_img[class_name_one])
            
        query_dir, support_dir = [], []

        for class_name_one in data_img.keys():
            for image_index in range(0, len(data_img[class_name_one]), self.batch_size):
                query_sub_dir = []
                support_sub_dir = []
            
                for batch_count in range(0, self.batch_size):
                    if image_index + batch_count >= len(data_img[class_name_one]):
                        break
                    
                    # build query set
                    image_dir_one = data_img[class_name_one][image_index + batch_count]
                    query_sub_dir.append(image_dir_one)

                    # build support set
                    support_dir_one = []

                    for k in range(self.shot):
                        # 每次选出与 query set 不同的图像添加
                        random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        while data_img[class_name_one][random_choose] == image_dir_one:
                            random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        
                        support_dir_one.append(data_img[class_name_one][random_choose]) # [shot]

                    support_sub_dir.append(support_dir_one) # [batch_size, shot]
                
                support_dir.append(support_sub_dir) # [num_batch, batch_size, shot]
                query_dir.append(query_sub_dir) # [num_batch,batch_size]         
        
        assert len(query_dir) == len(support_dir), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir

                





