'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
'''

import sys

from matplotlib import pyplot as plt

sys.path.append('.')
import pywt
import cv2
import yaml
import torch
import numpy as np
from copy import deepcopy
import albumentations as A
from training.dataset.albu import IsotropicResize
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.sbi_api import SBI_API


class SBIDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label == 0]

        # Init SBI
        self.sbi = SBI_API(phase=mode,image_size=config['resolution'])

        # Init data augmentation method
        self.transform = self.init_data_aug_method()

        self.w = "sym2"
        self.m = "reflect"

    def __getitem__(self, index):
        # print('11111111111')
        # Get the real image paths and labels
        real_image_path, real_label = self.real_imglist[index]

        # Get the landmark paths for real images
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        landmark = self.load_landmark(real_landmark_path).astype(np.int32)

        # Load the real images
        real_image = self.load_rgb(real_image_path)
        real_image = np.array(real_image)  # Convert to numpy array

        # Generate the corresponding SBI sample
        fake_image, real_image = self.sbi(real_image, landmark)
        # plt.imshow(fake_image)
        # plt.show()
        # fake_image = self.get_dwt_rgb(fake_image)
        # real_image = self.get_dwt_rgb(real_image)
        # print(self.get_dwt_rgb(fake_image).shape, self.get_dwt_rgb(real_image).shape,'______________')
        fake_eye,fake_nose,fake_mouse = self.produce_part_img(fake_image)
        # plt.imshow(fake_mouse)
        # plt.show()
        real_eye,real_nose,real_mouse = self.produce_part_img(real_image)
        # print(self.get_dwt_rgb(fake_mouse).shape, self.get_dwt_rgb(real_mouse).shape, self.get_dwt_rgb(fake_nose).shape, self.get_dwt_rgb(real_nose).shape )
        if fake_image is None:
            fake_image = deepcopy(real_image)
            fake_label = 0
        else:
            fake_label = 1

        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image))
        fake_mouse_trans = self.normalize(self.to_tensor(fake_mouse))
        fake_nose_trans = self.normalize(self.to_tensor(fake_nose))
        fake_eye_trans = self.normalize(self.to_tensor(fake_eye))
        real_image_trans = self.normalize(self.to_tensor(real_image))
        real_mouse_trans = self.normalize(self.to_tensor(real_mouse))
        real_nose_trans = self.normalize(self.to_tensor(real_nose))
        real_eye_trans = self.normalize(self.to_tensor(real_eye))

        return {"fake": (fake_image_trans, fake_label), 
                "real": (real_image_trans, real_label),
                'real_mouse':real_mouse_trans,
                'fake_mouse':fake_mouse_trans,
                'real_nose':real_nose_trans,
                'fake_nose':fake_nose_trans,
                'real_eye':real_eye_trans,
                'fake_eye':fake_eye_trans}

    def __len__(self):
        return len(self.real_imglist)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor and label tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images, fake_labels = zip(*[data["fake"] for data in batch])
        real_images, real_labels = zip(*[data["real"] for data in batch])
        fake_mouse = [data['fake_mouse'] for data in batch]
        real_mouse = [data['real_mouse'] for data in batch]
        fake_nose = [data['fake_nose'] for data in batch]
        real_nose = [data['real_nose'] for data in batch]
        fake_eye = [data['fake_eye'] for data in batch]
        real_eye = [data['real_eye'] for data in batch]

        # Stack the image, label, landmark, and mask tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)
        fake_mouse = torch.stack(fake_mouse, dim=0)
        real_mouse = torch.stack(real_mouse, dim=0)
        fake_nose = torch.stack(fake_nose, dim=0)
        real_nose = torch.stack(real_nose, dim=0)
        fake_eye = torch.stack(fake_eye, dim=0)
        real_eye = torch.stack(real_eye, dim=0)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        mouse = torch.cat([real_mouse, fake_mouse], dim=0)
        nose = torch.cat([real_nose, fake_nose], dim=0)
        eye = torch.cat([real_eye, fake_eye], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        
        data_dict = {
            'image': images,
            'mouse':mouse,
            'label': labels,
            'landmark': None,
            'mask': None,
            'nose':nose,
            'eye':eye

        }
        return data_dict

    def get_dwt_rgb(self, img):
        b, g, r = cv2.split(img.astype('float32'))

        cA_r, (_, _, _) = pywt.dwt2(r, self.w, mode=self.m)
        cA_g, (_, _, _) = pywt.dwt2(g, self.w, mode=self.m)
        cA_b, (_, _, _) = pywt.dwt2(b, self.w, mode=self.m)

        cA_r = cv2.resize(cA_r, (224,224), interpolation=cv2.INTER_LINEAR).astype('float32')
        cA_g = cv2.resize(cA_g, (224,224), interpolation=cv2.INTER_LINEAR).astype('float32')
        cA_b = cv2.resize(cA_b, (224,224), interpolation=cv2.INTER_LINEAR).astype('float32')

        cA_r = (cA_r + r) / 2
        cA_g = (cA_g + g) / 2
        cA_b = (cA_b + b) / 2

        img_dwt = np.stack([
            (cA_r + r) / 2,
            (cA_g + g) / 2,
            (cA_b + b) / 2
        ], axis=-1)


        return img_dwt

    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p = 0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            additional_targets={'real': 'sbi'},
        )
        return trans


if __name__ == '__main__':
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/sbi.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = SBIDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break