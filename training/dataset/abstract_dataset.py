# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys
import dlib
import lmdb
import pywt
from imutils.face_utils import FaceAligner
from matplotlib import pyplot as plt
from transformers import CLIPProcessor

sys.path.append('.')
import os
import math
import yaml
import glob
import json
import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T
import albumentations as A
from.albu import IsotropicResize

FFpp_pool=['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT']#

def all_in_pool(inputs,pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """

        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.processor = CLIPProcessor.from_pretrained("/media/gpu/1cd8a4a1-b4ef-4f56-ac4d-10dff3dc7b13/zzx4985/DeepfakeBench/DeepfakeBench-main/pretrained_model/clip-vit-base-patch16")

        self.lmdb = config.get('lmdb', False)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
            if self.lmdb:
                if len(dataset_list)>1:
                    if all_in_pool(dataset_list,FFpp_pool):
                        lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
                        self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                    else:
                        raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
                else:
                    lmdb_path = os.path.join(config['lmdb_dir'], f"{dataset_list[0] if dataset_list[0] not in FFpp_pool else 'FaceForensics++'}_lmdb")
                    self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        elif mode == 'test':
            one_data = config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list = self.collect_img_and_label_for_one_dataset(one_data)
            if self.lmdb:
                lmdb_path = os.path.join(config['lmdb_dir'], f"{one_data}_lmdb" if one_data not in FFpp_pool else 'FaceForensics++_lmdb')
                self.env = lmdb.open(lmdb_path, create=True, subdir=True, readonly=True, lock=False)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list


        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        self.transform = self.init_data_aug_method()
        self.w = "sym2"
        self.m = "reflect"
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
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        return rescaled_landmarks


    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.
        
        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []
        
        # Record video name for video-level metrics
        video_name_list = []

        # Try to get the dataset information from the JSON file
        if not os.path.exists(self.config['dataset_json_folder']):
            self.config['dataset_json_folder'] = self.config['dataset_json_folder'].replace('/Youtu_Pangu_Security_Public', '/Youtu_Pangu_Security/public')
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        # If JSON file exists, do the following data collection
        # FIXME: ugly, need to be modified here.
        cp = None
        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'
        elif dataset_name == 'FF-DF_c40':
            dataset_name = 'FF-DF'
            cp = 'c40'
        elif dataset_name == 'FF-F2F_c40':
            dataset_name = 'FF-F2F'
            cp = 'c40'
        elif dataset_name == 'FF-FS_c40':
            dataset_name = 'FF-FS'
            cp = 'c40'
        elif dataset_name == 'FF-NT_c40':
            dataset_name = 'FF-NT'
            cp = 'c40'
        # Get the information for the current dataset
        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                sub_dataset_info = sub_dataset_info['c40']

            # Iterate over the videos in the dataset
            for video_name, video_info in sub_dataset_info.items():
                # Unique video name
                unique_video_name = video_info['label'] + '_' + video_name

                # Get the label and frame paths for the current video
                if video_info['label'] not in self.config['label_dict']:
                    raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                label = self.config['label_dict'][video_info['label']]
                frame_paths = video_info['frames']
                if not frame_paths:

                    continue

                    # 健壮的排序逻辑
                try:
                    frame_paths = sorted(frame_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                except ValueError:
                    frame_paths = sorted(frame_paths, key=lambda x: os.path.basename(x))
                # print(frame_paths[0])
                # sorted video path to the lists
                # if '\\' in frame_paths[0]:
                #     frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                # else:
                #     frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))

                # Consider the case when the actual number of frames (e.g., 270) is larger than the specified (i.e., self.frame_num=32)
                # In this case, we select self.frame_num frames from the original 270 frames
                total_frames = len(frame_paths)
                if self.frame_num < total_frames:
                    total_frames = self.frame_num
                    if self.video_level:
                        # Select clip_size continuous frames
                        start_frame = random.randint(0, total_frames - self.frame_num) if self.mode == 'train' else 0
                        frame_paths = frame_paths[start_frame:start_frame + self.frame_num]  # update total_frames
                    else:
                        # Select self.frame_num frames evenly distributed throughout the video
                        step = total_frames // self.frame_num
                        frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]
                
                # If video-level methods, crop clips from the selected frames if needed
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError('clip_size must be specified when video_level is True.')
                    # Check if the number of total frames is greater than or equal to clip_size
                    if total_frames >= self.clip_size:
                        # Initialize an empty list to store the selected continuous frames
                        selected_clips = []

                        # Calculate the number of clips to select
                        num_clips = total_frames // self.clip_size

                        if num_clips > 1:
                            # Calculate the step size between each clip
                            clip_step = (total_frames - self.clip_size) // (num_clips - 1)

                            # Select clip_size continuous frames from each part of the video
                            for i in range(num_clips):
                                # Ensure start_frame + self.clip_size - 1 does not exceed the index of the last frame
                                start_frame = random.randrange(i * clip_step, min((i + 1) * clip_step, total_frames - self.clip_size + 1)) if self.mode == 'train' else i * clip_step
                                continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                assert len(continuous_frames) == self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                                selected_clips.append(continuous_frames)

                        else:
                            start_frame = random.randrange(0, total_frames - self.clip_size + 1) if self.mode == 'train' else 0
                            continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                            assert len(continuous_frames)==self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                            selected_clips.append(continuous_frames)

                        # Append the list of selected clips and append the label
                        label_list.extend([label] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        # video name save
                        video_name_list.extend([unique_video_name] * len(selected_clips))

                    else:
                        print(f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames}).")
                
                # Otherwise, extend the label and frame paths to the lists according to the number of frames
                else:
                    # Extend the label and frame paths to the lists according to the number of frames
                    label_list.extend([label] * total_frames)
                    frame_path_list.extend(frame_paths)
                    # video name save
                    video_name_list.extend([unique_video_name] * len(frame_paths))
            
        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)
        
        return frame_path_list, label_list, video_name_list

     
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution']
        # file_path = file_path.replace('\\', '/')
        # if self.mode == "train" else self.config['resolution']
        if not self.lmdb:
            # 跨平台路径处理
            if not file_path.startswith('.'):
                # print(f"Non-LMDB loading path: {file_path}")
                file_path = os.path.join('.', self.config['rgb_dir'], file_path)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            # print(f"Non-LMDB loading path: {file_path}")
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Failed to decode image: {file_path}")
        else:
            with self.env.begin(write=False) as txn:

                file_path = file_path.replace(os.path.join('.', 'datasets'), '')
                image_bin = txn.get(file_path.encode())

                if not image_bin:
                    raise ValueError(f"LMDB key not found: {file_path}")

                img = cv2.imdecode(np.frombuffer(image_bin, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode LMDB image: {file_path}")


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        interpolation = cv2.INTER_AREA if (img.shape[0] > size or img.shape[1] > size) else cv2.INTER_LINEAR
        img = cv2.resize(img, (size, size), interpolation=interpolation)

        return Image.fromarray(img)

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        file_path = file_path.replace('\\', '/')
        if file_path is None:
            return np.zeros((size, size, 1))
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  f'./{self.config["rgb_dir"]}\\'+file_path
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    mask = np.zeros((size, size,3))
                else:
                    image_buf = np.frombuffer(image_bin, dtype=np.uint8)

                    mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        # file_path = file_path.replace('\\', '/')
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if not self.lmdb:
            if not file_path[0] == '.':
                prefix_a = os.path.join('.',self.config['rgb_dir'])
                file_path =  os.path.join(prefix_a,file_path)
            if os.path.exists(file_path):

                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')
                # print(file_path)
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
                landmark=self.rescale_landmarks(np.float32(landmark), original_size=256, new_size=self.config['resolution'])
        return landmark

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"Invalid image dimensions: {img.shape}")


        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)


        kwargs = {'image': img}
        if landmark is not None:

            landmark = np.clip(landmark, 0, [w - 1e-6, h - 1e-6])
            kwargs['keypoints'] = landmark.tolist() if isinstance(landmark, np.ndarray) else landmark
            kwargs['keypoint_params'] = A.KeypointParams(
                format='xy',
                remove_invisible=True,
                angle_in_degrees=False
            )
        if mask is not None:
            mask = mask.squeeze()
            kwargs['mask'] = mask if mask.max() > 0 else np.zeros_like(img[:, :, 0])


        try:
            transformed = self.transform(**kwargs)
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return img, np.zeros((0, 2), dtype=np.float32), kwargs.get('mask', np.zeros_like(img[:, :, 0]))


        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask', kwargs.get('mask'))


        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)
            if augmented_landmark.size > 0:
                new_h, new_w = augmented_img.shape[:2]
                eps = 1e-3
                augmented_landmark = np.clip(
                    augmented_landmark,
                    [0 + eps, 0 + eps],
                    [new_w - eps, new_h - eps]
                )


        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def resize_with_padding(self,img: np.ndarray, target_size=(224, 224), pad_color=(255,255,255)) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        h, w = img.shape[:2]
        if w == 0 or h == 0:
            print('--------------------------')
            return np.full((target_size[1], target_size[0], 3), pad_color, dtype=np.uint8)

        target_w, target_h = target_size
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top


        padded_img = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=pad_color
        )

        return padded_img

    def random_crop(self, img, crop_size, h, w, center_ratio=0.6):
        max_x = w - crop_size
        max_y = h - crop_size

        if max_x <= 0 or max_y <= 0:
            return img


        center_range_x = int(max_x * center_ratio)
        center_range_y = int(max_y * center_ratio)


        center_start_x = (max_x - center_range_x) // 2
        center_start_y = (max_y - center_range_y) // 2

        x = np.random.randint(center_start_x, center_start_x + center_range_x)
        y = np.random.randint(center_start_y, center_start_y + center_range_y)

        return img[y:y + crop_size, x:x + crop_size]

    def produce_part_img(self, img):

        h, w, _ = img.shape

        crop_150 = self.random_crop(img, 200, h, w,0.8)
        crop_150 = self.resize_with_padding(crop_150)
        crop_50 = self.random_crop(crop_150, 130, h, w,0.8)
        crop_50 = self.resize_with_padding(crop_50 )
        crop_100 = self.random_crop(crop_50, 160, h, w,0.7)
        crop_100 = self.resize_with_padding(crop_100)
        return crop_50, crop_100, crop_150


    def get_dwt_rgb(self, img):
        b, g, r = cv2.split(img)

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
    def __getitem__(self, index, no_norm=False):

        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]
        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)
            image = np.array(image)
            # image = self.get_dwt_rgb(image)
            # print(image.shape)# Convert to numpy array for data augmentation
            if self.config['with_mask']:
                try:
                    mask = self.load_mask(mask_path)
                except Exception as e:

                    image_height, image_width = image.shape[:2]
                    mask = np.zeros((image_height, image_width), dtype=np.float32)
            else:
                image_height, image_width = image.shape[:2]
                mask = np.zeros((image_height, image_width), dtype=np.float32)
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
                if landmarks is None:
                    landmarks = np.zeros((0, 2), dtype=np.float32)
            else:
                landmarks = None

            eye_img, nose_img, mouse_img = self.produce_part_img(image)

            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
                eye_image_trans,_,_ = self.data_aug(eye_img,landmarks, mask, augmentation_seed)
                nose_image_trans,_,_ = self.data_aug(nose_img,landmarks, mask, augmentation_seed)
                mouse_image_trans,_,_ = self.data_aug(mouse_img,landmarks, mask, augmentation_seed)

            else:
                eye_image_trans,nose_image_trans,mouse_image_trans,image_trans, landmarks_trans, mask_trans =deepcopy(eye_img),deepcopy(nose_img),deepcopy(mouse_img), deepcopy(image), deepcopy(landmarks), deepcopy(mask)

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                eye_image_trans = self.normalize(self.to_tensor(eye_image_trans))
                nose_image_trans = self.normalize(self.to_tensor(nose_image_trans))
                mouse_image_trans = self.normalize(self.to_tensor(mouse_image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)
                landmark_tensors.append(landmarks_trans)
                mask_tensors.append(mask_trans)

            # Get the first landmark and mask tensors
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_trans,eye_image_trans,nose_image_trans,mouse_image_trans, label, landmark_tensors, mask_tensors

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, eye_images, nose_images, mouse_images,labels, landmarks_list, masks_list = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        eye_images = torch.stack(eye_images, dim=0)
        nose_images = torch.stack(nose_images, dim=0)
        mouse_images = torch.stack(mouse_images, dim=0)
        labels = torch.LongTensor(labels)

        # Special case for landmarks and masks if they are None
        if not all(landmark is None for landmark in landmarks_list):
            valid_landmark = [lm for lm in landmarks_list if lm is not None]
            if len(valid_landmark)>0:
                max_len = max([lm.shape[0] for lm in valid_landmark])
                padded_landmarks = [
                    torch.cat([lm,torch.full((max_len-lm.shape[0],2),-1,dtype=torch.float32)],dim=0)
                    for lm in valid_landmark
                ]
                landmarks = torch.stack(padded_landmarks, dim=0)
            else:
                landmarks = None
        else:
            landmarks = None

        if not all(m is None for m in masks_list):
            masks = []
            for m in masks_list:
                if isinstance(m, (list, np.ndarray)):
                    m = torch.from_numpy(np.array(m)) if isinstance(m, list) else torch.from_numpy(m)
                masks.append(m)
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {
            'image': images,
            'eye': eye_images,
            'nose': nose_images,
            'mouse': mouse_images,
            'label': labels,
            'landmark': landmarks,
            'mask': masks

        }
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)
