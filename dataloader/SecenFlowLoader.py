import torch.utils.data as data
import random
from PIL import Image
from . import preprocess
# import preprocess
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


# def disparity_loader(path):
#     path_prefix = path.split('.')[0]
#     # print(path_prefix)
#     path1 = path_prefix + '_exception_assign_minus_1.npy'
#     path2 = path_prefix + '.npy'
#     path3 = path_prefix + '.pfm'
#     import os.path as ospath
#     if ospath.exists(path1):
#         return np.load(path1)
#     else:
#         if ospath.exists(path2):
#             data = np.load(path2)
#         else:
#             # from readpfm import readPFMreadPFM
#             from readpfm import readPFM
#             data, _ = readPFM(path3)
#             np.save(path2, data)
#         for i in range(data.shape[0]):
#             for j in range(data.shape[1]):
#                 if j - data[i][j] < 0:
#                     data[i][j] = -1
#         np.save(path1, data)
#         return data


def disparity_loader(path):
    path_prefix = path.split('.')[0]
    # print(path_prefix)
    path1 = path_prefix + '_exception_assign_minus_1.npy'
    path2 = path_prefix + '.npy'
    path3 = path_prefix + '.pfm'
    import os.path as ospath
    if ospath.exists(path1):
        return np.load(path1)
    else:

        # from readpfm import readPFMreadPFM
        from readpfm import readPFM
        data, _ = readPFM(path3)
        np.save(path2, data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if j - data[i][j] < 0:
                    data[i][j] = -1
        np.save(path1, data)
        return data

class myImageFloder(data.Dataset):
    def __init__(self,
                 left,
                 right,
                 left_disparity,
                 training,
                 normalize,
                 loader=default_loader,
                 dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.normalize = normalize

    def __getitem__(self, index):
        
        left = self.left[index]
        
        right = self.right[index]
        disp_L = self.disp_L[index]
        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)
        
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        processed = preprocess.get_transform(
            augment=False, normalize=self.normalize)
        left_img = processed(left_img)
        right_img = processed(right_img)


        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
if __name__ == '__main__':
    path = '/media/lxy/sdd1/stereo_coderesource/dataset_nie/SceneFlowData/frames_cleanpass/flyingthings3d_disparity/TRAIN/A/0024/left/0011.pfm'
    res = disparity_loader(path)
    print(res.shape)
