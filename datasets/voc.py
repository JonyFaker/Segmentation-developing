from PIL import Image
import torch.utils.data as data
import os.path
import numpy as np
import torch
import collections


# def make_dataset(dir):
#     images = []
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir

#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in fnames:
#             if is_image_file(fname):
#                 path = os.path.join(root, fname)
#                 images.append(path)

#     return images


def make_dataset(name, filelist):
    files = collections.defaultdict(list)
    datalist = [l.strip('\n') for l in open(filelist).readlines()]
    for item in datalist:
        its = item.strip(' ')
        files[name] = [its[1], its[2]]

    return files

class pascalVOCLoader(data.Dataset):
    """docstring for pascalVOCLoader"""
    def __init__(self, arg):
        super(pascalVOCLoader, self).__init__()
        self.arg = arg
        self.root = arg.datadir
        self.trainlist = os.path.join(self.root, 'train.txt')
        self.testlist = os.path.join(self.root, 'test.txt')
        self.vallist = os.path.join(self.root, 'val.txt')
        self.argument = arg.argument
        self.transform = arg.transform
        self.img_norm = arg.img_norm
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.img_width = arg.width
        self.img_height = arg.height

        self.imgs_train = make_dataset('train', self.trainlist)
        self.lable_train = make_dataset('train', self.trainlist)
        self.imgs_test = make_dataset('test', self.testlist)
        self.label_test = make_dataset('test', self.testlist)
        self.imgs_val = make_dataset('val', self.vallist)
        self.label_val = make_dataset('val', self.vallist)


    def __getitem__(self, index):
        pass

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors """
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0],
                          [0,0,128], [128,0,128], [0,128,128], [128,128,128],
                          [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                          [64,0,128], [192,0,128], [64,128,128], [192,128,128],
                          [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
                          [0,64,128]])

    def name(self):
        return 'VOC2012'