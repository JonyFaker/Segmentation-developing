import torch.utils.data


def get_loader(name):
    return {
        'pascal': pascalVOCLoader,
        'sunrgbd': SUNRGBDLoader,
    }[name]


def get_data_path(name):
    return {
        'pascal': '/home/hjb/jf/workspace/datasets/VOC2012/',
        'sunrgbd': '/home/hjb/jf/workspace/datasets/SUNRGBD/',
    }[name]