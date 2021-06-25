import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision.transforms import ToPILImage


# TODO move get mask here
class PILToLongTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.long()
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        return img.transpose(0, 1).transpose(0,
                                             2).contiguous().long().squeeze_()


class LongTensorToRGBPIL(object):
    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(tensor)))
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)
        size = list(tensor.size())
        color_tensor = torch.ByteTensor(3, size[1], size[2])
        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            mask = torch.eq(tensor, index).squeeze_()
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)
        return ToPILImage()(color_tensor)
