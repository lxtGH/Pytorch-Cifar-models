from models.vgg import VGG
from models.densenet import ChooseDenseNet
from models.dpn import DPN
from models.mobilenet import MobileNet
from models.resnet import ResNet

class Model(object):
    @staticmethod
    def model(key):
        key = key.lower()
        if key[:3] == 'vgg':
            return VGG("VGG"+key[3:])
        if key[:3] =="dpn":
            return DPN(key)
        if key[:8] == 'densenet':
            return ChooseDenseNet(key[8:])


if __name__ == "__main__":
    from models.model import Model
    model = Model.model('vgg')