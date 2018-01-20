


from models.vgg import VGG
from models.densenet import DenseNet


class Model(object):

    @staticmethod
    def model(key):
        if key == 'VGG':
            return VGG()

        if key == 'densenet':
            return DenseNet()


if __name__ == "__main__":
    from models.model import Model
    model = Model.model('vgg')