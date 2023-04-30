from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenetv1 import MobileNetV1


imagenet_model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "MobileNetV1": MobileNetV1,
}
