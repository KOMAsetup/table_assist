import segmentation_models_pytorch as smp
import torchvision


model_masrrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)


model_unet = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=1,  # один класс
    activation="sigmoid"  # для бинарной маски
)


model_resnet = smp.DeepLabV3(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    classes=1,
    activation="sigmoid"
)

