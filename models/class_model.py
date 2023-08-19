from models.wideresnet import build_wideresnet as WideResNet


def class_model(modelType, num_classes=10, mode="default", dataset=None, nograd=False):
    if "WideResNet" in modelType:
        if dataset == "CIFAR10":
            model = WideResNet(mode=mode, depth=28, widen_factor=2, dropout=0, num_classes=num_classes).cuda()
        elif dataset == "CIFAR100":
            model = WideResNet(mode=mode, depth=28, widen_factor=6, dropout=0, num_classes=num_classes).cuda()
        elif dataset == "SVHN":
            model = WideResNet(mode=mode, depth=28, widen_factor=2, dropout=0, num_classes=num_classes).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model