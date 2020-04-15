from torch import nn
from torch.nn import functional as F

__all__ = ["FCN"]


class FCN(nn.Module):
    def __init__(self, backbone, classifier, skip_arch_0=None, skip_arch_1=None):
        super(FCN, self).__init__()
        self.n_classes = classifier.n_classes
        self.backbone = backbone
        self.classifier = classifier
        self.skip_arch_0 = skip_arch_0
        self.skip_arch_1 = skip_arch_1

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        x = features["out"]
        x = self.classifier(x)

        if self.skip_arch_0 is not None:
            x = self.skip_arch_0(x, features["pool4"])

            if self.skip_arch_1 is not None:
                x = self.skip_arch_1(x, features["pool3"])

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        self.n_classes = channels
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]
        super(FCNHead, self).__init__(*layers)


class SkipArch(nn.Module):
    def __init__(self, classifier, num_classes):
        super(SkipArch, self).__init__()
        self.classifier = classifier
        self.deconv = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2)

    def forward(self, x, feature):
        x = self.deconv(x)
        x += self.classifier(feature)
        return x


# skip architecture
# def _fuse(out, feature, classifier):
#     deconv = nn.ConvTranspose2d(out.shape[1], out.shape[1], 2, stride=2)
#     out = deconv(out)  # Deconvolution
#     x = classifier(feature)
#     x.add_(out)
#     return x
