import torch
import torch.nn.functional as F
from torchvision import models, transforms
import torch.nn as nn

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True, layer_ids=(3, 8, 15), weights=None):
        """
        Args:
            layer_ids: indices of VGG16 features to extract from (e.g., relu1_2, relu2_2, relu3_3).
            weights: optional weighting for each layer loss.
        """
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.layer_ids = layer_ids
        self.layers = nn.ModuleList([self.vgg[i] for i in range(max(layer_ids) + 1)])
        self.resize = resize
        self.weights = weights if weights else [1.0] * len(layer_ids)

        # Normalize to match ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        # Ensure input size and normalization
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)

        x = torch.stack([self.normalize(xi) for xi in x])
        y = torch.stack([self.normalize(yi) for yi in y])

        loss = 0.0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                layer_idx = self.layer_ids.index(i)
                loss += self.weights[layer_idx] * F.mse_loss(x, y)

        return loss
