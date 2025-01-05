from torch import nn
import torchvision
from torchvision.models import ViT_B_16_Weights


class VisionTransformer(nn.Module):
    """
    torchvision's Vision Transformer (ViT) model.

    Args:
        num_classes: An integer indicating the number of output classes.
        pretrained: A boolean indicating whether to use pretrained weights.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        # Load pre-trained ViT model
        if pretrained:
            self.vit_weights = ViT_B_16_Weights.DEFAULT
            self.model = torchvision.models.vit_b_16(weights=self.vit_weights)
        else:
            self.model = torchvision.models.vit_b_16(weights=None)

        # Freeze all layers except the classification head
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classification head with a custom one
        self.model.heads = nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

    def get_transforms(self):
        """
        Returns the required transforms for data preprocessing.
        Only works if pretrained weights are used.
        """
        if hasattr(self, "vit_weights"):
            return self.vit_weights.transforms()
        else:
            raise ValueError("Transforms are only available for pretrained models.")
