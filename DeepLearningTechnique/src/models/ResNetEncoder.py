import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


#Here I will use the ResNet34 architecture as the encoder backbone for our LaneNet.
#The ResNet34 architecture consist of these layers: 

#conv1, bn1, relu, maxpool -> this part is called the stem and it downsamples the input image by a factor of 4. 
#It has 64 output channels.

#layer1 (s4) -> it consists of 3 BasicBlocks (BasicBlocks consist of a conv layer, a batch norm layer, and a 
#ReLU activation) and it keeps the spatial resolution the same as the input to layer1 (which is after the stem, 
#so it's already downsampled by 4x). It outputs 64 channels.

#layer2 (s8) -> it consists of 4 BasicBlocks and it downsamples the spatial resolution by a factor of 2 
#(so it's downsampled by 8x compared to the original input image). It outputs 128 channels.

#layer3 (s16) -> it consists of 6 BasicBlocks and it downsamples the spatial resolution by a factor of 2 (so it's 
#downsampled by 16x compared to the original input image). It outputs 256 channels.

#layer4 (s32) -> it consists of 3 BasicBlocks and it downsamples the spatial resolution by a factor of 2 (so it's 
#downsampled by 32x compared to the original input image). It outputs 512 channels.

#After that, I will use a U-Net style decoder that takes the deepest features from layer4 (s32) and upsamples them step 
#by step, concatenating the corresponding skip features from layer3 (s16), layer2 (s8), and layer1 (s4) on the way 
#back to full resolution. This allows the decoder to leverage both high-level semantic information from the deepest 
#features and low-level spatial details from the earlier layers, which is crucial for accurate lane detection.

#I won't use the global average pool and fully connected classification head of ResNet34 because we don't need 
#image-level class scores, we need dense spatial features that a segmentation decoder can consume. So I will just 
#take the convolutional layers up to layer4 and expose the outputs of layer1, layer2, layer3, and layer4 as skip 
#connections for the decoder.
class ResNet34Encoder(nn.Module):
    """
    ResNet34 encoder that returns a list of skip features [s4, s8, s16, s32] for a
    U-Net-style decoder. Pretrained on ImageNet by default.
    """

    #Channel widths at each stride, in the same order returned by forward(). The decoder needs
    #to know these to size its concat layers, so we expose them as a class attribute instead of
    #magic numbers scattered around.
    OUT_CHANNELS = (64, 128, 256, 512)

    def __init__(self, pretrained=True):
        super().__init__()

        #ResNet34_Weights.DEFAULT picks the latest ImageNet weights torchvision has. If
        #pretrained=False we get a random init, which is useful for ablation studies but you
        #usually want the ImageNet weights since that's the whole point of switching to ResNet.
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        #Stem: 7x7 stride-2 conv + bn + relu + 3x3 stride-2 maxpool. This brings spatial
        #resolution down by 4x in total before layer1.
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        #Each layerN is a stack of BasicBlocks. layer1 keeps stride, layer2/3/4 each downsample
        #by 2x via the first block in each stage.
        self.layer1 = backbone.layer1   # stride 4,  64 channels
        self.layer2 = backbone.layer2   # stride 8,  128 channels
        self.layer3 = backbone.layer3   # stride 16, 256 channels
        self.layer4 = backbone.layer4   # stride 32, 512 channels

    def forward(self, x):
        #Run the stem once, then collect features at the end of each stage. We don't expose the
        #post-stem feature as a skip because layer1 doesn't downsample further so that feature
        #has the same spatial size as s4 but is less refined.
        x = self.stem(x)
        s4  = self.layer1(x)
        s8  = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)
        return [s4, s8, s16, s32]


    #Convenience helper. Useful when the training loop wants to freeze the pretrained backbone
    #for the first few epochs so the freshly-initialized decoder can catch up before we start
    #updating ImageNet weights. Call freeze() before training, unfreeze() once warmup is done.
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        #Switch BN layers to eval mode so their running stats stop updating while frozen.
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        #BN layers will go back to train mode the next time model.train() is called.
