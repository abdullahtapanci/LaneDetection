import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import ENetEncoder, RegularBottleneck, UpsamplingBottleneck
from src.models.ResNetEncoder import ResNet34Encoder


class LaneNetDecoder(nn.Module):
    """
    Decoder branch for LaneNet, following ENet stages 4 and 5 plus a final
    transposed convolution that projects to the desired number of output
    channels.

    The same architecture is used for both LaneNet output heads:
    - Binary segmentation branch: out_channels = 2 (background, lane)
    - Instance embedding branch:  out_channels = embedding_dim (e.g. 4)

    Two instances of this class are created in LaneNet — one per head —
    so the weights are independent even though the structure is identical.

    Keyword arguments:
    - out_channels (int): number of output channels for the final convolution.
    """

    def __init__(self, out_channels):
        super().__init__()

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, dropout_prob=0.1)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, dropout_prob=0.1)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

    def forward(self,
                features,
                max_indices1_0,
                max_indices2_0,
                stage1_input_size,
                stage2_input_size,
                input_size):

        # Stage 4 - Decoder
        x = self.upsample4_0(features, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)

        x = self.transposed_conv(x, output_size=input_size)

        return x
    

class LaneNet(nn.Module):
    """
    LaneNet architecture combining the ENet encoder with two separate
    decoder branches for binary segmentation and instance embedding.

    Keyword arguments:
    - embedding_dim (int): number of channels in the instance embedding output.
    """

    def __init__(self, embedding_dim=4):
        super().__init__()

        self.encoder = ENetEncoder()
        self.binary_decoder = LaneNetDecoder(out_channels=2)
        self.embedding_decoder = LaneNetDecoder(out_channels=embedding_dim)

    def forward(self, x):
        #Encoder returns the embeddings from the last layer of the encoder, the max pooling indices from 
        #stages 1 and 2, the input sizes for stages 1 and 2, and the original input size.
        features, idx1, idx2, s1, s2, inp = self.encoder(x)

        binary_output = self.binary_decoder(
            features, idx1, idx2, s1, s2, inp)
        embedding_output = self.embedding_decoder(
            features, idx1, idx2, s1, s2, inp)

        return binary_output, embedding_output
    

#This is one step of the U-Net decoder. It takes a feature map at some stride, upsamples it 2x by bilinear
#interpolation, optionally concatenates a skip feature from the encoder at the same resolution,
#then runs a small conv block (3x3 conv -> BN -> ReLU, twice) to fuse them.

#Two design choices worth noting:
# 1. Bilinear upsample + conv is preferred over ConvTranspose2d here because transposed conv
#    tends to introduce checkerboard artifacts on thin structures like lanes.
# 2. We use F.interpolate with size=skip.shape[-2:] (rather than scale_factor=2) so odd-sized
#    inputs still align perfectly with their skip features instead of being off by one.
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.1):
        super().__init__()
        #After the optional concat, the conv input has in_channels + skip_channels channels.
        #When there's no skip (deepest two upsamples), skip_channels is 0.

        #Here we calculate the total number of input channels to the conv block after concatenation.
        total_in = in_channels + skip_channels

        #Here we define the conv block that will process the upsampled features (and concatenated skip if available).
        #Dropout2d at the end zeroes whole feature channels (not individual pixels), which is the right form
        #of dropout for convolutional features. p=0 effectively disables it.
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x, skip=None):
        if skip is not None:
            #Here we upsample x to the spatial size of the skip feature using bilinear interpolation. This ensures 
            #that the upsampled features align perfectly with the skip features, and then we concatenate them.
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            #Here we just upsample x by a factor of 2 using bilinear interpolation since there's no skip to align with.
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #Apply the conv block to fuse the features (and reduce the channel count to out_channels).
        return self.conv(x)


#Here is the U-Net decoder using ResNet34Encoder. The encoder gives us features at strides 4, 8, 16, 32.
#We start from the deepest (s32) and walk back up, fusing with each shallower skip in turn, then
#do two more upsamples without skips to reach full input resolution. A final 1x1 conv projects
#to the desired number of output channels (2 for binary, embedding_dim for the instance head).
class ResNet34Decoder(nn.Module):
    """
    U-Net decoder. Consumes the [s4, s8, s16, s32] skip list produced by ResNet34Encoder
    and outputs a dense map at full input resolution with `out_channels` channels.
    """

    def __init__(self, out_channels, encoder_channels=ResNet34Encoder.OUT_CHANNELS, dropout_p=0.1):
        super().__init__()
        c4, c8, c16, c32 = encoder_channels   #(64, 128, 256, 512) for ResNet34

        #Channel widths going up: 512 -> 256 -> 128 -> 64 -> 32 -> 16. Modest decoder so the
        #encoder remains the part with most of the model's capacity. dropout_p is applied at
        #the end of every block; set to 0.0 to disable.
        self.up1 = _DecoderBlock(c32, c16, 256, dropout_p=dropout_p)  # s32 -> s16, fuse with s16
        self.up2 = _DecoderBlock(256, c8,  128, dropout_p=dropout_p)  # s16 -> s8,  fuse with s8
        self.up3 = _DecoderBlock(128, c4,   64, dropout_p=dropout_p)  # s8  -> s4,  fuse with s4
        self.up4 = _DecoderBlock( 64,  0,   32, dropout_p=dropout_p)  # s4  -> s2,  no skip available
        self.up5 = _DecoderBlock( 32,  0,   16, dropout_p=dropout_p)  # s2  -> s1,  no skip available

        #1x1 conv just rescales the channel count. No activation, so we return raw logits for
        #the binary head and raw embeddings for the embedding head, matching what the loss expects.
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, skips, output_size):
        s4, s8, s16, s32 = skips

        x = self.up1(s32, s16)
        x = self.up2(x,   s8)
        x = self.up3(x,   s4)
        x = self.up4(x)
        x = self.up5(x)

        #Safety: if the input dimensions weren't a clean multiple of 32, the cumulative bilinear
        #upsamples might land one pixel short. Force-resize to the exact input shape.
        if x.shape[-2:] != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)

        return self.final_conv(x)


#Drop-in replacement for LaneNet that uses the ResNet34 encoder + U-Net decoders. The forward
#contract is identical: returns (binary_logits, embedding) so loss.py and postprocess.py don't
#need to change. The encoder runs once per forward and feeds both decoders, which saves a chunk
#of compute compared to running ResNet34 twice.
class LaneNetResNet34(nn.Module):
    """
    LaneNet variant with an ImageNet-pretrained ResNet34 encoder and two parallel U-Net
    decoders (binary segmentation + instance embedding).

    Keyword arguments:
    - embedding_dim (int): channels in the instance embedding output (default 4).
    - pretrained (bool): load ImageNet weights into the encoder (default True).
    """

    def __init__(self, embedding_dim=4, pretrained=True, decoder_dropout=0.1):
        super().__init__()

        self.encoder = ResNet34Encoder(pretrained=pretrained)
        enc_channels = ResNet34Encoder.OUT_CHANNELS

        #Same dropout rate on both heads. If the embedding head suffers (disc loss climbs unexpectedly),
        #drop this to 0.0 just for the embedding decoder by passing decoder_dropout=0 here and passing
        #a separate value below. For now, keep them symmetric.
        self.binary_decoder    = ResNet34Decoder(out_channels=2,             encoder_channels=enc_channels, dropout_p=decoder_dropout)
        self.embedding_decoder = ResNet34Decoder(out_channels=embedding_dim, encoder_channels=enc_channels, dropout_p=decoder_dropout)

    def forward(self, x):
        #Remember the input H, W so the decoder can guarantee the output matches exactly.
        input_size = x.shape[-2:]

        skips = self.encoder(x)                                  # [s4, s8, s16, s32]
        binary_logits = self.binary_decoder(skips,    output_size=input_size)
        embedding     = self.embedding_decoder(skips, output_size=input_size)

        return binary_logits, embedding