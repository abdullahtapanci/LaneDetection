import torch
import torch.nn as nn

from src.models.encoder import ENetEncoder, RegularBottleneck, UpsamplingBottleneck


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
        features, idx1, idx2, s1, s2, inp = self.encoder(x)

        binary_output = self.binary_decoder(
            features, idx1, idx2, s1, s2, inp.shape[2:])
        embedding_output = self.embedding_decoder(
            features, idx1, idx2, s1, s2, inp.shape[2:])

        return binary_output, embedding_output