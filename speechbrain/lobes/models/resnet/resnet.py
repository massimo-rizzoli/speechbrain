"""
Implementation of a ResNet model for one dimensional signals.

Authors
 * Massimo Rizzoli 2022
"""


import torch
import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.pooling import Pooling1d, AdaptivePool
from speechbrain.nnet.activations import Softmax

class ResNetBlock1d(torch.nn.Module):
    """Building block for the ResNet model. It considers one-dimensional signals.

    Arguments
    ---------
    in_channels: int
        Number of channels of the input features.
    out_channels: int
        Number of output channels.
    downsample: bool
        Whether to perform downsampling (on the time dimension) or not.
    kernel_size: int
        Kernel size for the Conv1d layers.
    activation: torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> input_feats = torch.rand([32, 64, 256])
    >>> resnet_block1d = ResNetBlock1d(in_channels=256, out_channels=512, downsample=True)
    >>> output = resnet_block1d(input_feats)
    >>> output.shape
    torch.Size([32, 32, 512])
    >>> input_feats = torch.rand([32, 64, 256])
    >>> resnet_block1d = ResNetBlock1d(in_channels=256, out_channels=512, downsample=False)
    >>> output = resnet_block1d(input_feats)
    >>> output.shape
    torch.Size([32, 64, 512])
    >>> input_feats = torch.rand([32, 64, 256])
    >>> resnet_block1d = ResNetBlock1d(in_channels=256, out_channels=256, downsample=False)
    >>> output = resnet_block1d(input_feats)
    >>> output.shape
    torch.Size([32, 64, 256])
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample,
        kernel_size=3,
        activation=torch.nn.LeakyReLU
    ):
        super(ResNetBlock1d, self).__init__()

        if downsample:
            conv1 = Conv1d(out_channels, kernel_size, in_channels=in_channels, stride=2, padding="same")
            # when downsampling it is necessary to also downsample through the skip connection
            self.shortcut = Sequential(
                Conv1d(out_channels, 1, in_channels=in_channels, stride=2, padding="same"),
                BatchNorm1d(input_size=out_channels),
            )
        else:
            conv1 = Conv1d(out_channels, kernel_size, in_channels=in_channels, stride=1, padding="same")
            # empty Sequential: corresponds to identity function
            self.shortcut = Sequential(input_shape=[None])

        conv2 = Conv1d(out_channels, kernel_size, in_channels=out_channels, stride=1, padding="same")

        conv_block = lambda conv, out_channels : Sequential(
            conv,
            BatchNorm1d(input_size=out_channels),
            activation()
        )
        self.conv_block1 = conv_block(conv1, out_channels)
        self.conv_block2 = conv_block(conv2, out_channels)

        self.activation = activation()


    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.activation(x + shortcut)
        return x

class ResNet(torch.nn.Module):
    """This model extracts ResNet features.

    Arguments
    ---------
    in_channels: int
        Number of channels of the input features.
    input_layer_out_channels: int
        Starting number of channels for ResNet blocks (will be doubled after each block).
    input_layer_kernel_size: int
        Kernel size for the ResNet input layer.
    n_blocks: int
        Number of ResNet blocks. Each block conists of two ResNetBlock1d: the first one downsampling on the time dimension (by half) and doubling the number of channels, the second keeping the same shape.
    kernel_size: int
        Kernel size for the ResNet blocks.
    activation: torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> input_feats = torch.rand([32, 1024, 40])
    >>> resnet = ResNet(in_channels=40, n_blocks=6 input_layer_out_channels=64)
    >>> output = resnet(input_feats)
    >>> output.shape
    torch.Size([32, 8, 2048])
    """
    def __init__(
        self,
        in_channels,
        input_layer_out_channels=64,
        input_layer_kernel_size=7,
        n_blocks=5,
        kernel_size=3,
        activation=torch.nn.LeakyReLU
    ):
        super(ResNet, self).__init__()
        self.input_layer = Sequential(
            Conv1d(input_layer_out_channels, input_layer_kernel_size, in_channels=in_channels, stride=2, padding="same"),
            Pooling1d("max", 3, input_dims=3, pool_axis=1, stride=2, padding=1),
            BatchNorm1d(input_size=input_layer_out_channels),
            activation()
        )

        resblock = lambda out_channels, downsample : ResNetBlock1d(
            in_channels=out_channels//(2 if downsample else 1),
            out_channels=out_channels,
            downsample=downsample,
            kernel_size=kernel_size,
            activation=activation
        )

        self.res_blocks = Sequential(input_shape=[input_layer_out_channels])

        out_channels = input_layer_out_channels
        self.res_blocks.append(Sequential(
            resblock(out_channels=input_layer_out_channels, downsample=False),
            resblock(out_channels=input_layer_out_channels, downsample=False)
        ))

        # the first block is the first Sequential appended to self.layers
        out_channels = out_channels*2
        for _ in range(n_blocks-1):
            self.res_blocks.append(Sequential(
                resblock(out_channels=out_channels, downsample=True),
                resblock(out_channels=out_channels, downsample=False)
            ))
            out_channels = out_channels*2

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return x

class Classifier(Sequential):
    """This class implements the last MLP on the top of ResNet features.
    Arguments
    ---------
    input_size: int
        Expected size of the last dimension of the input.
    outputs: int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([32, 1024, 40])
    >>> resnet = ResNet(in_channels=40, n_blocks=6 input_layer_out_channels=64)
    >>> resnet_out = resnet(input_feats)
    >>> classif = Classifier(input_size=resnet_out.shape[-1], outputs=10)
    >>> output = classif(resnet_out)
    >>> output.shape
    torch.Size([32, 10])
    """
    def __init__(
        self,
        input_size=2048, # ResNet.input_layer_out_channels * 2^(ResNet.n_blocks - 1)
        outputs=10, # number of classes
    ):
        super(Classifier, self).__init__(input_shape=[None, None, input_size])
        self.append(AdaptivePool(1))
        self.append(Linear(outputs, input_size=input_size))
        self.append(Softmax(apply_log=True))
