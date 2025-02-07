"""
"Residual network" modulation classification network implementation

References
----------
[1] Scholl, S., “Classification of Radio Signals and HF Transmission Modes
with Deep Learning”, <i>arXiv e-prints</i>, Art. no. arXiv:1906.04459,
2019. doi:10.48550/arXiv.1906.04459.
"""
import numpy as np
import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from .net_utils import init_weights


class ResidualBlock(nn.Module):
    """
    Residual network building block class
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        """
        Residual block class object constructor

        Parameters
        ----------
        self: ResidualBlock
            ResidualBlock class object reference

        kwargs: dict
            Stores optional parameters

        Returns
        -------
        self: ResidualBlock
            ResidualBlock class object reference
        """
        super().__init__()

        """
        conv, 1, N, linear
        """
        self.conv1 =\
            nn.Sequential(
                nn.Conv1d(in_channels,
                          out_channels,
                          kernel_size=1),
                nn.BatchNorm1d(out_channels))

        """
        conv, 3, N, ReLU
        conv, 3, N, linear
        """
        self.conv2 =\
            nn.Sequential(
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels))

        """
        conv, 3, N, ReLU
        conv, 3, N, linear
        """
        self.conv3 =\
            nn.Sequential(
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels))

        """
        Shortcut layers
        """
        self.shortcut1 =\
            nn.Sequential(
                    nn.Conv1d(out_channels,
                              out_channels,
                              kernel_size=1,
                              bias=False),
                    nn.BatchNorm1d(out_channels))

        self.shortcut2 =\
            nn.Sequential(
                    nn.Conv1d(out_channels,
                              out_channels,
                              kernel_size=1,
                              bias=False),
                    nn.BatchNorm1d(out_channels))

        self.max_pool =\
            nn.MaxPool1d(kernel_size=3,
                         stride=2,
                         padding=1)

        self.apply(init_weights)

    def forward(self,
                x):
        """
        Implements the foward pass of a residual network building block

        Parameters
        ----------
        self: ResidualBlock
            ResidualBlock class object reference

        x: torch.Tensor
            Input tensor

        Returns
        -------
        y: torch.Tensor
            Output tensor
        """
        layer1_out = self.conv1(x.clone())
        layer2_out = self.conv2(layer1_out)
        sc1_out = self.shortcut1(layer1_out)
        layer3_in = layer2_out + sc1_out

        layer3_out = self.conv2(layer3_in)
        sc2_out = self.shortcut2(layer3_in)

        return self.max_pool(layer3_out + sc2_out)


class ResidualNetwork(pl.LightningModule):
    """
    Automatic Modulation Classification (AMC) Convolutional Neural
    Network [1]
    """

    def __init__(self,
                 **kwargs):
        """
        ResidualNetwork object constructor

        Parameters
        ----------
        self: ResidualNetwork
            ResidualNetwork class object reference

        kwargs: dict
            Stores optional parameters

        Returns
        -------
        self: ResidualNetwork
            ResidualNetwork class object reference
        """
        super(ResidualNetwork,
              self).__init__()

        self.num_classes = 18
        self.optim_params = dict()

        self.optim_params["lr"] =\
            kwargs.get("lr", 1E-3)

        self.loss = nn.CrossEntropyLoss()

        self.train_accuracy =\
            tm.Accuracy(task="multiclass",
                        num_classes=self.num_classes)

        self.val_accuracy =\
            tm.Accuracy(task="multiclass",
                        num_classes=self.num_classes)

        self.test_accuracy =\
            tm.Accuracy(task="multiclass",
                        num_classes=self.num_classes)

        self.res_stack1 = ResidualBlock(2, 96)
        self.res_stack2 = ResidualBlock(96, 96)
        self.drop1 = nn.Dropout1d(0.5, inplace=True)
        self.res_stack3 = ResidualBlock(96, 96)
        self.res_stack4 = ResidualBlock(96, 96)
        self.res_stack5 = ResidualBlock(96, 128)
        self.drop2 = nn.Dropout1d(0.5, inplace=True)
        self.res_stack6 = ResidualBlock(128, 128)
        self.res_stack7 = ResidualBlock(128, 128)
        self.res_stack8 = ResidualBlock(128, 128)
        self.drop3 = nn.Dropout1d(0.7, inplace=True)

        self.linear_layer =\
            nn.Sequential(nn.Linear(1024, self.num_classes),
                          nn.BatchNorm1d(self.num_classes))

        self.softmax = nn.Softmax(dim=1)

    def forward(self,
                x,
                applySoftmax=True):
        """
        Implements the forward pass of a residual network

        Parameters
        ----------
        self: ResidualNetwork
            ResidualNetwork class object reference

        x: torch.Tensor
            [Batch size] x 2048 sample (I/Q) sample vector

        applySoftmax: bool (Optional)
            Boolean that controls whether to apply softmax to linear
            layers output. This input should be set to false during
            training to be compatible with CrossEntropy loss

        Returns
        -------
        [batch size x number of classes] tensor that stores the
            predicted classes for each network input
        """
        net_output = self.res_stack1(x.clone())
        net_output = self.res_stack2(net_output)
        net_output = self.drop1(net_output)
        net_output = self.res_stack3(net_output)
        net_output = self.res_stack4(net_output)
        net_output = self.res_stack5(net_output)
        net_output = self.drop2(net_output)
        net_output = self.res_stack6(net_output)
        net_output = self.res_stack7(net_output)
        net_output = self.res_stack8(net_output)
        net_output = self.drop3(net_output)
        net_output = net_output.reshape(-1, 128 * 8)

        net_output = self.linear_layer(net_output)

        if applySoftmax:
            net_output = self.softmax(net_output)

        return net_output

    def configure_optimizers(self):
        """
        Configures optimizer function(s)

        Parameters
        ----------
        self: ResidualNetwork
            ResidualNetwork CNN class object reference

        Returns
        ----------
        self: ResidualNetwork
            ResidualNetwork CNN class object reference
        """
        optimizer = optim.Adam(params=self.parameters(),
                               lr=self.optim_params.get("lr", 1E-3))

        return optimizer

    def training_step(self,
                      batch,
                      batch_idx):
        """
        Implements a Pytorch Lightning module training step

        Parameters
        ----------
        self: ResidualNetwork
            ResidualNetwork class object reference

        batch: tuple
            Tuple that stores batch data, ordinal encoded modulation mode &
            ordinal encoded Signal-to-Noise Ratio (SNR)

        batch_idx: integer
            Batch index

        Returns
        -------
        train_loss: float
            Batch training loss
        """
        batch_data, modeordenc, _ = batch

        logits = self.forward(batch_data,
                              applySoftmax=False)

        target = self.init_target(modeordenc)

        train_loss = self.loss(logits, target)

        self.train_accuracy(self.softmax(logits),
                            modeordenc)

        self.log("train_loss",
                 train_loss,
                 prog_bar=True)

        self.log("train_accuracy",
                 self.train_accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return train_loss

    def validation_step(self,
                        batch,
                        batch_idx):
        """
        Implements a Pytorch Lightning module validation step

        Parameters
        ----------
        self: ResidualNetwork
            ResidualNetwork class object reference

        batch: tuple
            Tuple that stores batch data, ordinal encoded modulation mode &
            ordinal encoded Signal-to-Noise Ratio (SNR)

        batch_idx: integer
            Batch index

        Returns
        -------
        None
        """
        batch_data, modeordenc, _ = batch

        logits = self.forward(batch_data,
                              applySoftmax=False)

        target = self.init_target(modeordenc)

        self.val_accuracy(self.softmax(logits),
                          modeordenc)

        self.log("validation_loss",
                 self.loss(logits, target),
                 prog_bar=True)

        self.log("val_acc",
                 self.val_accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def predict_step(self,
                     batch,
                     batch_idx):
        """
        Implements a Pytorch Lightning module prediction step

        Parameters
        ----------
        self: ResidualNetwork
            ResidualNetwork class object reference

        batch: tuple
            Tuple that stores batch data, ordinal encoded modulation mode &
            ordinal encoded Signal-to-Noise Ratio (SNR)

        batch_idx: integer
            Batch index

        Returns
        -------
        predictions : Tensor
            [Batch size x number of classes] tensor that stores batch
            class predicitions

        modeordenc : Tensor
            [Batch size x 1] tensor that stores the ordinal encoded
            modulation mode

        snrordenc : Tensor
            [Batch size x 1] tensor that stores the ordinal encoded
            Signal-to-Noise Ratio (SNR)
        """
        batch_data, modeordenc, snrordenc = batch

        predictions = self.forward(batch_data)

        return predictions, modeordenc, snrordenc

    def test_step(self,
                  batch,
                  batch_idx):
        """
        Implements a Pytorch Lightning module test step

        Parameters
        ----------
        self: Deep CNN
            Deep CNN class object reference

        batch: tuple
            Tuple that stores batch data, ordinal encoded modulation mode &
            ordinal encoded Signal-to-Noise Ratio (SNR)

        batch_idx: integer
            Batch index

        Returns
        -------
        None
        """
        batch_data, modeordenc, _ = batch

        logits = self.forward(batch_data,
                              applySoftmax=False)

        target = self.init_target(modeordenc)

        self.test_accuracy(self.softmax(logits),
                           modeordenc)

        self.log("test_loss",
                 self.loss(logits, target),
                 prog_bar=True)

        self.log("test_acc",
                 self.test_accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def init_target(self,
                    modeordenc):
        """
        Initializes a tensor that stores target predicted class confidence
        (for the cross entropy loss) given a batch's ordinal encoded
        modulation mode

        Parameters:
        ----------
        self: Deep CNN
            Deep CNN class object reference

        modeordenc : Tensor
            [Batch size x 1] tensor that stores the ordinal encoded
            modulation mode

        Returns
        -------
        Tensor that stores target predicted class confidence (for the cross
        entropy loss) given a batch's ordinal encoded modulation mode
        """
        shape2d = [len(modeordenc), self.num_classes]

        linear_idx =\
            np.ravel_multi_index([np.arange(len(modeordenc)),
                                  np.array(modeordenc.cpu())],
                                 shape2d)

        target = np.zeros(shape2d)
        target.ravel()[linear_idx] = 15

        return torch.Tensor(target).softmax(axis=1).to(self.device)
