"""
Modulation classification networks implementation

References
----------
[1] Scholl, S., “Classification of Radio Signals and HF Transmission Modes
with Deep Learning”, <i>arXiv e-prints</i>, Art. no. arXiv:1906.04459,
2019. doi:10.48550/arXiv.1906.04459.
"""
import re
import numpy as np
import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from .net_utils import init_weights


class AllConvNet(pl.LightningModule):
    """
    All Convolution Neutral Netwok for Automatic Modulation Classification
    (AMC) Convolutional Neural Network [1]
    """
    def __init__(self,
                 **kwargs):
        """
        AllConvNet class constructor

        Parameters
        ----------
        self: AllConvNet class reference

        **kwargs : dict
            Dictionary that stores optional parameters

            lr : float
                Learning rate (default value of 1E-3)

        Returns
        ----------
        self: AllConvNet class reference
        """
        super(AllConvNet,
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

        self.conv1 =\
            nn.Sequential(
                nn.Conv1d(in_channels=2,
                          out_channels=64,
                          kernel_size=3,
                          stride=2,
                          padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(in_channels=64,
                          out_channels=64,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(in_channels=64,
                          out_channels=64,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU())

        self.conv2 =\
            nn.Sequential(
                nn.Conv1d(in_channels=64,
                          out_channels=128,
                          kernel_size=3,
                          stride=2,
                          padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(in_channels=128,
                          out_channels=128,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(in_channels=128,
                          out_channels=128,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU())

        self.drop1 = nn.Dropout1d(0.5,
                                  inplace=True)

        self.conv3 =\
            nn.Sequential(
                nn.Conv1d(in_channels=128,
                          out_channels=256,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU())

        self.drop2 = nn.Dropout1d(0.8,
                                  inplace=True)

        self.conv4 =\
            nn.Sequential(
                nn.Conv1d(in_channels=256,
                          out_channels=320,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Conv1d(in_channels=320,
                          out_channels=320,
                          kernel_size=3,
                          stride=2,
                          padding=2),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Conv1d(in_channels=320,
                          out_channels=320,
                          kernel_size=1,
                          padding=0),
                nn.BatchNorm1d(320),
                nn.ReLU())

        self.conv5 =\
            nn.Sequential(
                nn.Conv1d(in_channels=320,
                          out_channels=18,
                          kernel_size=1,
                          padding=0),
                nn.BatchNorm1d(self.num_classes))

        self.gap = nn.AdaptiveAvgPool1d(self.num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.apply(init_weights)

    def forward(self,
                x,
                applySoftmax=True):
        """
        Implements the forward pass of a "All Conv Net" network

        Parameters
        ----------
        self: AllConvNet
            AllConvNet class object reference

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
        layer_out = x.clone()

        po = re.compile(r"^conv[0-9]+$")

        for layerid in [elem for elem in self.__dict__["_modules"].keys()
                        if po.match(elem) is not None]:

            layer_out = self.__dict__["_modules"][layerid](layer_out)

        net_output = self.gap(layer_out.reshape(-1,
                                                self.num_classes ** 2))

        if applySoftmax:
            net_output = self.softmax(net_output)

        return net_output

    def configure_optimizers(self):
        """
        Configures optimizer function(s)

        Parameters
        ----------
        self: AllConvNet
            AllConvNet class object reference

        Returns
        ----------
        self: AllConvNet
            AllConvNet class object reference
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
        self: AllConvNet
            AllConvNet class object reference

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
        self: AllConvNet
            AllConvNet class object reference

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
        self: AllConvNet
            AllConvNet class object reference

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
        self: AllConvNet
            AllConvNet class object reference

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
        self: AllConvNet
            AllConvNet class object reference

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
