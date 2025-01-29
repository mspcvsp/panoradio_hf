"""
Deep learning algorithm utility functions
"""
import os
from pathlib import Path
import joblib as jl
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import classification_report, confusion_matrix
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from .data import get_data_dir


def init_weights(m):
    """
    Linear layer initialiation function

    Parameters
    ----------
    m: torch.nn.modules object

    Returns
    -------
    None
    """
    if isinstance(m, nn.Linear):

        init.kaiming_normal_(m.weight,
                             mode='fan_in',
                             nonlinearity='relu')

        m.bias.data.fill_(0.01)
    # --------------------------------
    elif isinstance(m, nn.Conv1d):

        init.kaiming_normal_(m.weight,
                             mode='fan_in',
                             nonlinearity='relu')

        if m.bias is not None:
            m.bias.data.fill_(0.01)


def get_model_checkpoint_dir(net_arch_id,
                             **kwargs):
    """
    Returns the full path to the top-level model checkpoint directory

    Parameters
    ----------
    net_arch_id: str
        Neural network architecture identifier

    kwargs: dict
        Dictionary that stores optional parameters

        data_dir : str
            Top-level model checkpoint directory name

    Returns
    -------
    data_dir: str
        Full path to the top-level model checkpoint directory name
    """
    path_elems = ["/home",
                  os.getenv("USER"),
                  "panoradio_hf",
                  kwargs.get("model_checkpoint_dir", "ModelCheckpoints"),
                  net_arch_id]

    return Path(*path_elems)


def get_mlflow_tracking_uri():

    mlflow_tracking_uri =\
        Path(*["/home",
               os.getenv("USER"),
               "ml-runs"])

    return mlflow_tracking_uri


def initialize_pl_trainer(net_arch_id,
                          **kwargs):

    patience = kwargs.get("patience", 5)
    min_delta = kwargs.get("min_delta", 0.01)

    mlflow_tracking_uri = kwargs.get("mlflow_tracking_uri",
                                     get_mlflow_tracking_uri())

    checkpoint_callback = ModelCheckpoint(
        dirpath=get_model_checkpoint_dir(net_arch_id,
                                         **kwargs),
        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}')

    mlf_logger =\
        MLFlowLogger(experiment_name=net_arch_id,
                     tracking_uri=f"file:{mlflow_tracking_uri}")

    early_stop_callback =\
        EarlyStopping(monitor="val_acc",
                      min_delta=min_delta,
                      patience=patience,
                      verbose=False,
                      mode="max")

    callbacks = [checkpoint_callback,
                 early_stop_callback]

    return callbacks, mlf_logger


def get_ordenc_intid_lut(ordenc):

    values = ordenc.categories_[0]
    enc_values = ordenc.transform(values.reshape(-1, 1))

    return dict(zip(enc_values.flatten().astype(int), values))


class ModelPredictionFormatter(object):

    def __init__(self,
                 **kwargs):

        data_dir = get_data_dir(**kwargs)

        self.mode_ordenc =\
            jl.load(data_dir.joinpath("modeordenc.jl"))

        self.snr_ordenc =\
            jl.load(data_dir.joinpath("snrordenc.jl"))

        self.modeint_modeid_lut =\
            get_ordenc_intid_lut(self.mode_ordenc)

        self.snrint_snrid_lut =\
            get_ordenc_intid_lut(self.snr_ordenc)

    def __call__(self,
                 batch_preds):

        mode_conf_preds =\
            pd.DataFrame(np.array(batch_preds[0]))

        mode_conf_preds.rename(columns=self.modeint_modeid_lut,
                               inplace=True)

        predictedmodeid =\
            np.zeros(mode_conf_preds.shape[0],
                     dtype=object)

        for index, row in mode_conf_preds.iterrows():

            predictedmodeid[index] =\
                self.modeint_modeid_lut[row.argmax()]

        mode_ints =\
            np.array(batch_preds[1]).reshape(-1, 1)

        mode_conf_preds["truemodeid"] =\
            self.mode_ordenc.inverse_transform(mode_ints).flatten()

        mode_conf_preds["predictedmodeid"] = predictedmodeid

        snr_ints =\
            np.array(batch_preds[2]).reshape(-1, 1)

        mode_conf_preds["snrid"] =\
            self.snr_ordenc.inverse_transform(snr_ints).flatten()

        return mode_conf_preds


def evaluate_model_predictions(predictions):

    pred_fmt = ModelPredictionFormatter()

    mode_conf_preds = \
        pd.concat([pred_fmt(elem) for elem in predictions])

    snrid_clf_report = dict()
    snrid_conf_mat = dict()
    modeids = list(mode_conf_preds.columns[:18])

    for snrid in mode_conf_preds["snrid"].value_counts().keys():

        select_row = mode_conf_preds["snrid"] == snrid

        truemodeid =\
            mode_conf_preds.loc[select_row, "truemodeid"].values

        predictedmodeid =\
            mode_conf_preds.loc[select_row, "predictedmodeid"].values

        snrid_clf_report[snrid] =\
            classification_report(truemodeid,
                                  predictedmodeid,
                                  zero_division=0,
                                  output_dict=True)

        cur_conf_mat =\
            confusion_matrix(truemodeid,
                             predictedmodeid,
                             normalize="true")

        snrid_conf_mat[snrid] =\
            pd.DataFrame(cur_conf_mat,
                         index=modeids,
                         columns=modeids)

    return snrid_clf_report, snrid_conf_mat


def parse_snrid(snrid):
    snr = snrid.replace("snr", "")
    return float(snr.replace("minus", "-"))


def compute_conv1d_lout(l_in,
                        kernel_size,
                        **kwargs):

    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)

    l_out = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
    l_out = l_out / stride + 1
    return int(l_out)


def init_snrid_accuracy(snrid_clf_report):

    snrid_acc =\
        pd.Series({key: snrid_clf_report[key]["accuracy"]
                   for key in snrid_clf_report})

    snrid_acc = pd.DataFrame(snrid_acc)
    snrid_acc.reset_index(inplace=True)

    snrid_acc.rename(columns={0: "accuracy",
                              "index": "snrid"}, inplace=True)

    snrid_acc["snr"] =\
        snrid_acc["snrid"].apply(lambda elem: parse_snrid(elem))

    return snrid_acc
