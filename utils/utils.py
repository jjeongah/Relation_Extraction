import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import confusion_matrix

import data_loader.data_loaders as datamodule_arch
import model.model as module_arch
import wandb


def new_instance(config):
    dataloader = getattr(datamodule_arch, config.dataloader.architecture)(config)
    model = getattr(module_arch, config.model.architecture)(config, dataloader.new_vocab_size)

    return dataloader, model


def load_model(args, config, dataloader, model):
    """
    불러온 모델이 저장되어 있는 디렉터리를 parsing함
    ex) 'save_models/klue/roberta-small_maxEpoch1_batchSize32_blooming-wind-57'
    """
    save_path = "/".join(args.saved_model.split("/")[:-1])

    """
    huggingface에 저장된 모델명을 parsing함
    ex) 'klue/roberta-small'
    """
    model_name = "/".join(args.saved_model.split("/")[1:-1]).split("_")[0]

    if args.saved_model.split(".")[-1] == "ckpt":
        model = model.load_from_checkpoint(args.saved_model)
    elif args.saved_model.split(".")[-1] == "pt" and args.mode != "continue train" and args.mode != "ct":
        model = torch.load(args.saved_model)
    else:
        exit("saved_model 파일 오류")

    config.path.save_path = save_path + "/"
    config.model.model_name = "/".join(model_name.split("/")[1:])
    return model, args, config


def new_instance(config):

    dataloader = getattr(datamodule_arch, config.dataloader.architecture)(config)

    model = getattr(module_arch, config.model.architecture)(config, dataloader.new_vocab_size)

    return dataloader, model


def load_model(args, config, dataloader, model):
    """
    불러온 모델이 저장되어 있는 디렉터리를 parsing함
    ex) 'save_models/klue/roberta-small_maxEpoch1_batchSize32_blooming-wind-57'
    """
    save_path = "/".join(args.saved_model.split("/")[:-1])

    """
    huggingface에 저장된 모델명을 parsing함
    ex) 'klue/roberta-small'
    """
    model_name = "/".join(args.saved_model.split("/")[1:-1]).split("_")[0]

    if args.saved_model.split(".")[-1] == "ckpt":
        model = model.load_from_checkpoint(args.saved_model)
    elif args.saved_model.split(".")[-1] == "pt" and args.mode != "continue train" and args.mode != "ct":
        model = torch.load(args.saved_model)
    else:
        exit("saved_model 파일 오류")

    config.path.save_path = save_path + "/"
    config.model.model_name = "/".join(model_name.split("/")[1:])
    return model, args, config


def text_preprocessing(sentence):
    # s = re.sub(r"!!+", "!!!", sentence)  # !한개 이상 -> !!! 고정
    # s = re.sub(r"\?\?+", "???", s)  # ?한개 이상 -> ??? 고정
    # s = re.sub(r"\.\.+", "...", s)  # .두개 이상 -> ... 고정
    # s = re.sub(r"\~+", "~", s)  # ~한개 이상 -> ~ 고정
    # s = re.sub(r"\;+", ";", s)  # ;한개 이상 -> ; 고정
    # s = re.sub(r"ㅎㅎ+", "ㅎㅎㅎ", s)  # ㅎ두개 이상 -> ㅎㅎㅎ 고정
    # s = re.sub(r"ㅋㅋ+", "ㅋㅋㅋ", s)  # ㅋ두개 이상 -> ㅋㅋㅋ 고정
    # s = re.sub(r"ㄷㄷ+", "ㄷㄷㄷ", s)  # ㄷ두개 이상 -> ㄷㄷㄷ 고정
    return sentence


def label_to_num(label):
    with open("./data/dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    num_label = [dict_label_to_num[v] for v in label]
    return num_label


def num_to_label(label):
    with open("./data/dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    origin_label = [dict_num_to_label[v] for v in label]
    return origin_label


def get_confusion_matrix(pred, label_ids, mode=None):
    cm = confusion_matrix(label_ids, np.argmax(pred, axis=-1))

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)

    cm_plot = sns.heatmap(cm, cmap="Blues", fmt="d", annot=True, ax=ax)
    cm_plot.set_xlabel("pred")
    cm_plot.set_ylabel("true")
    cm_plot.set_title(f"{mode} confusion matrix")

    wandb.log({f"{mode} confusion_matrix": wandb.Image(fig)})


def monitor_config(key, on_step):
    """Returns proper metric monitor-mode pair."""
    mapping = {
        "val_loss": {"monitor": "val_loss", "mode": "min"},
        "val_pearson": {"monitor": "val_pearson", "mode": "max"},
        "val_f1": {"monitor": "val_f1", "mode": "max"},
    }
    new_mapping = mapping.copy()
    if on_step is True:
        for m in mapping:
            for detail in ["step", "epoch"]:
                new_mapping[f"{m}_{detail}"] = mapping[m]
    else:
        if key.endswith("step"):
            raise ValueError(f"Cannot monitor {key} when on_step is set 'False'")

    return new_mapping[key]


