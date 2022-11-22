import random
from os import PathLike

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from literal import LABELS, RAW_DATA, RAW_LABELS, SENT


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_dataset(csv_path: PathLike) -> Dataset:
    df = pd.read_csv(csv_path)
    sents = df[RAW_DATA].to_list()
    labels = df[RAW_LABELS].to_list()

    return Dataset.from_dict({SENT: sents, LABELS: labels})


def preprocess_dataset(raw, tokenizer):
    pass


def compute_metric(pred):
    references = pred.label_ids
    predictions = np.argmax(pred.predictions, axis=-1)
    accuracy = evaluate.load("accuracy").compute
    f1 = evaluate.load("f1").compute
    recall = evaluate.load("recall").compute
    precision = evaluate.load("precision").compute
    roc_auc = evaluate.load("roc_auc", "multiclass").compute

    metric = {
        "accuracy": accuracy(references=references, predictions=predictions),
        "f1": f1(references=references, predictions=predictions),
        "recall": recall(references=references, predictions=predictions),
        "precision": precision(references=references, predictions=predictions),
        "roc_auc": roc_auc(references=references, prediction_scores=pred.predictions, multi_class="ovr"),
    }

    return metric
