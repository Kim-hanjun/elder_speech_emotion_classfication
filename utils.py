import random
from os import PathLike

import numpy as np
import torch
from datasets import Dataset


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_dataset(csv_path: PathLike) -> Dataset:
    pass


def preprocess_dataset(raw, tokenizer):
    pass


def compute_metric(pred):
    pass
