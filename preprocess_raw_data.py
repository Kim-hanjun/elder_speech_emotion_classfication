import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from literal import LABEL2ID, PREPROCESS_FOLDER, RAW_DATA, RAW_LABELS


def main(args):
    data_path = args.raw_data_path

    raw_data = pd.read_excel(data_path, header=1)

    train_df, test_df = train_test_split(
        raw_data, test_size=args.test_ratio, random_state=args.seed, stratify=raw_data[RAW_LABELS]
    )

    train_info = train_df[RAW_LABELS].value_counts()
    test_info = test_df[RAW_LABELS].value_counts()
    logging.info(f"train info\nlen:{len(train_df)}\n{train_info}")
    logging.info(f"test info\nlen:{len(test_df)}\n{test_info}")
    logging.info(f"LABEL2ID\n{LABEL2ID}")

    train_df: pd.DataFrame = train_df[[RAW_DATA, RAW_LABELS]].reset_index(drop=True)
    test_df: pd.DataFrame = test_df[[RAW_DATA, RAW_LABELS]].reset_index(drop=True)

    if not os.path.exists(PREPROCESS_FOLDER):
        os.mkdir(PREPROCESS_FOLDER)

    train_df.to_csv(f"{PREPROCESS_FOLDER}/train.csv", index=False)
    test_df.to_csv(f"{PREPROCESS_FOLDER}/test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw_data_path", type=str, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    args = parser.parse_args()

    if not os.path.exists("log"):
        os.mkdir("log")

    logging.basicConfig(
        filename="log/preprocess.log",
        level=logging.INFO,
    )

    main(args=args)
