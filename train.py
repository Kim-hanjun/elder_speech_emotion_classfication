import json
import logging
import os
import time
from functools import partial

import torch
from setproctitle import setproctitle
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
)
from transformers.trainer_utils import is_main_process

from arguments import DatasetsArguments, ModelArguments, MyTrainingArguments
from literal import LABEL2ID
from utils import (
    compute_metrics,
    get_dataset,
    preprocess_dataset,
    preprocess_logits_for_metrics,
    seed_everything,
)

logger = logging.getLogger(__name__)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(model_args: ModelArguments, dataset_args: DatasetsArguments, training_args: MyTrainingArguments):
    setproctitle("elder_speach_cls")
    seed_everything(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=len(LABEL2ID))
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    train_dataset = get_dataset(dataset_args.train_csv_path) if dataset_args.train_csv_path else None
    test_dataset = get_dataset(dataset_args.test_csv_path) if dataset_args.test_csv_path else None

    return_token_type_ids = False if "roberta" in model_args.model_name_or_path.lower() else True
    preprocess_func = partial(preprocess_dataset, tokenizer=tokenizer, return_token_type_ids=return_token_type_ids)
    if train_dataset is not None:
        train_dataset = train_dataset.map(preprocess_func, remove_columns=train_dataset.column_names)
    if test_dataset is not None:
        test_dataset = test_dataset.map(preprocess_func, remove_columns=test_dataset.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    if training_args.local_rank > 0:
        torch.distributed.barrier()
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        torch.distributed.barrier()

    if training_args.do_eval:
        metric = trainer.evaluate(test_dataset)
        json_metric = {model_args.model_name_or_path: metric}
        if training_args.local_rank == 0:
            with open(os.path.join("log", f"{time.strftime('%Y%m%d-%H:%M:%S')}_result.json"), "w") as file:
                json.dump(json_metric, file, indent=4)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    main(model_args=model_args, dataset_args=dataset_args, training_args=training_args)
