import argparse
import random

from typing import Callable

from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

from models.t4rec_base import create_model as create_model_base, create_schema as create_schema_base
from models.t4rec_enriched import create_model as create_model_enriched, create_schema as create_schema_enriched


def get_args_parser():
    parser = argparse.ArgumentParser("RecSys pre-process", add_help=False)
    parser.add_argument(
        "--split",
        default="small",
        type=str,
        metavar="taskssplit",
        help="select small or large ",
    )
    parser.add_argument(
        "--history_size",
        default=20,
        type=int,
        metavar="hist",
        help="select history size",
    )

    parser.add_argument(
        "--epochs", default=20, type=int, metavar="epochs", help="select epochs number"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        default=128,
        type=int,
        metavar="train_batch",
        help="train batch size",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        default=32,
        type=int,
        metavar="eval_batch",
        help="eval batch size",
    )

    parser.add_argument(
        "--dataset_type",
        default="base",
        type=str,
        choices=["base", "enriched"],
        metavar="dataset_type",
        help="Type of dataset, base or enriched."
    )

    return parser


def main_train(
    split: str,
    history_size: int,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    dataset_type: str
):

    create_model: Callable = create_model_base if dataset_type == "base" else create_model_enriched
    create_schema: Callable = create_schema_base if dataset_type == "base" else create_schema_enriched

    # Initialize an empty schema
    model = create_model(max_sequence_length=history_size, d_model=64, load=False)

    output_dir = f"../checkpoints/{dataset_type}/{split}"

    train_args = T4RecTrainingArguments(
        data_loader_engine="merlin",
        dataloader_drop_last=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        output_dir=output_dir,
        learning_rate=0.0005,
        lr_scheduler_type="cosine",
        learning_rate_num_cosine_cycles_by_epoch=1.5,
        num_train_epochs=epochs,
        max_sequence_length=history_size,
        report_to=[],
        logging_strategy="epoch",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,  # keep only the last and best
        no_cuda=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        schema=create_schema(),
    )

    data_path: str = "final" if dataset_type == "base" else "advanced_ts"

    train_paths = [
        f"../data/{data_path}_ebnerd_processed/train/train_data_{split}_0.parquet",
        f"../data/{data_path}_ebnerd_processed/train/train_data_{split}_1.parquet",
        f"../data/{data_path}_ebnerd_processed/train/train_data_{split}_2.parquet",
        f"../data/{data_path}_ebnerd_processed/train/train_data_{split}_3.parquet",
        f"../data/{data_path}_ebnerd_processed/train/train_data_{split}_4.parquet",
        f"../data/{data_path}_ebnerd_processed/train/train_data_{split}_5.parquet",
        f"../data/{data_path}_ebnerd_processed/train/train_data_{split}_6.parquet",
    ]

    # pick a random set from the validation lists
    eval_paths = random.sample(
        [
            f"../data/{data_path}_ebnerd_processed/validation/validation_data_{split}_0.parquet",
            f"../data/{data_path}_ebnerd_processed/validation/validation_data_{split}_1.parquet",
            f"../data/{data_path}_ebnerd_processed/validation/validation_data_{split}_2.parquet",
            f"../data/{data_path}_ebnerd_processed/validation/validation_data_{split}_3.parquet",
            f"../data/{data_path}_ebnerd_processed/validation/validation_data_{split}_4.parquet",
            f"../data/{data_path}_ebnerd_processed/validation/validation_data_{split}_5.parquet",
            f"../data/{data_path}_ebnerd_processed/validation/validation_data_{split}_6.parquet",
        ],
        k=2,
    )

    trainer.train_dataset_or_path = train_paths
    trainer.eval_dataset_or_path = eval_paths
    trainer.reset_lr_scheduler()
    trainer.train()
    trainer.state.global_step += 1
    
    print("Finished training")



if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    print(f"Split: {args.split}")
    print(f"History size: {args.history_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Train batch size: {args.per_device_train_batch_size}")
    print(f"Evaluate batch size: {args.per_device_eval_batch_size}")

    main_train(
        split=args.split,
        history_size=args.history_size,
        epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataset_type=args.dataset_type
    )
