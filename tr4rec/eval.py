import argparse
import torch

import polars as pl
import pandas as pd

from typing import List, Tuple, Callable
from tqdm import tqdm

from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from models.t4rec_base import load_model as load_model_base, create_trainer as create_trainer_base
from models.t4rec_enriched import load_model as load_model_enriched, create_trainer as create_trainer_enriched

def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("RecSys pre-process", add_help=False)

    parser.add_argument(
        "--split",
        default="small",
        type=str,
        metavar="taskssplit",
        help="select small or large or testset",
    )

    parser.add_argument(
        "--eval_batch_size",
        default=512,
        type=int,
        metavar="eval_batch_size",
        help="Inference batch size.",
    )

    parser.add_argument(
        "--data_category",
        default="train",
        type=str,
        metavar="cat",
        help="validation or test",
    )

    parser.add_argument(
        "--path",
        default=f"../checkpoints/enriched/small/checkpoint-339/pytorch_model.bin",
        type=str,
        metavar="path",
        help="The path of the model to load.",
    )

    parser.add_argument(
        "--dataset_type",
        default="base",
        type=str,
        metavar="dataset_type",
        choices=["base", "enriched"],
        help="Dataset type between base and enriched",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        metavar="path",
        help="The path to save the output. Use None if you dont want to save it.",
    )
    return parser


def get_inview_articles_score(
    ids_inview: List[int], prob_list: List[float]
) -> List[float]:

    inview_scores: List[float] = []

    for inview_id in ids_inview:
        inview_scores.append(prob_list[inview_id])

    return inview_scores


def inference(
    model_path: str,
    paths: str,
    per_device_eval_batch_size: int,
    max_sequence_length: int,
    dataset_type: str
) -> Tuple[List[List[float]], pd.DataFrame]:
    # load the model
    load_model: Callable = load_model_base if dataset_type == "base" else load_model_enriched
    create_trainer: Callable = create_trainer_base if dataset_type == "base" else create_trainer_enriched

    model = load_model(model_path, max_sequence_length=max_sequence_length, d_model=64)
    model = model.cuda()

    # load the validation parquet
    df = pl.read_parquet(paths)
    all_list_ids_inview: List[List[int]] = df["article_ids_inview"].to_list()

    trainer = create_trainer(
        model=model,
        output_dir="../checkpoints/tmp",
        epochs=1,
        history_size=max_sequence_length,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_steps=500,
        drop_last_batch=False,
    )

    # take the dataloader
    trainer.eval_dataset_or_path = paths
    dataloader = trainer.get_eval_dataloader()
    print(f"Ensure that Dataloader is deterministic: {not dataloader.shuffle}")

    # for each batch from the dataloader
    all_inview_scores: List[List[float]] = []
    index: int = 0
    for sp_batch in tqdm(dataloader):
        # make all the data in the batch to be in the cuda
        history = {name: value.cuda() for name, value in sp_batch[0].items()}

        # make the prediction and do softmax
        preds = model.forward(history)
        preds = torch.nn.functional.softmax(preds, dim=1)

        # for all the predictions find the inview scores
        for list_of_probs in preds.cpu().detach().numpy():
            ids_inview = all_list_ids_inview[index]
            inview_scores = get_inview_articles_score(ids_inview, list_of_probs)
            all_inview_scores.append(inview_scores)
            index += 1
    return all_inview_scores, df


def compute_metrics(df: pd.DataFrame, all_inview_scores: List[List[float]]) -> dict:
    # compute and print the metrics
    metrics = MetricEvaluator(
        labels=df["labels"].to_list(),
        predictions=all_inview_scores,
        metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
    )
    print("METRICS:")
    matrixs_evaluated = metrics.evaluate()
    print(matrixs_evaluated)
    return matrixs_evaluated


def add_score_in_df(
    df: pd.DataFrame, all_inview_scores: List[List[float]], save_path: str = None
) -> pd.DataFrame:
    # put the scores in validation
    df = df.with_columns(pl.Series("scores", all_inview_scores))
    # put the ranking in validation
    df = df.with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("ranked_scores")
    )

    print(df.head(5))

    if save_path:
        df.write_parquet(save_path)

    return df


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()

    if args.data_category not in ["test", "validation"]:
        raise ValueError(f"data_category must be either 'test' or 'validation'")

    print(f"Execution type: {args.data_category}")
    print(f"Model path: {args.path}")

    data_path: str = "final" if args.dataset_type == "base" else "advanced_ts"

    paths = [
        f"../data/{data_path}_ebnerd_processed/{args.data_category}/{args.data_category}_data_{args.split}_0.parquet",
        f"../data/{data_path}_ebnerd_processed/{args.data_category}/{args.data_category}_data_{args.split}_1.parquet",
        f"../data/{data_path}_ebnerd_processed/{args.data_category}/{args.data_category}_data_{args.split}_2.parquet",
        f"../data/{data_path}_ebnerd_processed/{args.data_category}/{args.data_category}_data_{args.split}_3.parquet",
        f"../data/{data_path}_ebnerd_processed/{args.data_category}/{args.data_category}_data_{args.split}_4.parquet",
        f"../data/{data_path}_ebnerd_processed/{args.data_category}/{args.data_category}_data_{args.split}_5.parquet",
        f"../data/{data_path}_ebnerd_processed/{args.data_category}/{args.data_category}_data_{args.split}_6.parquet",
    ]
    print(f"Split: {args.split} with paths: {paths}")

    # do inference
    eval_batch_size: int = args.eval_batch_size
    all_inview_scores, df = inference(args.path, paths, eval_batch_size, 20, args.dataset_type)

    # save what we need
    add_score_in_df(df, all_inview_scores, args.save_path)
    if args.data_category == "validation":
        compute_metrics(df, all_inview_scores)
