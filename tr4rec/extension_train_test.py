import torch
import argparse
import polars as pl
import pandas as pd
import numpy as np

from typing import List, Tuple


from models.t4rec_enriched import load_model, create_trainer
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from torch.utils.data import DataLoader
from dataloaders.extension import ParquetDataset, dataloader_collate_fn, preprocess
from models.pci import PredictClickedInview
from models.pci_attn import AttnPredictClickedInview

from torch import nn
from tqdm import tqdm
from typing import List, Tuple


def get_args_parser():
    parser = argparse.ArgumentParser("RecSys extension train-test", add_help=False)
    parser.add_argument(
        "--split",
        default="small",
        type=str,
        metavar="taskssplit",
        help="select small or large ",
    )
    parser.add_argument(
        "--model_trained_on",
        default="small",
        type=str,
        metavar="model_trained_on",
        help="tr4rec model trained on data split",
    )

    parser.add_argument(
        "--epochs", default=2, type=int, metavar="epochs", help="select epochs number"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        default=2048,
        type=int,
        metavar="train_batch",
        help="train batch size",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        default=2048,
        type=int,
        metavar="eval_batch",
        help="eval batch size",
    )

    parser.add_argument(
        "--extension_model",
        default="attn",
        type=str,
        metavar="extension_model",
        choices=["attn", "mlp"],
        help="extension_model",
    )

    return parser


def pipeline(
    epochs: int, 
    data_split: str, 
    model_trained_on: str, 
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    extension_model: str
    ):
    max_sequence_length: int = 20
    output_dir = f"checkpoints/{data_split}-{model_trained_on}-extension"

    if model_trained_on == "small":
        model_path = "checkpoints/enriched/small/checkpoint-226/pytorch_model.bin"
    else:
        model_path = "checkpoints/enriched/large/checkpoint-23560/pytorch_model.bin"

    model = load_model(model_path, max_sequence_length=max_sequence_length, d_model=64)
    model = model.cuda()
    params = {}
    params["max_inview"] = 100
    params["user_feature_dim"] = 0
    params["ovewrite_meta_model"] = True
    params["use_summarizer"] = True
    params["use_metadata"] = False

    if extension_model == "attn":
        predict_model = AttnPredictClickedInview(model, params)
    else: 
        predict_model = PredictClickedInview(model, params)
    
    predict_model = predict_model.cuda()

    trainer = create_trainer(
        model,
        output_dir,
        epochs,
        max_sequence_length,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        500,
    )

    train_paths = [
            f"data/advanced_ts_ebnerd_processed/train/train_data_{data_split}_0.parquet",
            f"data/advanced_ts_ebnerd_processed/train/train_data_{data_split}_1.parquet",
            f"data/advanced_ts_ebnerd_processed/train/train_data_{data_split}_2.parquet",
            f"data/advanced_ts_ebnerd_processed/train/train_data_{data_split}_3.parquet",
            f"data/advanced_ts_ebnerd_processed/train/train_data_{data_split}_4.parquet",
            f"data/advanced_ts_ebnerd_processed/train/train_data_{data_split}_5.parquet",
            f"data/advanced_ts_ebnerd_processed/train/train_data_{data_split}_6.parquet",
        ]

    trainer.train_dataset_or_path = train_paths
    merlin_dataloader = trainer.get_train_dataloader()

    train_dataset = ParquetDataset(train_paths, ["age"], ['article_ids_inview', 'labels'], transform=preprocess)

    prq_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dataloader_collate_fn,
    )

    predict_model.train()
    loss_module = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.SGD(
        predict_model.parameters(), lr=1e-2, momentum=0.9
    )

    print("Starting training...")
    for i in range(epochs):
        predict_model.train()
        epoch_loss = 0
        for sample in zip(tqdm(merlin_dataloader), prq_dataloader):
            # convert inputs to cuda
            history, (user, inviews, length, labels) = sample
            history = {name: value.cuda() for name, value in history[0].items()}
            inviews, length = inviews.cuda(), length.cuda()
            user = user.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            preds = predict_model.forward(history, user, inviews, length)

            loss = loss_module(preds.squeeze(-1), labels.float())

            # Apply the mask to zero out the loss contributions from the padding elements
            mask = (torch.arange(100).expand(len(length.cpu()), 100) < length.cpu().unsqueeze(1)).cuda()
            masked_loss = loss * mask.float()

            # Calculate the mean loss over the non-padding elements
            mean_loss = masked_loss.sum() / mask.sum()
            
            epoch_loss += mean_loss
            mean_loss.backward()
            optimizer.step()
        
        print("Epoch Loss: ", epoch_loss / len(merlin_dataloader))

    print("Finished training")

    eval_paths = [
            f"data/advanced_ts_ebnerd_processed/validation/validation_data_{data_split}_0.parquet",
            f"data/advanced_ts_ebnerd_processed/validation/validation_data_{data_split}_1.parquet",
            f"data/advanced_ts_ebnerd_processed/validation/validation_data_{data_split}_2.parquet",
            f"data/advanced_ts_ebnerd_processed/validation/validation_data_{data_split}_3.parquet",
            f"data/advanced_ts_ebnerd_processed/validation/validation_data_{data_split}_4.parquet",
            f"data/advanced_ts_ebnerd_processed/validation/validation_data_{data_split}_5.parquet",
            f"data/advanced_ts_ebnerd_processed/validation/validation_data_{data_split}_6.parquet",
        ]

    validation_df = pd.read_parquet(eval_paths, engine="pyarrow")

    trainer.eval_dataset_or_path = eval_paths
    dlv = trainer.get_eval_dataloader()
    val_dataset = ParquetDataset(eval_paths, ["age"], ['article_ids_inview', 'labels'], transform=preprocess)

    val_dataset.max_inview_lenght = train_dataset.max_inview_lenght # set train max length

    eval_prq_dataloader = DataLoader(val_dataset, batch_size=per_device_eval_batch_size, shuffle=False, drop_last=False, collate_fn=dataloader_collate_fn)


    print(f"Ensure that Validation Dataloader is deterministic: {not dlv.shuffle}")

    print("Test Inference...")
    all_inview_scores = []
    index: int = 0
    predict_model.eval()
    # Sample one from data loader
    for sp_batch in zip(tqdm(dlv), eval_prq_dataloader):
        history, (user, inviews, length, labels) = sp_batch
        history = {name: value.cuda() for name, value in history[0].items()}
        inviews, length = inviews.cuda(), length.cuda()
        user = user.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            preds = predict_model.forward(history, user, inviews, length)
        
        preds = preds.squeeze(-1).cpu().detach().numpy()
        preds = [p[:ln] for p, ln in zip(preds, length)]

        all_inview_scores.extend(preds)


    print("Calculating metrics...")
    metrics = MetricEvaluator(
        labels=validation_df["labels"].to_list()[:-7],
        predictions=all_inview_scores,
        metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
    )
    metrics.evaluate()



if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    print(f"Split: {args.split}")
    print(f"Tr4Rec trained on: {args.model_trained_on}")
    print(f"Extension model: {args.extension_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Train batch size: {args.per_device_train_batch_size}")
    print(f"Evaluate batch size: {args.per_device_eval_batch_size}")

    pipeline(
        data_split=args.split,
        model_trained_on=args.model_trained_on,
        epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        extension_model=args.extension_model
    )