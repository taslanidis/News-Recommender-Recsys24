from merlin.schema.tags import Tags
from merlin.schema import Schema, ColumnSchema, Tags

import polars as pl
import numpy as np

import torch

from transformers4rec import torch as tr
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer
from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt

from typing import List, Tuple


def create_schema() -> Schema:
    schema = Schema()

    article_id_fixed_col = ColumnSchema(
        name="article_id_fixed",
        dtype="int32",
        tags=[Tags.LIST, Tags.CATEGORICAL, Tags.ITEM_ID, Tags.ITEM],
        is_list=True,
        is_ragged=True,
    ).with_properties(
        {"domain": {"min": 0, "max": 125541}, "value_count": {"min": 1, "max": 500}}
    )

    read_time_fixed_col = ColumnSchema(
        name="read_time_fixed",
        dtype="float32",
        tags=[Tags.LIST, Tags.CONTINUOUS],
        is_list=True,
        is_ragged=True,
    )

    # Add columns to the schema
    columns = [article_id_fixed_col, read_time_fixed_col]

    for col in columns:
        schema[col.name] = col

    return schema


def create_model(
    max_sequence_length: int, d_model: int, load: bool = False
) -> tr.Model:
    schema = create_schema()

    if not load:
        pretrained_embeds_df = pl.read_parquet(
            "/home/scur1565/News-Recommender-Recsys24/data/eb_contrastive_vector/contrastive_vector.parquet"
        )

        pretrained_embeds = pretrained_embeds_df["contrastive_vector"].to_list()
        pretrained_embeds = np.vstack(
            [
                np.zeros(
                    768,
                )
            ]
            + [np.array(vec) for vec in pretrained_embeds]
        )

        pretrained_embeds = torch.from_numpy(pretrained_embeds)

    emb_dims = {"article_id_fixed": 768}
    inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=max_sequence_length,
        continuous_projection=d_model,
        aggregation="concat",
        d_output=100,
        masking="causal",
        infer_embedding_sizes=True,
        embedding_dims=emb_dims,
    )

    if not load:
        with torch.no_grad():
            inputs.categorical_module.embedding_tables["article_id_fixed"].weight.copy_(
                pretrained_embeds
            )

        inputs.categorical_module.embedding_tables["article_id_fixed"].requires_grad = (
            False
        )
        inputs.categorical_module.embedding_tables[
            "article_id_fixed"
        ].weight.requires_grad = False

    # Define XLNetConfig class and set default parameters for HF XLNet config
    transformer_config = tr.XLNetConfig.build(
        d_model=d_model, n_head=4, n_layer=2, total_seq_length=max_sequence_length
    )

    body = tr.SequentialBlock(
        inputs,
        tr.MLPBlock([64]),
        tr.TransformerBlock(transformer_config, masking=inputs.masking),
    )

    metrics = [
        NDCGAt(top_ks=[20, 40], labels_onehot=True),
        RecallAt(top_ks=[20, 40], labels_onehot=True),
    ]

    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True, metrics=metrics),
        inputs=inputs,
    )

    model = tr.Model(head)

    return model


def load_model(model_path: str, max_sequence_length: int, d_model: int) -> tr.Model:
    # Add columns to the schema as done during training
    model = create_model(max_sequence_length, d_model, load=True)

    # Load the saved model checkpoint
    model.load_state_dict(torch.load(model_path))

    return model


def create_trainer(
    model: tr.Model,
    output_dir: str,
    epochs: int,
    history_size: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    save_steps: int,
    drop_last_batch: bool = True,
) -> tr.Trainer:
    # make the training arguments in order to take the dataloader
    train_args = T4RecTrainingArguments(
        data_loader_engine="merlin",
        dataloader_drop_last=drop_last_batch,
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
        logging_steps=200,
        save_steps=save_steps,
        no_cuda=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        schema=create_schema(),
    )
    return trainer
