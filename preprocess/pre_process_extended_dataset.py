from merlin.io import Dataset
from nvtabular import ops
import nvtabular as nvt
from merlin.schema.tags import Tags

from merlin.schema import Schema, ColumnSchema, Tags

import polars as pl
import numpy as np
import pyarrow as pa

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_READ_TIME_COL,
)


from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
import argparse


from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from collections import OrderedDict, defaultdict
from typing import Dict

import torch
import polars as pl

import pandas as pd


def mapper_article_to_index(row, mapper: dict, unknown: int = 0) -> list:
    return [mapper.get(item, unknown) for item in row]


def mapper_article_to_val(row, mapper: dict, col: str):
    return [mapper.get(item).get(col) for item in row]


def convert_article_id_to_index(
    df: pl.DataFrame, mapper: dict, data_category: str
) -> pl.DataFrame:

    if data_category == "test":
        df = df.with_columns(
            [
                pl.col("article_id_fixed")
                .map_elements(
                    lambda row: mapper_article_to_index(row, mapper),
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("article_id_fixed"),
                pl.col("article_ids_inview")
                .map_elements(
                    lambda row: mapper_article_to_index(row, mapper),
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("article_ids_inview"),
            ]
        )
    else:
        df = df.with_columns(
            [
                pl.col("article_id_fixed")
                .map_elements(
                    lambda row: mapper_article_to_index(row, mapper),
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("article_id_fixed"),
                pl.col("article_ids_inview")
                .map_elements(
                    lambda row: mapper_article_to_index(row, mapper),
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("article_ids_inview"),
                pl.col("article_ids_clicked")
                .map_elements(
                    lambda row: mapper_article_to_index(row, mapper),
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("article_ids_clicked"),
            ]
        )
    return df


def construct_lookup_from_df(df_articles) -> Dict:
    articles_dict_df = df_articles.to_dict(as_series=False)
    articles_dict_df.keys()

    columns_needed: list = [
        'category', 'subcategory', 'article_type',
        'sentiment_score', 'topics', 'total_pageviews'
    ]

    articles_lookup: dict = defaultdict(dict)

    for col in columns_needed:
        for article_id, val in zip(articles_dict_df['article_id'], articles_dict_df[col]):
            articles_lookup[article_id][col] = val

    for col in columns_needed:
        if col in ['category', 'total_pageviews']:
            articles_lookup[0][col] = 0
        elif col in ['topics', 'subcategory']:
            articles_lookup[0][col] = []
        elif col in ['sentiment_score']:
            articles_lookup[0][col] = 0.5
        else:
            articles_lookup[0][col] = "Unknown"

    del articles_dict_df
    
    return articles_lookup


def get_args_parser():
    parser = argparse.ArgumentParser("RecSys pre-process", add_help=False)
    parser.add_argument(
        "--split",
        default="small",
        type=str,
        metavar="taskssplit",
        help="select small or large or testset",
    )
    parser.add_argument(
        "--data_category",
        default="train",
        type=str,
        metavar="cat",
        help="train or validation or test",
    )
    parser.add_argument(
        "--history_size",
        default=20,
        type=int,
        metavar="hist",
        help="select history size",
    )

    return parser


def main_flow(split, data_category, history_size):
    pathfile = f"/home/scur1565/News-Recommender-Recsys24/data/ebnerd_{split}/{data_category}/behaviors.parquet"

    behaviors = pd.read_parquet(pathfile, engine="pyarrow")

    print("this is the length of behavior", len(behaviors))

    # split processing in 7 parts
    max_samples = len(behaviors)
    file_num: int = 7
    batch: int = max_samples // 6

    # Initialize an empty schema
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


    ########################
    ### Pre-Trained Embs ###
    ########################

    # load pre_trained embeddings
    pretrained_embeds_df = pl.read_parquet(
        "/home/scur1565/News-Recommender-Recsys24/data/eb_contrastive_vector/contrastive_vector.parquet"
    )

    article_to_index = {
        art_id: num + 1
        for num, art_id in enumerate(pretrained_embeds_df["article_id"].to_list())
    }

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

    ##################################
    ### Load Article Lookup values ###
    ##################################

    df_articles = (
        pl.read_parquet(
            f"/home/scur1565/News-Recommender-Recsys24/data/ebnerd_{split}/articles.parquet"
        )
        .select(
            "article_id",
            "category",
            "subcategory",
            "article_type",
            "sentiment_score",
            "topics",
            "total_pageviews"
        )
    )

    articles_lookup: dict = construct_lookup_from_df(df_articles)
    del df_articles

    ############################
    ### Load History Dataset ###
    ############################

    df_history = (
        pl.scan_parquet(
            f"/home/scur1565/News-Recommender-Recsys24/data/ebnerd_{split}/{data_category}/history.parquet"
        )
        .select(
            DEFAULT_USER_COL,
            DEFAULT_HISTORY_ARTICLE_ID_COL,
            DEFAULT_HISTORY_READ_TIME_COL,
        )
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_READ_TIME_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
        .cast(
            {
                "article_id_fixed": pl.List(pl.Int32),
                "read_time_fixed": pl.List(pl.Float32),
            }
        )
    )

    ####################################
    ### Enrich History from Articles ###
    ####################################

    df_history = df_history.with_columns(
        [
            pl.col("article_id_fixed")
            .map_elements(
                lambda row: mapper_article_to_val(row, articles_lookup, "sentiment_score"),
                return_dtype=pl.List(pl.Float64),
            )
            .alias("sentiment_list"),
            pl.col("article_id_fixed")
            .map_elements(
                lambda row: mapper_article_to_val(row, articles_lookup, "category"),
                return_dtype=pl.List(pl.Int64),
            )
            .alias("category_list"),
            pl.col("article_id_fixed")
            .map_elements(
                lambda row: mapper_article_to_val(row, articles_lookup, "article_type"),
                return_dtype=pl.List(pl.String),
            )
            .alias("article_type_list"),
            pl.col("article_id_fixed")
            .map_elements(
                lambda row: mapper_article_to_val(row, articles_lookup, "total_pageviews"),
                return_dtype=pl.List(pl.Int64),
            )
            .alias("total_pageviews_list")
        ]
    )

    ############################
    ### Create Train Dataset ###
    ############################

    for i in range(file_num):
        print(f"Calculating file number: {i+1}")

        if data_category == "test":
            print("---------------------")
            print(i)
            print(batch)
            print(i * batch)
            print(file_num)

            total_df = (
                pl.scan_parquet(
                    f"/home/scur1565/News-Recommender-Recsys24/data/ebnerd_{split}/{data_category}/behaviors.parquet",
                )
                .slice(i * batch, batch)
                .select(
                    "user_id",
                    "article_ids_inview",
                    "age",
                    "device_type",
                    "gender",
                    "postcode",
                    "is_subscriber",
                    "impression_id",
                )
                .cast(
                    {
                        "article_ids_inview": pl.List(pl.Int32),
                    }
                )
                .collect()
                .pipe(
                    slice_join_dataframes,
                    df2=df_history.collect(),
                    on=DEFAULT_USER_COL,
                    how="left",
                )
                .pipe(
                    convert_article_id_to_index,
                    mapper=article_to_index,
                    data_category=data_category,
                )
            )
            pa_schema = pa.schema(
                [
                    ("user_id", pa.int32()),
                    ("article_ids_inview", pa.list_(pa.int32())),
                    ("age", pa.int32()),
                    ("device_type", pa.int32()),
                    ("gender", pa.int32()),
                    ("postcode", pa.int32()),
                    ("is_subscriber", pa.bool_()),
                    ("impression_id", pa.int32()),
                    ("article_id_fixed", pa.list_(pa.int32())),
                    ("read_time_fixed", pa.list_(pa.float32())),
                    ("sentiment_list", pa.list_(pa.float32())),
                    ("category_list", pa.list_(pa.int32())),
                    ("total_pageviews_list", pa.list_(pa.int32())),
                    ("article_type_list", pa.list_(pa.string()))
                ]
            )

        else:

            total_df = (
                pl.scan_parquet(
                    f"/home/scur1565/News-Recommender-Recsys24/data/ebnerd_{split}/{data_category}/behaviors.parquet",
                )
                .slice(i * batch, batch)
                .select(
                    "user_id",
                    "article_ids_inview",
                    "article_ids_clicked",
                    "age",
                    "device_type",
                    "gender",
                    "postcode",
                    "is_subscriber",
                    "impression_id",
                )
                .cast(
                    {
                        "article_ids_inview": pl.List(pl.Int32),
                        "article_ids_clicked": pl.List(pl.Int32),
                    }
                )
                .collect()
                .pipe(
                    slice_join_dataframes,
                    df2=df_history.collect(),
                    on=DEFAULT_USER_COL,
                    how="left",
                )
                .pipe(create_binary_labels_column)
                .pipe(
                    convert_article_id_to_index,
                    mapper=article_to_index,
                    data_category=data_category,
                )
            )

            pa_schema = pa.schema(
                [
                    ("user_id", pa.int32()),
                    ("article_ids_inview", pa.list_(pa.int32())),
                    ("article_ids_clicked", pa.list_(pa.int32())),
                    ("age", pa.int32()),
                    ("device_type", pa.int32()),
                    ("gender", pa.int32()),
                    ("postcode", pa.int32()),
                    ("is_subscriber", pa.bool_()),
                    ("impression_id", pa.int32()),
                    ("article_id_fixed", pa.list_(pa.int32())),
                    ("read_time_fixed", pa.list_(pa.float32())),
                    ("sentiment_list", pa.list_(pa.float32())),
                    ("category_list", pa.list_(pa.int32())),
                    ("article_type_list", pa.list_(pa.string())),
                    ("total_pageviews_list", pa.list_(pa.int32())),
                    ("labels", pa.list_(pa.int32())),
                ]
            )

        total_df = total_df.to_arrow()
        total_df = total_df.cast(pa_schema)

        print(total_df)

        pa.parquet.write_table(
            total_df,
            f"/home/scur1565/News-Recommender-Recsys24/data/advanced_ts_ebnerd_processed/{data_category}/{data_category}_data_{split}_{i}.parquet",
        )


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    print(f"Split: {args.split}")
    print(f"Data category: {args.data_category}")
    print(f"History size: {args.history_size}")

    main_flow(
        split=args.split,
        data_category=args.data_category,
        history_size=args.history_size,
    )
