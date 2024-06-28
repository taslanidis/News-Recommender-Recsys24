import pandas as pd
import polars as pl
import pyarrow as pa
import numpy as np

import torch
import argparse

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

from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from preprocess.utils import (
    construct_lookup_from_df, 
    mapper_article_to_val, 
    convert_article_id_to_index
)


def main_flow(split, data_category, history_size):
    pathfile = f"/home/scur1565/News-Recommender-Recsys24/data/ebnerd_{split}/{data_category}/behaviors.parquet"

    behaviors = pd.read_parquet(pathfile, engine="pyarrow")

    print("this is the length of behavior", len(behaviors))

    # split processing in 7 parts
    max_samples = len(behaviors)
    file_num: int = 7
    batch: int = max_samples // 6

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

    df_articles = df_articles.with_columns(
        pl.col("total_pageviews").fill_null(
            pl.col("total_pageviews").median().over("category")
        ).cast(pl.Int32)
    )

    # fill rest with zero
    df_articles = df_articles.with_columns(
        pl.col("total_pageviews").fill_null(0)
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