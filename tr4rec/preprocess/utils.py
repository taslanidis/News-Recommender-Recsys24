import polars as pl

from collections import defaultdict, OrderedDict
from typing import Dict


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
