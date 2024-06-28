import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader


def preprocess(data, user_col, inview_col):
    for i in user_col:
        if i in ["age", "device_type", "gender", "postcode"]:
            data = data.with_columns(pl.col(i).fill_null(-1))

    columns_to_drop = list(
        set(data.columns)
        - set(user_col)
        - set(inview_col)
        - set(["article_ids_clicked"])
        - set(["is_subscriber"])
    )
    data = data.drop(columns_to_drop)
    return data


class ParquetDataset(Dataset):
    def __init__(
        self,
        file_path,
        user_col,
        inview_col,
        transform=None,
        contains_labels=True,
    ):
        self.file_path = file_path
        self.data = pl.read_parquet(file_path)
        if transform:
            self.data = transform(self.data, user_col, inview_col)
        self.user_col = user_col
        self.inview_col = inview_col

        self.contains_labels = contains_labels
        self.max_inview_lenght = (
            self.data.with_columns(len=pl.col("article_ids_inview").list.len())
            .select(pl.max("len"))
            .to_numpy()[0][0]
        )

        self.length_inviews = torch.tensor(
            [len(i) for i in self.data["article_ids_inview"]]
        )
        self.padding_inview_col = torch.nn.utils.rnn.pad_sequence(
            [i.to_torch() for i in self.data["article_ids_inview"]],
            batch_first=True,
            padding_value=0,
        )
        # TODO: bring validation same as train
        self.padding_inview_col = torch.nn.functional.pad(
            self.padding_inview_col,
            (0, 100 - self.padding_inview_col.shape[1]),
            "constant",
            0,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = torch.tensor([0])
        if self.contains_labels:
            labels = self.data["labels"][idx].to_torch()
            labels = torch.nn.functional.pad(
                labels, (0, self.max_inview_lenght - labels.shape[0]), "constant", 0
            )  # TODO: Fanis changed this from 0 to 2

        if len(self.user_col) > 0:
            return (
                self.data[self.user_col][idx].to_torch(),
                self.padding_inview_col[idx].unsqueeze(0),
                self.length_inviews[idx].unsqueeze(0),
                labels.unsqueeze(0),
            )

        return (
                self.padding_inview_col[idx].unsqueeze(0),
                self.length_inviews[idx].unsqueeze(0),
                labels.unsqueeze(0),
            )


def dataloader_collate_fn(inputs):
    user_col, inview_col, length_inviews, labels = zip(*inputs)
    return (
        torch.cat(user_col, 0),
        torch.cat(inview_col, 0),
        torch.cat(length_inviews, 0),
        torch.cat(labels, 0),
    )


def dataloader_collate_fn_inview_only(inputs):
    inview_col, length_inviews, labels = zip(*inputs)
    return (
        torch.cat(inview_col, 0),
        torch.cat(length_inviews, 0),
        torch.cat(labels, 0),
    )
