import argparse

from preprocess.base_pipe import main_flow as main_flow_base
from preprocess.enriched_pipe import main_flow as main_flow_enriched


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
    parser.add_argument(
        "--dataset_type",
        default="base",
        type=str,
        choices=["base", "enriched"],
        metavar="dataset_type",
        help="select dataset type",
    )
    return parser



if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    print(f"Split: {args.split}")
    print(f"Data category: {args.data_category}")
    print(f"History size: {args.history_size}")

    if args.dataset_type == "base":
        main_flow_base(
            split=args.split,
            data_category=args.data_category,
            history_size=args.history_size,
        )
    else:
        main_flow_enriched(
            split=args.split,
            data_category=args.data_category,
            history_size=args.history_size,
        )