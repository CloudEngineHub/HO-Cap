import argparse
from hocap_toolkit.factory import HOCapFactory


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Split HOCAP dataset")
    parser.add_argument(
        "--task",
        type=str,
        choices=["hpe", "odet", "ope"],
        required=True,
        help="Dataset task (hpe, odet, ope)",
    )
    parser.add_argument(
        "--anno_type",
        type=str,
        default="coco",
        choices=["coco", "yolo"],
        help="Annotation type for odet (coco, yolo)",
    )
    args = parser.parse_args()

    factory = HOCapFactory()

    if args.task == "hpe":
        factory.create_hpe_dataset()

    if args.task == "odet":
        factory.create_odet_dataset(args.anno_type)

    if args.task == "ope":
        factory.create_ope_dataset()
