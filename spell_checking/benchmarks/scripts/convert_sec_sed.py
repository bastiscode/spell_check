import argparse
import os.path

from nsc.api.utils import load_text_file, save_text_file
from nsc.utils import edit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--convert",
                        choices=["sedw_to_seds", "sec_to_sedw", "sec_to_seds"],
                        required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sec_to_sed(args: argparse.Namespace) -> None:
    if os.path.exists(args.out_file) and not args.overwrite:
        print(f"out file {args.out_file} already exists")
        return

    predictions = load_text_file(args.prediction_file)
    inputs = load_text_file(args.input_file)
    assert len(predictions) == len(inputs)

    sed_predictions = []
    if args.convert == "sec_to_sedw":
        batch_edited_indices, _ = edit.get_edited_words(inputs, predictions)
        for edited_indices, ipt in zip(batch_edited_indices, inputs):
            sed_predictions.append(" ".join(["0" if i not in edited_indices else "1" for i in range(len(ipt.split()))]))
    elif args.convert == "sec_to_seds":
        for ipt, pred in zip(inputs, predictions):
            sed_predictions.append(str(int(ipt != pred)))
    else:
        for pred in predictions:
            sed_predictions.append(str(int(any([int(p) for p in pred.split()]))))

    save_text_file(args.out_file, sed_predictions)


if __name__ == "__main__":
    sec_to_sed(parse_args())
