import argparse
import os.path

from nsc.api.utils import load_text_file, save_text_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    return parser.parse_args()


def sec_to_sed(args: argparse.Namespace) -> None:
    if os.path.exists(args.out_file):
        print(f"out file {args.out_file} already exists")
        return

    predictions = load_text_file(args.prediction_file)
    inputs = load_text_file(args.input_file)
    assert len(predictions) == len(inputs)

    sed_predictions = []
    for i, (prediction, ipt) in enumerate(zip(predictions, inputs)):
        predicted_words = prediction.split()
        ipt_words = ipt.split()
        sed_predictions.append(" ".join([str(int(p != i)) for p, i in zip(predicted_words, ipt_words)]))

    save_text_file(args.out_file, sed_predictions)


if __name__ == "__main__":
    sec_to_sed(parse_args())
