import argparse
import os
import random

from nsc.data.tokenization import Tokenizers, WordTokenizer, BPETokenizer, CharTokenizer, TokenizerConfig
from nsc.utils import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer",
                        choices=[tok.name for tok in Tokenizers],
                        type=str,
                        required=True)
    in_file_group = parser.add_mutually_exclusive_group()
    in_file_group.add_argument("--in-files", type=str, nargs="+")
    in_file_group.add_argument("--in-file", type=str, nargs="+")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--out-file",
                        type=str,
                        required=True)
    parser.add_argument("--vocab-size",
                        type=int,
                        default=10000)
    parser.add_argument("--bpe-prefix-space",
                        action="store_true")
    return parser.parse_args()


def train_tokenizer(args: argparse.Namespace) -> None:
    logger = common.get_logger("TRAIN_TOKENIZER")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if os.path.exists(args.out_file):
        logger.info(f"{args.tokenizer} tokenizer at {args.out_file} already exists")
        return

    if args.in_file:
        files = []
        for file in args.in_file:
            with open(file, "r", encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    files.append(os.path.join(os.path.dirname(file), line))
    else:
        files = args.in_files

    random.shuffle(files)

    if len(files) == 0:
        logger.error("Got no input files")
        return

    if args.tokenizer == Tokenizers.CHAR.name:
        tokenizer = CharTokenizer()
    elif args.tokenizer == Tokenizers.WORD.name:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        WordTokenizer.train(files, args.out_file, args.vocab_size, args.max_sequences)
        tokenizer = WordTokenizer(TokenizerConfig(
            type=Tokenizers.WORD,
            file_path=args.out_file
        ))
    elif args.tokenizer == Tokenizers.BPE.name:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        BPETokenizer.train(files, args.out_file, args.vocab_size, args.max_sequences, args.bpe_prefix_space)
        tokenizer = BPETokenizer(TokenizerConfig(
            type=Tokenizers.BPE,
            file_path=args.out_file
        ))
    else:
        raise ValueError(f"Unexpected tokenizer {args.tokenizer}")

    test_sentence = "Test the tokenization with this sentence"

    logger.info(f"Testing {args.tokenizer} tokenizer (vocab_size={tokenizer.vocab_size}):")
    logger.info(f"Test sequence: {test_sentence}")
    logger.info(f"Split: {tokenizer.split(test_sentence)}")
    token_ids = tokenizer.tokenize(test_sentence)
    logger.info(f"Tokenize: {token_ids}")
    logger.info(f"De-tokenize: {tokenizer.de_tokenize(token_ids)}")


if __name__ == "__main__":
    train_tokenizer(parse_args())
