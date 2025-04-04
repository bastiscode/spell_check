import argparse
import os
import pprint
import random
from typing import List, Tuple

import altair as alt
import pandas as pd

from nsc.api import tables
from nsc.api.utils import load_text_file, save_text_file
from nsc.data import utils
from nsc.utils import io, common
from nsc.utils.metrics import is_real_word
from nsc.utils.edit import edit_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=str, required=True)
    parser.add_argument("--benchmark-type", choices=["sed_sequence", "sed_words_and_sec"], required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--fmt", choices=["markdown", "latex"], default="latex")
    return parser.parse_args()


def generate_edit_distance_histogram(edit_distances: List[Tuple[int, bool]]) -> alt.Chart:
    edit_dists = [item[0] for item in edit_distances]
    real_word = [item[1] for item in edit_distances]
    data = pd.DataFrame({"ed": edit_dists, "rw": real_word})
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("ed:Q", bin=True),
        y="count()",
        color="rw:N"
    )
    return chart


def generate_statistics(args: argparse.Namespace) -> None:
    logger = common.get_logger("BENCHMARK_STATISTICS")
    dictionary = io.dictionary_from_file(args.dictionary)

    benchmarks = io.glob_safe(os.path.join(args.benchmark_dir, "*", "*"))

    benchmark_groups = {}
    for benchmark in benchmarks:
        group, split = benchmark.split("/")[-2:]
        if group not in {"wikidump", "bookcorpus", "neuspell"} or (
                args.benchmark_type == "sed_sequence" and split in {"bea322", "bea4660"}
        ):
            continue
        if group not in benchmark_groups:
            benchmark_groups[group] = []
        benchmark_groups[group].append(split)

    logger.info(f"Found the following benchmark groups: {benchmark_groups}")

    samples = {}
    edit_distance_statistics = {}
    statistics = {}
    for group, splits in benchmark_groups.items():
        group_edit_distance_statistics = {}
        group_statistics = {}
        group_samples = {}
        for split in splits:
            corrupt_lines = load_text_file(os.path.join(args.benchmark_dir, group, split, "corrupt.txt"))
            correct_lines = load_text_file(os.path.join(args.benchmark_dir, group, split, "correct.txt"))
            correct_sequences_file = os.path.join(args.benchmark_dir, group, split, "correct.sequences.txt")
            if os.path.exists(correct_sequences_file):
                correct_sequences = load_text_file(correct_sequences_file)
            else:
                correct_sequences = [None] * len(corrupt_lines)

            assert len(correct_lines) == len(corrupt_lines) == len(correct_sequences)

            num_sequences = len(corrupt_lines)
            avg_seq_length = sum(len(l) for l in corrupt_lines) / num_sequences
            num_words = 0
            real_word_errors = 0
            non_word_errors = 0
            error_edit_distances = []
            additional_stats = []
            real_word_misspellings = {}
            non_word_misspellings = {}
            for corrupt_line, correct_line, correct_sequence in zip(corrupt_lines, correct_lines, correct_sequences):
                corrupt_words = corrupt_line.split()
                correct_split = correct_line.split()
                correct_words = correct_sequence.split() if correct_sequence is not None else correct_split
                assert len(corrupt_words) == len(correct_words)

                for correct_word, corrupt_word in zip(correct_words, corrupt_words):
                    if correct_word == corrupt_word:
                        continue

                    if corrupt_word in dictionary:
                        if corrupt_word not in real_word_misspellings and dictionary[corrupt_word] >= 10000:
                            real_word_misspellings[corrupt_word] = correct_word
                    else:
                        if corrupt_word not in non_word_misspellings:
                            non_word_misspellings[corrupt_word] = correct_word

                for corrupt_word, correct_word in zip(corrupt_words, correct_words):
                    num_words += len(utils.tokenize_words_regex(corrupt_word)[0])
                    ed = edit_distance(corrupt_word, correct_word)
                    if corrupt_word != correct_word:
                        if is_real_word(corrupt_word, dictionary):
                            real_word_errors += 1
                            error_edit_distances.append((ed, True))
                        else:
                            non_word_errors += 1
                            error_edit_distances.append((ed, False))

                if args.benchmark_type == "sed_sequence":
                    if len(additional_stats) == 0:
                        additional_stats.append(0)
                    assert len(correct_split) == 1
                    additional_stats[0] += int(correct_split[0])

            total_errors = real_word_errors + non_word_errors

            group_samples[split] = (real_word_misspellings, non_word_misspellings)
            group_statistics[split] = (
                num_sequences,
                avg_seq_length,
                num_words,
                real_word_errors,
                non_word_errors,
                total_errors,
                additional_stats
            )
            group_edit_distance_statistics[split] = error_edit_distances

        samples[group] = group_samples
        statistics[group] = group_statistics
        edit_distance_statistics[group] = group_edit_distance_statistics

    logger.info(f"Got the following statistics:\n{pprint.pformat(statistics)}")

    groups = sorted(statistics)
    headers = [
        ["Benchmark", r"\#Sequences", r"\#Words", r"$\overline{\text{Sequence length}}\tnote{*}$",
         r"Word errors\tnote{\textdagger}", r"Real-word errors\tnote{\textdaggerdbl}",
         r"Nonword errors\tnote{\textdaggerdbl}"]
    ]
    is_sed_sequence = args.benchmark_type == "sed_sequence"
    if is_sed_sequence:
        headers[0].extend(["Sequence errors"])

    num_samples = 5
    rand = random.Random(24)
    sample_headers = [["Error type"], [""]]
    sample_data = [[""] for _ in range(num_samples * 2)]
    sample_data[0][0] = "Real-word"
    sample_data[num_samples][0] = "Nonword"

    charts = []
    data = []
    horizontal_lines = []
    for group in groups:
        horizontal_lines.extend([False] * (len(statistics[group]) - 1))
        horizontal_lines.append(True)
        splits = sorted(statistics[group])
        for split in splits:
            num_seq, avg_seq_length, num_words, rw_err, nw_err, total_err, additional = statistics[group][split]
            error_percentage = 100 * total_err / num_words
            real_word_percentage = 100 * rw_err / total_err
            non_word_percentage = 100 * nw_err / total_err

            data.append([
                f"{group} {split}", f"{num_seq:,}", f"{num_words:,}", f"{avg_seq_length:.1f}",
                f"{total_err:,} ({error_percentage: >4.1f}%)",
                f"{rw_err:,} ({real_word_percentage: >4.1f}%)", f"{nw_err:,} ({non_word_percentage: >4.1f}%)",
            ])

            if is_sed_sequence:
                sequences_with_errors = additional[0]
                sequence_error_percentage = 100 * sequences_with_errors / num_seq
                data[-1].extend([f"{sequences_with_errors:,} ({sequence_error_percentage:.1f}%)"])

            sample_headers[0].append(group)
            sample_headers[1].append(split)

            for j, misspelling_pairs in enumerate(samples[group][split]):
                misspelling_pairs = list(misspelling_pairs.items())
                indices = list(range(len(misspelling_pairs)))
                rand.shuffle(indices)
                for i, idx in enumerate(indices[:num_samples]):
                    sample_data[i + j * num_samples].append(
                        f"{misspelling_pairs[idx][0]} --> {misspelling_pairs[idx][1]}"
                    )

            chart = generate_edit_distance_histogram(edit_distance_statistics[group][split])
            charts.append(chart)

    combined_chart = alt.vconcat(*[alt.hconcat(*charts[i:i + 2]) for i in range(0, len(charts), 2)])
    combined_chart.save(os.path.join(args.out_dir, "edit_distance_chart.pdf"))

    table = tables.generate_table(
        headers=headers,
        data=data,
        horizontal_lines=horizontal_lines,
        fmt=args.fmt
    )

    # sample_table = tables.generate_table(
    #     headers=sample_headers,
    #     data=sample_data,
    #     horizontal_lines=([False] * (num_samples - 1) + [True]) * 2,
    #     fmt=args.fmt
    # )

    logger.info(f"Output table:\n{table}")
    # logger.info(f"Output sample table:\n{sample_table}")

    save_text_file(os.path.join(args.out_dir, "statistics.tex"), [table])
    # save_text_file(os.path.join(args.out_dir, "samples.tex"), [sample_table])


if __name__ == "__main__":
    generate_statistics(parse_args())
