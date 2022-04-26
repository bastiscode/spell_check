from typing import List, Tuple, Set

import edit_distance_rs


def edit_operations(
        a: str,
        b: str,
        with_swap: bool = False,
        spaces_insert_delete_only: bool = False
) -> List[Tuple[str, int, int]]:
    """
    Returns the edit operations transforming a into b, optionally allowing swap/transposition operations
    and restricting operations on spaces to insertions and deletions.
    Follows optimal string alignment distance algorithm
    from https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance. Uses a high performance implementation
    in Rust under the hood.

    Args:
        a: source string
        b: destination string
        with_swap: allow swap/transposition operations
        spaces_insert_delete_only: make sure that spaces can only be inserted or deleted

    Returns: list of edit operations

    """
    return edit_distance_rs.edit_operations(a, b, with_swap, spaces_insert_delete_only)


def batch_edit_operations(
        a_list: List[str],
        b_list: List[str],
        with_swap: bool = False,
        spaces_insert_delete_only: bool = False,
        batch_size: int = 256
) -> List[List[Tuple[str, int, int]]]:
    """
    For each a, b pair in a_list, b_list returns the edit operations transforming a into b,
    optionally allowing swap/transposition operations and restricting operations on spaces to insertions and deletions.
    Follows optimal string alignment distance algorithm
    from https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance. Uses a high performance implementation
    in Rust under the hood.

    Args:
        a_list: list of source strings
        b_list: list of destination strings
        with_swap: allow swap/transposition operations
        spaces_insert_delete_only: make sure that spaces can only be inserted or deleted
        batch_size: number of strings per batch

    Returns: list of lists of edit operations

    """
    return edit_distance_rs.batch_edit_operations(a_list, b_list, with_swap, spaces_insert_delete_only, batch_size)


def edit_distance(
        a: str,
        b: str,
        with_swap: bool = False,
        spaces_insert_delete_only: bool = False
) -> int:
    """
    Calculate the minimum number of edit operations required to turn a into b. Uses a high performance implementation
    in Rust under the hood.

    Args:
        a: source string
        b: destination string
        with_swap: allow swap/transposition operations
        spaces_insert_delete_only: make sure that spaces can only be inserted or deleted

    Returns: edit distance

    """
    return edit_distance_rs.edit_distance(a, b, with_swap, spaces_insert_delete_only)


def batch_edit_distance(
        a_list: List[str],
        b_list: List[str],
        with_swap: bool = False,
        spaces_insert_delete_only: bool = False,
        batch_size: int = 256
) -> List[int]:
    """
    For each a, b pair in a_list, b_list calculates the minimum number of edit operations required to turn a into b.
    Uses a high performance implementation in Rust under the hood.

    Args:
        a_list: list of source strings
        b_list: list of destination strings
        with_swap: allow swap/transposition operations
        spaces_insert_delete_only: make sure that spaces can only be inserted or deleted
        batch_size: number of strings per batch

    Returns: edit distance

    """
    return edit_distance_rs.batch_edit_distance(a_list, b_list, with_swap, spaces_insert_delete_only, batch_size)


def match_words(a: str, b: str) -> List[Tuple[int, int]]:
    return edit_distance_rs.match_words(a, b)


def batch_match_words(a_list: List[str], b_list: List[str], batch_size: int = 256) -> List[List[Tuple[int, int]]]:
    return edit_distance_rs.batch_match_words(a_list, b_list, batch_size)


def find_word_boundaries(s: str) -> List[Tuple[int, int]]:
    word_boundaries = []
    start_idx = 0
    for word in s.split():
        word_boundaries.append((start_idx, start_idx + len(word)))
        start_idx += len(word) + 1
    return word_boundaries


def get_edited_words(ipts: List[str], tgts: List[str]) -> Tuple[List[Set[int]], List[Set[int]]]:
    edited_in_ipts = []
    edited_in_tgts = []
    batch_edit_ops = batch_edit_operations(ipts, tgts, spaces_insert_delete_only=True)
    for ipt, tgt, edit_ops in zip(ipts, tgts, batch_edit_ops):
        ipt_word_boundaries = find_word_boundaries(ipt)
        tgt_word_boundaries = find_word_boundaries(tgt)
        edited_ipt_indices = set()
        edited_tgt_indices = set()
        for op_code, ipt_idx, tgt_idx in edit_ops:
            ipt_word_idx = 0
            for wb_s, wb_e in ipt_word_boundaries:
                if wb_s <= ipt_idx < wb_e:
                    edited_ipt_indices.add(ipt_word_idx)
                    break
                elif ipt_idx == wb_e:
                    import edit_distance_rs
                    graphemes = edit_distance_rs.get_graphemes(tgt[tgt_idx - 15: tgt_idx + 16])
                    assert op_code == "delete" or \
                           op_code == "insert", \
                        (ipt[ipt_idx - 15: ipt_idx + 16],
                         tgt[tgt_idx - 15: tgt_idx + 16],
                         list(tgt[tgt_idx - 15: tgt_idx + 16]),
                         graphemes,
                         "".join(graphemes),
                         list("".join(graphemes)))
                    if op_code == "delete":
                        assert ipt[ipt_idx] == " "
                        edited_ipt_indices.add(ipt_word_idx)
                        edited_ipt_indices.add(ipt_word_idx + 1)
                    else:
                        edited_ipt_indices.add(ipt_word_idx)
                    break
                ipt_word_idx += 1
            assert ipt_word_idx < len(ipt_word_boundaries)
            tgt_word_idx = 0
            for wb_s, wb_e in tgt_word_boundaries:
                if wb_s <= tgt_idx < wb_e:
                    edited_tgt_indices.add(tgt_word_idx)
                    break
                elif tgt_idx == wb_e:
                    assert op_code == "delete" or op_code == "insert", (op_code, ipt[ipt_idx], tgt[tgt_idx])
                    if op_code == "delete":
                        edited_tgt_indices.add(tgt_word_idx)
                    else:
                        assert tgt[tgt_idx] == " "
                        edited_tgt_indices.add(tgt_word_idx)
                        edited_tgt_indices.add(tgt_word_idx + 1)
                    break
                tgt_word_idx += 1
            assert tgt_word_idx < len(tgt_word_boundaries)

        edited_in_ipts.append(edited_ipt_indices)
        edited_in_tgts.append(edited_tgt_indices)

    return edited_in_ipts, edited_in_tgts
