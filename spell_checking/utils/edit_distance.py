from typing import List, Tuple

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
