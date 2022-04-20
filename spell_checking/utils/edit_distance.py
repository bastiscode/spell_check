from typing import Union, List, Tuple


def edit_operations(
        a: str,
        b: str,
        with_swap: bool = False,
        spaces_insert_delete_only: bool = False,
        return_distance_only: bool = False
) -> Union[int, List[Tuple[str, int, int]]]:
    """
    Returns the edit operations transforming a into b, optionally allowing swap/transposition operations
    and restricting operations on spaces to insertions and deletions.
    Follows optimal string alignment distance algorithm
    from https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance.

    Args:
        a: source string
        b: destination string
        with_swap: allow swap/transposition operations
        spaces_insert_delete_only: make sure that spaces can only be inserted or deleted
        return_distance_only: whether to return the edit distance only and not all edit ops
            (saves computing the backtrace)

    Returns: edit distance if return_distance_only else list of edit operations

    """
    d = [
        list(range(len(b) + 1)) if i == 0 else [i if j == 0 else -1 for j in range(len(b) + 1)]
        for i in range(len(a) + 1)
    ]
    # operations: ' ' --> not yet filled in, k --> keep, i --> insert, d --> delete, r --> replace, s --> swap
    ops = [
        ["i"] * (len(b) + 1) if i == 0 else ["d" if j == 0 else " " for j in range(len(b) + 1)]
        for i in range(len(a) + 1)
    ]
    ops[0][0] = "k"

    # fill in matrices
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            # string indices are offset by -1
            i_str = i - 1
            j_str = j - 1

            # delete and insert
            costs = [(d[i - 1][j] + 1, "d"), (d[i][j - 1] + 1, "i")]
            if a[i_str] == b[j_str]:
                costs.append((d[i - 1][j - 1], "k"))
            else:
                # chars are not equal, only allow replacement if no space is involved,
                # or we are allowed to replace spaces
                if (
                        not spaces_insert_delete_only
                        or (a[i_str] != " " and b[j_str] != " ")
                ):
                    costs.append((d[i - 1][j - 1] + 1, "r"))
            # check if we can swap chars, that is if we are allowed to swap and if the chars to swap match
            if with_swap and i > 1 and j > 1 and a[i_str] == b[j_str - 1] and a[i_str - 1] == b[j_str]:
                # we can swap the chars, but only allow swapping if no space is involved,
                # or we are allowed to swap spaces
                if (
                        not spaces_insert_delete_only
                        or (a[i_str] != " " and a[i_str - 1] != " ")
                ):
                    costs.append((d[i - 2][j - 2] + 1, "s"))

            min_cost, min_op = min(costs, key=lambda item: item[0])
            d[i][j] = min_cost
            ops[i][j] = min_op

    # make sure that it worked
    assert all(v >= 0 for row in d for v in row)

    if return_distance_only:
        return d[-1][-1]

    # backtrace matrices
    edit_ops = []
    i = len(a)
    j = len(b)
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == "k":
            # we do not add keep operation to edit_ops
            i -= 1
            j -= 1
            continue

        if op == "d":
            op_name = "delete"
            i -= 1
        elif op == "i":
            op_name = "insert"
            j -= 1
        elif op == "r":
            op_name = "replace"
            i -= 1
            j -= 1
        elif op == "s":
            op_name = "swap"
            i -= 2
            j -= 2
        else:
            raise RuntimeError("should not happen")

        edit_ops.append((op_name, i, j))

    return list(reversed(edit_ops))


def edit_distance(
        a: str,
        b: str,
        with_swap: bool = True,
        spaces_insert_delete_only: bool = False
) -> int:
    return edit_operations(a, b, with_swap, spaces_insert_delete_only, return_distance_only=True)
