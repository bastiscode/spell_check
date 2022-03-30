import re
from enum import Enum
from typing import List


class TokenizationRepairTokens(Enum):
    KEEP_CHAR = "#"  # "<k>"
    INSERT_WS = "_"  # "<iw>"
    DELETE_WS = "x"  # "<dw>"


def get_repair_tokens(repair_sequence: str) -> List[str]:
    return re.findall(
        f"({TokenizationRepairTokens.DELETE_WS.value}"
        f"|{TokenizationRepairTokens.INSERT_WS.value}"
        f"|{TokenizationRepairTokens.KEEP_CHAR.value})",
        repair_sequence
    )


def remove_whitespace(sequence: str) -> str:
    return re.sub(r"\s", "", sequence)


def get_whitespace_operations(
        from_sequence: str,
        to_sequence: str
) -> List[int]:
    """

    Get the repair sequence that turns from_sequence into to_sequence
    (after applying the repair_whitespace function)

    :param from_sequence: sequence that the returned repair tokens
        should be applied to to get the to_sequence
    :param to_sequence: sequence that should result from applying
        the whitespace operations to the from_sequence
    :return: list of repair tokens
    """
    assert from_sequence.replace(" ", "") == to_sequence.replace(" ", ""), \
        f"make sure from_sequence and to_sequence only differ in whitespaces:\n{from_sequence}\n{to_sequence}"

    from_sequence_ptr = 0
    to_sequence_ptr = 0

    repair_tokens = []

    while from_sequence_ptr < len(from_sequence):  # and to_sequence_ptr < len(to_sequence):
        from_char = from_sequence[from_sequence_ptr]
        to_char = to_sequence[to_sequence_ptr] if to_sequence_ptr < len(to_sequence) else ""

        if from_char == to_char:
            repair_tokens.append(0)
            from_sequence_ptr += 1
            to_sequence_ptr += 1

        elif to_char == " ":
            repair_tokens.append(1)
            from_sequence_ptr += 1
            to_sequence_ptr += 2

        elif from_char == " ":
            repair_tokens.append(2)
            from_sequence_ptr += 1

        else:
            raise ValueError("should not happen")

    assert len(repair_tokens) == len(from_sequence), \
        f"{''.join(str(r) for r in repair_tokens)}\n'{from_sequence}'\n'{to_sequence}'"

    return repair_tokens


def repair_whitespace(
        sequence: str,
        repair_tokens: List[int]) -> str:
    """
    Repair the white spacing in the given sequence
    using the given repair tokens.

    :param sequence: string which has to be repaired
    :param repair_tokens: list with 0's, 1's and 2's
        indicating to keep the char, insert a whitespace or
        delete a whitespace.
    :return: repaired string
    """
    if len(sequence) > len(repair_tokens):
        repair_tokens.extend([0] * (len(sequence) - len(repair_tokens)))
    else:
        repair_tokens = repair_tokens[:len(sequence)]

    allowed_tokens = {0, 1, 2}
    assert all([token in allowed_tokens for token in repair_tokens]), \
        f"Only 0's, 1's and 2's are allowed as repair tokens, " \
        f"but got {repair_tokens} for sequence \"{sequence}\""

    sequence_ptr = 0
    token_ptr = 0

    repaired_sequence = ""
    while sequence_ptr < len(sequence):
        char = sequence[sequence_ptr]
        prev_char = sequence[sequence_ptr - 1] if sequence_ptr > 0 else " "
        token = repair_tokens[token_ptr]

        if token == 1 and char != " " and prev_char != " ":
            # if we should insert a whitespace and the current
            # and previous character are not whitespaces,
            # add a whitespace in front of the character
            repaired_sequence += " " + char

        elif token == 2 and char == " ":
            # if we should delete a whitespace and
            # we are at a whitespace, just skip
            pass

        else:
            # keep current character in all other cases
            repaired_sequence += char

        sequence_ptr += 1
        token_ptr += 1

    assert (sequence_ptr == len(sequence) and token_ptr == len(repair_tokens))

    return repaired_sequence


def repair_whitespace_from_groups(
        sequence: str,
        char_repair_tokens: List[int],
        whitespace_repair_tokens: List[int]
) -> str:
    """
    Repair the white spacing in the given sequence
    using the given character and whitespace repair tokens.

    :param sequence: string which has to be repaired
    :param char_repair_tokens: list with 0's and 1's
        indicating to keep the char or insert a whitespace
    :param whitespace_repair_tokens: list with 0's and 1's
        indicating to keep the or delete a whitespace
    :return: repaired string
    """
    assert len(char_repair_tokens) + len(whitespace_repair_tokens) == len(sequence), \
        f"the number of whitespace and character repair tokens together must be " \
        f"equal to the length of the sequence, but got {len(char_repair_tokens)} + {len(whitespace_repair_tokens)}" \
        f" = {len(sequence)} for sequence '{sequence}'"

    allowed_tokens = {0, 1}
    assert (
            all([token in allowed_tokens for token in char_repair_tokens])
            and all([token in allowed_tokens for token in whitespace_repair_tokens])
    ), f"Only 0's and 1's are allowed as repair tokens, " \
       f"but got {char_repair_tokens} and {whitespace_repair_tokens} for sequence \"{sequence}\""

    sequence_ptr = 0
    char_token_ptr = 0
    whitespace_token_ptr = 0

    repaired_sequence = ""
    while sequence_ptr < len(sequence):
        char = sequence[sequence_ptr]
        prev_char = sequence[sequence_ptr - 1] if sequence_ptr > 0 else " "

        if char == " ":
            token = whitespace_repair_tokens[whitespace_token_ptr]
            whitespace_token_ptr += 1
        else:
            token = char_repair_tokens[char_token_ptr]
            char_token_ptr += 1

        if token == 1 and char != " " and prev_char != " ":
            # if we should insert a whitespace and the current
            # and previous character are not whitespaces,
            # add a whitespace in front of the character
            repaired_sequence += " " + char

        elif token == 1 and char == " ":
            # if we should delete a whitespace and
            # we are at a whitespace, just skip
            pass

        else:
            # keep current character in all other cases
            repaired_sequence += char

        sequence_ptr += 1

    assert (sequence_ptr == len(sequence)
            and char_token_ptr == len(char_repair_tokens)
            and whitespace_token_ptr == len(whitespace_repair_tokens))

    return repaired_sequence
