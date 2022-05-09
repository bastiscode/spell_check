import pytest

from nsc.utils import edit

_TEST_IPT_SEQUENCES = [
    "th s is a test",
    "we do no match",
    "Just a rwong sequence",
    "one last examples"
]
_TEST_TARGET_SEQUENCES = [
    "this is a test",
    "we do not match",
    "just a wrong sequence",
    "one last example"
]


class TestEdit:
    @pytest.mark.parametrize("with_swap", [True, False])
    @pytest.mark.parametrize("spaces_insert_delete_only", [True, False])
    def test_edit_distance(self, with_swap: bool, spaces_insert_delete_only: bool) -> None:
        dists = edit.batch_edit_distance(_TEST_IPT_SEQUENCES, _TEST_TARGET_SEQUENCES,
                                         with_swap, spaces_insert_delete_only)
        if with_swap:
            if spaces_insert_delete_only:
                assert dists == [2, 1, 2, 1]
            else:
                assert dists == [1, 1, 2, 1]
        else:
            if spaces_insert_delete_only:
                assert dists == [2, 1, 3, 1]
            else:
                assert dists == [1, 1, 3, 1]

    def test_match_words(self) -> None:
        matching_words = edit.batch_match_words(_TEST_IPT_SEQUENCES, _TEST_TARGET_SEQUENCES)
        assert matching_words == [
            [(2, 1), (3, 2), (4, 3)], [(0, 0), (1, 1), (3, 3)], [(1, 1), (3, 3)], [(0, 0), (1, 1)]
        ]

    def test_edited_words(self) -> None:
        edited_in_ipt, edited_in_tgt = edit.get_edited_words(_TEST_IPT_SEQUENCES, _TEST_TARGET_SEQUENCES)
        assert edited_in_ipt == [{0, 1}, {2}, {0, 2}, {2}]
        assert edited_in_tgt == [{0}, {2}, {0, 2}, {2}]
