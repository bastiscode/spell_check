import pytest

from nsc.utils import metrics

_TEST_INPUT_SEQUENCES = [
    "this is a tset",
    "we do nod match",
    "just a rwong sequence",
    "one last examples"
]
_TEST_PRED_SEQUENCES = [
    "this is a test",
    "we do no match",
    "Just a wrong sequence",
    "one last examples"
]
_TEST_TARGET_SEQUENCES = [
    "this is a test",
    "we do not match",
    "just a wrong sequence",
    "one last example"
]
_TR_TEST_INPUT_SEQUENCES = [
    "thisisatest",
    "we do not ma tch",
    "just awrong seq uence",
    "o n e l a s t e x a m p l e"
]
_TR_TEST_PRED_SEQUENCES = [
    "this is a test",
    "we do not match",
    "justa wrong sequence",
    "onelast example"
]
_TR_TEST_TARGET_SEQUENCES = [
    "this is a test",
    "we do not match",
    "just a wrong sequence",
    "one last example"
]


class TestMetrics:
    def test_accuracy(self) -> None:
        accuracy = metrics.accuracy(_TEST_PRED_SEQUENCES, _TEST_TARGET_SEQUENCES)
        assert accuracy == 0.25

    def test_mned(self) -> None:
        mned = metrics.mean_normalized_sequence_edit_distance(_TEST_PRED_SEQUENCES, _TEST_TARGET_SEQUENCES)
        assert mned == (1 / 15 + 1 / 21 + 1 / 17) / 4

    def test_correction_f1(self) -> None:
        # this test case is taken from Matthias Hertel's masters thesis
        # https://ad-publications.cs.uni-freiburg.de/theses/Master_Matthias_Hertel_2019.pdf
        ipt = "Te cute cteats delicious fi sh."
        tgt = "The cute cat eats delicious fish."
        pred = "The cute act eats delicate fi sh."

        (f1, prec, rec), _ = metrics.correction_f1_prec_rec([ipt], [pred], [tgt])
        assert prec == 0.5 and rec == 0.5 and f1 == 0.5

    def test_binary_f1(self) -> None:
        binary_preds = [int(p != i)
                        for pred, ipt in zip(_TEST_PRED_SEQUENCES, _TEST_INPUT_SEQUENCES)
                        for p, i in zip(pred.split(), ipt.split())]
        binary_tgt = [int(t != i)
                      for tgt, ipt in zip(_TEST_TARGET_SEQUENCES, _TEST_INPUT_SEQUENCES)
                      for t, i in zip(tgt.split(), ipt.split())]
        f1, prec, rec = metrics.binary_f1_prec_rec(binary_preds, binary_tgt)
        assert prec == 3 / 4 and rec == 3 / 4 and f1 == 3 / 4

    def test_tok_rep_f1(self) -> None:
        (f1, prec, rec), (avg_f1, avg_prec, avg_rec) = metrics.tok_rep_f1_prec_rec(
            _TR_TEST_INPUT_SEQUENCES, _TR_TEST_PRED_SEQUENCES, _TR_TEST_TARGET_SEQUENCES
        )
        assert (
                prec == pytest.approx((3 + 1 + 2 + 11) / (3 + 1 + 3 + 12))
                and rec == 1.
                and f1 == pytest.approx(0.9444, abs=0.0001)
        )
        assert (
                avg_prec == pytest.approx(0.9379, abs=0.0001)
                and avg_rec == 1.
                and avg_f1 == pytest.approx(0.9668, abs=0.0001)
        )
