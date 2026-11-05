"""Tests for evaluate.py — reference parsing, alignment, and passage scoring."""

from __future__ import annotations

import json

import pytest

from migs_tab.evaluate import (
    Passage,
    align_sequences,
    cluster_notes,
    load_passages,
    parse_ascii_tab,
    parse_token,
    score_passage,
)

OPEN = (40, 45, 50, 55, 59, 64)


class TestParsing:
    def test_token(self):
        assert parse_token("A2") == (1, 2)
        assert parse_token("e0") == (5, 0)
        assert parse_token("E12") == (0, 12)
        with pytest.raises(ValueError):
            parse_token("H3")

    def test_ascii_tab_columns_and_two_digit_frets(self):
        cols = parse_ascii_tab(
            [
                "e|-0-----12--|",
                "B|-1---------|",
                "G|---2p0-----|",
                "D|-----------|",
            ]
        )
        assert cols == [{(5, 0), (4, 1)}, {(3, 2)}, {(3, 0)}, {(5, 12)}]

    def test_load_passages_merges_sequence_into_vocabulary(self, tmp_path):
        (tmp_path / "vid.json").write_text(
            json.dumps(
                {
                    "passages": [
                        {
                            "label": "x",
                            "start": 1.0,
                            "end": 2.0,
                            "vocabulary": ["A0"],
                            "sequence": ["D|-2-4-|"],
                        }
                    ]
                }
            )
        )
        (p,) = load_passages("vid", tmp_path)
        assert p.vocabulary == {(1, 0), (2, 2), (2, 4)}
        assert p.sequence == [{(2, 2)}, {(2, 4)}]


class TestAlignment:
    def test_exact_match(self):
        seq = [{(1, 0)}, {(2, 2), (3, 2)}, {(2, 4)}]
        assert align_sequences(seq, seq) == (4, 4, 4)

    def test_insertion_and_substitution(self):
        ref = [{(1, 0)}, {(2, 2)}, {(2, 4)}]
        hyp = [{(1, 0)}, {(5, 7)}, {(2, 2)}, {(2, 4)}]  # one extra wrong chord
        matched, nref, nhyp = align_sequences(ref, hyp)
        assert (matched, nref, nhyp) == (3, 3, 4)

    def test_cluster_notes_groups_simultaneous_onsets(self):
        notes = [
            {"start": 1.00, "string": 1, "fret": 0},
            {"start": 1.02, "string": 2, "fret": 2},
            {"start": 1.50, "string": 2, "fret": 4},
        ]
        assert cluster_notes(notes) == [{(1, 0), (2, 2)}, {(2, 4)}]


class TestScorePassage:
    def _notes(self):
        # Open E power chord played right (E0, A2) plus one note pushed up the neck
        # (B3 on the G string at fret 4 instead of the open B string) and one phantom.
        return [
            {"start": 10.0, "pitch": 40, "string": 0, "fret": 0},
            {"start": 10.0, "pitch": 47, "string": 1, "fret": 2},
            {"start": 10.5, "pitch": 59, "string": 3, "fret": 4},
            {"start": 11.0, "pitch": 68, "string": 5, "fret": 4},
            {"start": 30.0, "pitch": 40, "string": 0, "fret": 0},  # outside the window
        ]

    def test_metrics(self):
        passage = Passage("E", 9.0, 20.0, {(0, 0), (1, 2), (1, 4), (4, 0)}, None)
        r = score_passage(passage, self._notes(), OPEN)
        assert r.note_count == 4
        assert r.position_precision == pytest.approx(0.5)
        # B3 (pitch 59) is in the vocabulary's pitch set via the open B string.
        assert r.pitch_precision == pytest.approx(0.75)
        assert r.vocab_recall == pytest.approx(0.5)
        assert r.high_fret_strays == 0  # max vocab fret 4, strays start above 6
        wrong = r.details["most_common_wrong_positions"]
        assert {(w["string"], w["fret"]) for w in wrong} == {(3, 4), (5, 4)}

    def test_sequence_f1(self):
        passage = Passage("riff", 9.0, 20.0, set(), [{(0, 0), (1, 2)}, {(3, 4)}])
        r = score_passage(passage, self._notes()[:3], OPEN)
        assert r.sequence_f1 == pytest.approx(1.0)

    def test_empty_window(self):
        passage = Passage("none", 100.0, 101.0, {(0, 0)}, None)
        r = score_passage(passage, self._notes(), OPEN)
        assert r.note_count == 0 and r.position_precision is None
