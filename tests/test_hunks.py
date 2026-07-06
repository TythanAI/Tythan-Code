import pytest

from tythancode.hunks import apply_selected_hunks, split_hunks


def test_single_line_replace_is_one_hunk():
    old = "a\nb\nc\n"
    new = "a\nX\nc\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 1
    assert hunks[0].tag == "replace"
    assert hunks[0].old_lines == ["b\n"]
    assert hunks[0].new_lines == ["X\n"]
    assert hunks[0].old_start == 2
    assert hunks[0].new_start == 2


def test_two_separate_edits_are_two_hunks():
    old = "a\nb\nc\nd\ne\n"
    new = "A\nb\nc\nD\ne\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 2
    assert hunks[0].old_lines == ["a\n"]
    assert hunks[1].old_lines == ["d\n"]


def test_pure_insertion_hunk():
    old = "a\nb\n"
    new = "a\nNEW\nb\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 1
    assert hunks[0].tag == "insert"
    assert hunks[0].old_lines == []
    assert hunks[0].new_lines == ["NEW\n"]


def test_pure_deletion_hunk():
    old = "a\nb\nc\n"
    new = "a\nc\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 1
    assert hunks[0].tag == "delete"
    assert hunks[0].old_lines == ["b\n"]
    assert hunks[0].new_lines == []


def test_identical_text_has_no_hunks():
    assert split_hunks("same\n", "same\n") == []


def test_apply_all_accepted_equals_new_text():
    old = "a\nb\nc\nd\ne\n"
    new = "A\nb\nc\nD\ne\n"
    hunks = split_hunks(old, new)
    result = apply_selected_hunks(old, new, [True] * len(hunks))
    assert result == new


def test_apply_all_rejected_equals_old_text():
    old = "a\nb\nc\nd\ne\n"
    new = "A\nb\nc\nD\ne\n"
    hunks = split_hunks(old, new)
    result = apply_selected_hunks(old, new, [False] * len(hunks))
    assert result == old


def test_apply_partial_acceptance():
    old = "a\nb\nc\nd\ne\n"
    new = "A\nb\nc\nD\ne\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 2
    # accept only the second hunk (d -> D), reject the first (a -> A)
    result = apply_selected_hunks(old, new, [False, True])
    assert result == "a\nb\nc\nD\ne\n"


def test_apply_partial_acceptance_other_order():
    old = "a\nb\nc\nd\ne\n"
    new = "A\nb\nc\nD\ne\n"
    hunks = split_hunks(old, new)
    result = apply_selected_hunks(old, new, [True, False])
    assert result == "A\nb\nc\nd\ne\n"


def test_apply_rejects_wrong_length_accepted_list():
    old, new = "a\nb\n", "A\nB\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 1
    with pytest.raises(ValueError):
        apply_selected_hunks(old, new, [])
    with pytest.raises(ValueError):
        apply_selected_hunks(old, new, [True, True])


def test_no_trailing_newline_handled():
    old = "a\nb"
    new = "a\nB"
    hunks = split_hunks(old, new)
    assert len(hunks) == 1
    result = apply_selected_hunks(old, new, [True])
    assert result == new


def test_new_file_from_empty_old_is_one_insert_hunk():
    old = ""
    new = "line1\nline2\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 1
    assert hunks[0].tag == "insert"
    result = apply_selected_hunks(old, new, [True])
    assert result == new
    result_rejected = apply_selected_hunks(old, new, [False])
    assert result_rejected == old


def test_many_scattered_edits_roundtrip():
    old_lines = [f"line{i}\n" for i in range(50)]
    new_lines = list(old_lines)
    changed_indices = [3, 10, 25, 40]
    for i in changed_indices:
        new_lines[i] = f"CHANGED{i}\n"
    old, new = "".join(old_lines), "".join(new_lines)

    hunks = split_hunks(old, new)
    assert len(hunks) == len(changed_indices)

    # accept every other hunk
    accepted = [i % 2 == 0 for i in range(len(hunks))]
    result = apply_selected_hunks(old, new, accepted)
    result_lines = result.splitlines(keepends=True)
    for idx, changed_idx in enumerate(changed_indices):
        if accepted[idx]:
            assert result_lines[changed_idx] == f"CHANGED{changed_idx}\n"
        else:
            assert result_lines[changed_idx] == f"line{changed_idx}\n"
