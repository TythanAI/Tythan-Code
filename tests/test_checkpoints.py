"""CheckpointStore tests. Every store here is pointed at tmp_path via
storage_dir= so nothing ever touches the real ~/.tythancode directory."""

from tythancode.checkpoints import CheckpointStore, FileChange


def make_store(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store_dir = tmp_path / "checkpoint_storage"
    return CheckpointStore(workspace, storage_dir=store_dir), workspace, store_dir


def test_no_directory_created_until_a_change_is_committed(tmp_path):
    store, workspace, store_dir = make_store(tmp_path)
    assert not store_dir.exists()
    store.begin_turn("did nothing")
    assert store.commit_turn() is None
    assert not store_dir.exists()


def test_record_and_commit_creates_checkpoint(tmp_path):
    store, workspace, store_dir = make_store(tmp_path)
    target = workspace / "a.txt"
    target.write_text("v1")

    store.begin_turn("edit a.txt")
    store.record_before(target)
    target.write_text("v2")
    cp = store.commit_turn()

    assert cp is not None
    assert cp.label == "edit a.txt"
    assert len(cp.changes) == 1
    assert cp.changes[0].path == str(target)
    assert cp.changes[0].existed_before is True
    assert cp.changes[0].before_content == "v1"
    assert store_dir.exists()


def test_record_before_new_file_marks_existed_before_false(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "new.txt"

    store.begin_turn("create new.txt")
    store.record_before(target)  # file doesn't exist yet
    target.write_text("content")
    cp = store.commit_turn()

    assert cp.changes[0].existed_before is False
    assert cp.changes[0].before_content is None


def test_record_before_keeps_first_state_when_touched_twice(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "a.txt"
    target.write_text("v1")

    store.begin_turn("double edit")
    store.record_before(target)
    target.write_text("v2")
    store.record_before(target)  # same path again — should be ignored
    target.write_text("v3")
    cp = store.commit_turn()

    assert len(cp.changes) == 1
    assert cp.changes[0].before_content == "v1"


def test_record_before_without_begin_turn_is_a_noop(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "a.txt"
    target.write_text("v1")
    store.record_before(target)  # no begin_turn() called
    assert store.commit_turn() is None


def test_large_file_is_skipped_but_noted(tmp_path, monkeypatch):
    import tythancode.checkpoints as checkpoints_mod

    monkeypatch.setattr(checkpoints_mod, "MAX_CHECKPOINT_FILE_BYTES", 10)
    store, workspace, store_dir = make_store(tmp_path)
    target = workspace / "big.txt"
    target.write_text("x" * 100)

    store.begin_turn("touch big file")
    store.record_before(target)
    cp = store.commit_turn()
    # Nothing was actually written/undoable, but the caller must still be able
    # to tell the user this file wasn't covered — commit_turn() must not
    # silently discard that by returning None.
    assert cp is not None
    assert cp.changes == []
    assert cp.skipped_large == [str(target)]
    # And since there's nothing undoable, nothing is persisted to disk.
    assert not store_dir.exists()
    assert store.list() == []


def test_large_file_skip_reported_alongside_other_changes(tmp_path, monkeypatch):
    import tythancode.checkpoints as checkpoints_mod

    monkeypatch.setattr(checkpoints_mod, "MAX_CHECKPOINT_FILE_BYTES", 10)
    store, workspace, _ = make_store(tmp_path)
    big = workspace / "big.txt"
    big.write_text("x" * 100)
    small = workspace / "small.txt"
    small.write_text("v1")

    store.begin_turn("touch both")
    store.record_before(big)
    store.record_before(small)
    cp = store.commit_turn()

    assert len(cp.changes) == 1
    assert cp.changes[0].path == str(small)
    assert cp.skipped_large == [str(big)]


def test_undo_last_restores_modified_file(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "a.txt"
    target.write_text("v1")

    store.begin_turn("edit")
    store.record_before(target)
    target.write_text("v2")
    store.commit_turn()

    restored = store.undo_last()
    assert restored is not None
    assert target.read_text() == "v1"


def test_undo_last_deletes_file_created_this_turn(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "new.txt"

    store.begin_turn("create")
    store.record_before(target)
    target.write_text("content")
    store.commit_turn()

    store.undo_last()
    assert not target.exists()


def test_undo_last_on_empty_store_returns_none(tmp_path):
    store, _, _ = make_store(tmp_path)
    assert store.undo_last() is None


def test_record_before_skips_directories(tmp_path):
    """write_file/edit_file targeting an existing directory must never produce
    a checkpoint entry — existed_before would wrongly read as False (since
    is_file() is False for a directory too), and /undo would later try to
    unlink() a directory path and crash."""
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "a_directory"
    target.mkdir()

    store.begin_turn("oops, wrote to a directory")
    store.record_before(target)
    cp = store.commit_turn()

    assert cp is None
    assert target.is_dir()  # untouched


def test_record_before_skips_non_utf8_files_instead_of_corrupting_them(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "legacy.txt"
    target.write_bytes(b"caf\xe9")  # not valid UTF-8 (Latin-1 "café")

    store.begin_turn("touch legacy file")
    store.record_before(target)
    cp = store.commit_turn()

    assert cp is not None
    assert cp.changes == []
    assert cp.skipped_binary == [str(target)]
    # The original bytes must be completely untouched by the attempted checkpoint.
    assert target.read_bytes() == b"caf\xe9"


def test_undo_ignores_a_directory_left_by_a_pre_fix_checkpoint(tmp_path):
    """Defense in depth: even if a bogus existed_before=False entry for a
    directory somehow ends up on disk (e.g. from an older Tythan Code
    version), undo_last() must not crash trying to unlink() it."""
    store, workspace, _ = make_store(tmp_path)
    a_directory = workspace / "a_directory"
    a_directory.mkdir()

    store.begin_turn("bogus entry")
    store._current.changes.append(
        FileChange(path=str(a_directory), existed_before=False, before_content=None)
    )
    store.commit_turn()

    restored = store.undo_last()  # must not raise
    assert restored is not None
    assert a_directory.is_dir()  # left alone, not crashed on


def test_list_limit_zero_returns_nothing(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "a.txt"
    target.write_text("v1")
    store.begin_turn("edit")
    store.record_before(target)
    target.write_text("v2")
    store.commit_turn()

    assert store.list(limit=0) == []
    assert store.list(limit=10) != []


def test_count_reports_total_regardless_of_list_limit(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "a.txt"
    target.write_text("v0")
    for i in range(5):
        store.begin_turn(f"edit {i}")
        store.record_before(target)
        target.write_text(f"v{i + 1}")
        store.commit_turn()

    assert store.count() == 5
    assert len(store.list(limit=2)) == 2


def test_undo_pops_one_checkpoint_at_a_time(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "a.txt"
    target.write_text("v1")

    store.begin_turn("first edit")
    store.record_before(target)
    target.write_text("v2")
    store.commit_turn()

    store.begin_turn("second edit")
    store.record_before(target)
    target.write_text("v3")
    store.commit_turn()

    assert target.read_text() == "v3"
    store.undo_last()
    assert target.read_text() == "v2"
    store.undo_last()
    assert target.read_text() == "v1"
    assert store.undo_last() is None  # nothing left


def test_undo_refuses_paths_outside_its_own_workspace(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("safe")

    # Simulate a corrupted/tampered checkpoint file pointing outside the workspace.
    store.begin_turn("malicious")
    store._current.changes.append(
        FileChange(path=str(outside), existed_before=True, before_content="HACKED")
    )
    store.commit_turn()

    store.undo_last()
    assert outside.read_text() == "safe"  # untouched


def test_undo_refuses_a_symlink_pointing_outside_the_workspace(tmp_path):
    """Even if a path was a plain in-workspace file when checkpointed, undo_last()
    must re-check containment against the live filesystem at undo time, not just
    the recorded path string — otherwise a symlink swapped in later (e.g. by an
    agent-run shell command, which isn't covered by checkpoints at all) could
    make /undo write outside the workspace."""
    store, workspace, _ = make_store(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("safe")
    link_path = workspace / "link.txt"

    store.begin_turn("edit via what was a real path")
    store._current.changes.append(
        FileChange(path=str(link_path), existed_before=True, before_content="HACKED")
    )
    store.commit_turn()

    # Simulate the path now being a symlink to something outside the workspace,
    # as if it were swapped after the checkpoint was recorded.
    link_path.symlink_to(outside)

    store.undo_last()
    assert outside.read_text() == "safe"  # untouched despite the symlink


def test_list_returns_most_recent_first(tmp_path):
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "a.txt"
    target.write_text("v1")

    for i in range(3):
        store.begin_turn(f"edit {i}")
        store.record_before(target)
        target.write_text(f"v{i + 2}")
        store.commit_turn()

    labels = [cp.label for cp in store.list()]
    assert labels == ["edit 2", "edit 1", "edit 0"]


def test_retention_prunes_oldest_checkpoints(tmp_path, monkeypatch):
    import tythancode.checkpoints as checkpoints_mod

    monkeypatch.setattr(checkpoints_mod, "MAX_CHECKPOINTS_PER_WORKSPACE", 3)
    store, workspace, _ = make_store(tmp_path)
    target = workspace / "a.txt"
    target.write_text("v0")

    for i in range(5):
        store.begin_turn(f"edit {i}")
        store.record_before(target)
        target.write_text(f"v{i + 1}")
        store.commit_turn()

    assert len(store._files()) == 3
    labels = [cp.label for cp in store.list(limit=10)]
    assert labels == ["edit 4", "edit 3", "edit 2"]


def test_persists_across_store_instances(tmp_path):
    """/undo must survive restarting Tythan Code — checkpoints are files on disk."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store_dir = tmp_path / "checkpoint_storage"

    store1 = CheckpointStore(workspace, storage_dir=store_dir)
    target = workspace / "a.txt"
    target.write_text("v1")
    store1.begin_turn("edit")
    store1.record_before(target)
    target.write_text("v2")
    store1.commit_turn()

    # Fresh instance, same storage_dir — simulates a new process starting up.
    store2 = CheckpointStore(workspace, storage_dir=store_dir)
    restored = store2.undo_last()
    assert restored is not None
    assert target.read_text() == "v1"
