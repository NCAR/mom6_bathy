import os
import numpy as np
import pytest
import git
import json

from mom6_bathy.grid import Grid
from mom6_bathy.topo import Topo
from mom6_bathy.edit_command import DepthEditCommand
from mom6_bathy.topo_editor import TopoEditor


@pytest.fixture
def minimal_grid_and_topo():
    """Test that a minimal 5x5 grid can be set up for the Panama region"""
    grid = Grid(
        resolution=0.1, xstart=278.0, lenx=0.5, ystart=7.0, leny=0.5, name="testpanama"
    )
    topo = Topo(grid=grid, min_depth=10.0)
    topo.set_flat(100.0)
    return topo


class Dummy:
    disabled = False
    value = None

    def __init__(self, value=None):
        if value is not None:
            self.value = value

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


class PatchedTopoEditor(TopoEditor):
    """
    A test-only subclass of TopoEditor that enforces the use of a provided snapshot_dir
    and disables UI construction by default. This ensures all file operations are
    isolated to the test environment and no interactive widgets are created.
    """

    def __init__(self, topo, build_ui=False, snapshot_dir=None):
        assert snapshot_dir is not None, "snapshot_dir must be provided for tests"
        super().__init__(topo, build_ui=build_ui, snapshot_dir=snapshot_dir)


def patch_all_widgets(editor):
    dummy_attrs = [
        "_undo_button",
        "_redo_button",
        "_reset_button",
        "_min_depth_specifier",
        "_depth_specifier",
        "_selected_cell_label",
        "_basin_specifier",
        "_basin_specifier_toggle",
        "_basin_specifier_delete_selected",
        "_snapshot_name",
        "_save_button",
        "_load_button",
        "_display_mode_toggle",
        "im",
        "cbar",
        "ax",
        "fig",
    ]
    for attr in dummy_attrs:
        value = "depth" if attr == "_display_mode_toggle" else None
        setattr(editor, attr, Dummy(value=value))


def setup_depth(editor, i=2, j=2, new_depth=777.0):
    orig_depth = float(editor.topo.depth.data[j, i])
    editor._select_cell(i, j)
    edit = DepthEditCommand(
        editor.topo,
        affected_indices=[(j, i)],
        new_values=[new_depth],
        old_values=[orig_depth],
    )
    editor.apply_edit(edit)
    return orig_depth, new_depth, i, j


def test_undo(minimal_grid_and_topo, tmp_path):
    topo = minimal_grid_and_topo
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)

    orig_depth, new_depth, i, j = setup_depth(editor)
    assert float(topo.depth.data[j, i]) == new_depth

    editor.undo_last_edit()
    assert float(topo.depth.data[j, i]) == orig_depth
    assert not editor.command_manager._undo_history


def test_redo(minimal_grid_and_topo, tmp_path):
    topo = minimal_grid_and_topo
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)

    orig_depth, new_depth, i, j = setup_depth(editor)
    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth

    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth
    assert not editor.command_manager._redo_history


def test_undo_redo_interleaving(minimal_grid_and_topo, tmp_path):
    topo = minimal_grid_and_topo
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)

    i, j = 2, 2
    orig = float(editor.topo.depth.data[j, i])

    edit_100 = DepthEditCommand(editor.topo, [(j, i)], [100.0], old_values=[orig])
    editor.apply_edit(edit_100)

    edit_200 = DepthEditCommand(editor.topo, [(j, i)], [200.0], old_values=[100.0])
    editor.apply_edit(edit_200)

    edit_300 = DepthEditCommand(editor.topo, [(j, i)], [300.0], old_values=[200.0])
    editor.apply_edit(edit_300)

    # Undo twice (should go from 300 -> 200 -> 100)
    editor.undo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == 200.0
    editor.undo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == 100.0

    # Make a new edit (should clear redo)
    edit_999 = DepthEditCommand(editor.topo, [(j, i)], [999.0], old_values=[100.0])
    editor.apply_edit(edit_999)
    assert float(editor.topo.depth.data[j, i]) == 999.0

    editor.redo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == 999.0


def test_redo_cleared_after_new_edit(minimal_grid_and_topo, tmp_path):
    topo = minimal_grid_and_topo
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)

    i, j = 2, 2
    orig = float(editor.topo.depth.data[j, i])

    # Apply three edits
    edit_100 = DepthEditCommand(editor.topo, [(j, i)], [100.0], old_values=[orig])
    editor.apply_edit(edit_100)
    edit_200 = DepthEditCommand(editor.topo, [(j, i)], [200.0], old_values=[100.0])
    editor.apply_edit(edit_200)
    edit_300 = DepthEditCommand(editor.topo, [(j, i)], [300.0], old_values=[200.0])
    editor.apply_edit(edit_300)

    # Undo twice (now redo stack has two)
    editor.undo_last_edit()
    editor.undo_last_edit()
    assert len(editor.command_manager._redo_history) == 2

    # Apply a new edit, should clear redo stack
    edit_999 = DepthEditCommand(editor.topo, [(j, i)], [999.0], old_values=[100.0])
    editor.apply_edit(edit_999)
    assert len(editor.command_manager._redo_history) == 0


def test_undo_redo_empty_history_noop(minimal_grid_and_topo, tmp_path):
    topo = minimal_grid_and_topo
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)

    i, j = 2, 2
    orig = float(editor.topo.depth.data[j, i])

    # Undo/redo with empty history should not raise or change anything
    editor.undo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == orig
    editor.redo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == orig


def test_save_and_load_histories_with_setup_depth(minimal_grid_and_topo, tmp_path):
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)

    topo = minimal_grid_and_topo
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)
    orig_depth, new_depth, i, j = setup_depth(editor)
    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth

    editor.command_manager.snapshot_dir = snapshot_dir
    editor.command_manager.save_commit("test_snapshot")

    # Create a new topo/editor and load the history
    topo2 = minimal_grid_and_topo
    editor2 = PatchedTopoEditor(topo2, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor2)
    editor2.command_manager.snapshot_dir = snapshot_dir
    from mom6_bathy.edit_command import COMMAND_REGISTRY

    print("COMMAND_REGISTRY keys:", list(COMMAND_REGISTRY.keys()))
    editor2.command_manager.load_commit("test_snapshot", COMMAND_REGISTRY, editor2.topo)
    editor2.command_manager.replay()
    # After replaying the edit history, topo2 should have new_depth at (j, i)
    assert float(editor2.topo.depth.data[j, i]) == new_depth
    # If you undo, you should get back to the original depth
    editor2.undo_last_edit()
    assert float(editor2.topo.depth.data[j, i]) == orig_depth


def test_save_and_load_histories_with_git(minimal_grid_and_topo, tmp_path):
    from mom6_bathy.edit_command import COMMAND_REGISTRY

    # Initialize a git repo in the temp directory
    repo = git.Repo.init(tmp_path)
    # Add a dummy file and commit so the repo isn't empty
    dummy_file = tmp_path / "README.md"
    dummy_file.write_text("dummy")
    repo.index.add([str(dummy_file)])
    repo.index.commit("initial commit")

    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)

    topo = minimal_grid_and_topo
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)

    orig_depth, new_depth, i, j = setup_depth(editor)
    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth

    # Set the snapshot directory to the repo root (for git tracking)
    editor.command_manager.snapshot_dir = snapshot_dir
    editor.command_manager.repo_root = str(tmp_path)  # If needed by your logic
    editor.command_manager.save_commit("test_snapshot")

    # Create a new topo/editor and load the history
    topo2 = minimal_grid_and_topo
    editor2 = PatchedTopoEditor(topo2, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor2)
    editor2.command_manager.snapshot_dir = snapshot_dir
    editor2.command_manager.repo_root = str(tmp_path)  # If needed by your logic
    print("COMMAND_REGISTRY keys:", list(COMMAND_REGISTRY.keys()))
    editor2.command_manager.load_commit("test_snapshot", COMMAND_REGISTRY, editor2.topo)
    editor2.command_manager.replay()

    # After replaying the edit history, topo2 should have new_depth at (j, i)
    assert float(editor2.topo.depth.data[j, i]) == new_depth

    # If you undo, you should get back to the original depth
    editor2.undo_last_edit()
    assert float(editor2.topo.depth.data[j, i]) == orig_depth


def test_in_memory_replay_does_not_reset_to_original(minimal_grid_and_topo, tmp_path):
    """Replay should apply edits on top of current topo, not reset to original."""
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)

    topo = minimal_grid_and_topo
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)
    i, j = 2, 2
    orig = float(editor.topo.depth.data[j, i])

    # Apply two edits
    edit_100 = DepthEditCommand(editor.topo, [(j, i)], [100.0], old_values=[orig])
    editor.apply_edit(edit_100)
    edit_200 = DepthEditCommand(editor.topo, [(j, i)], [200.0], old_values=[100.0])
    editor.apply_edit(edit_200)

    # Change topo directly (simulate user edit outside command stack)
    editor.topo.depth.data[j, i] = 777.0

    # Replay should apply edits on top of current topo, not reset to original
    editor.command_manager.replay()
    assert float(editor.topo.depth.data[j, i]) == 200.0


def test_commit_load_resets_to_original(minimal_grid_and_topo, tmp_path):
    """Loading a commit should reset topo to original before replaying history."""
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)

    topo = minimal_grid_and_topo
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)
    i, j = 2, 2
    orig = float(editor.topo.depth.data[j, i])

    # Apply and save an edit
    edit_100 = DepthEditCommand(editor.topo, [(j, i)], [100.0], old_values=[orig])
    editor.apply_edit(edit_100)
    editor.command_manager.save_commit("test_commit")

    # Change topo directly (simulate user edit outside command stack)
    editor.topo.depth.data[j, i] = 777.0

    # Load commit, should reset to original and replay history
    from mom6_bathy.edit_command import COMMAND_REGISTRY

    editor.command_manager.load_commit("test_commit", COMMAND_REGISTRY, editor.topo)
    editor.command_manager.replay()
    assert float(editor.topo.depth.data[j, i]) == 100.0


def test_switching_between_flows_branches(minimal_grid_and_topo, tmp_path):
    """Switching between two histories should replay the correct one."""
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)

    topo = minimal_grid_and_topo
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)
    i, j = 2, 2
    orig = float(editor.topo.depth.data[j, i])

    # Branch 1: edit to 100, save
    edit_100 = DepthEditCommand(editor.topo, [(j, i)], [100.0], old_values=[orig])
    editor.apply_edit(edit_100)
    editor.command_manager.save_commit("branch1")

    # Undo, Branch 2: edit to 200, save
    editor.undo_last_edit()
    edit_200 = DepthEditCommand(editor.topo, [(j, i)], [200.0], old_values=[orig])
    editor.apply_edit(edit_200)
    editor.command_manager.save_commit("branch2")

    from mom6_bathy.edit_command import COMMAND_REGISTRY

    # Load branch1
    editor.command_manager.load_commit("branch1", COMMAND_REGISTRY, editor.topo)
    editor.command_manager.replay()
    assert float(editor.topo.depth.data[j, i]) == 100.0

    # Load branch2
    editor.command_manager.load_commit("branch2", COMMAND_REGISTRY, editor.topo)
    editor.command_manager.replay()
    assert float(editor.topo.depth.data[j, i]) == 200.0


def test_undo_redo_after_commit_load(minimal_grid_and_topo, tmp_path):
    """Undo/redo should work after loading a commit."""
    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)

    topo = minimal_grid_and_topo
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=snapshot_dir)
    patch_all_widgets(editor)
    i, j = 2, 2
    orig = float(editor.topo.depth.data[j, i])

    # Apply and save two edits
    edit_100 = DepthEditCommand(editor.topo, [(j, i)], [100.0], old_values=[orig])
    editor.apply_edit(edit_100)
    edit_200 = DepthEditCommand(editor.topo, [(j, i)], [200.0], old_values=[100.0])
    editor.apply_edit(edit_200)
    editor.command_manager.save_commit("test_commit")

    # Load commit
    from mom6_bathy.edit_command import COMMAND_REGISTRY

    editor.command_manager.load_commit("test_commit", COMMAND_REGISTRY, editor.topo)
    editor.command_manager.replay()
    assert float(editor.topo.depth.data[j, i]) == 200.0

    # Undo should go to 100, redo should go to 200
    editor.undo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == 100.0
    editor.redo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == 200.0


def test_merge_branch(tmp_path):
    from mom6_bathy.git_utils import merge_branch

    # Initialize a git repo
    repo = git.Repo.init(tmp_path)
    dummy_file = tmp_path / "README.md"
    dummy_file.write_text("main branch")
    repo.index.add([str(dummy_file)])
    repo.index.commit("initial commit")

    # Create and switch to a new branch
    repo.git.checkout("-b", "feature")
    feature_file = tmp_path / "feature.txt"
    feature_file.write_text("feature branch")
    repo.index.add([str(feature_file)])
    repo.index.commit("add feature file")

    # Switch back to main branch
    repo.git.checkout("master")

    # Try to merge master into itself (should fail)
    success, msg = merge_branch(str(tmp_path), "master")
    assert not success
    assert "Cannot merge a branch into itself" in msg

    # Merge feature into master (should succeed)
    success, msg = merge_branch(str(tmp_path), "feature")
    assert success
    assert "Merged branch 'feature' into 'master'" in msg

    # After merge, feature.txt should exist in master
    assert (tmp_path / "feature.txt").exists()


def test_new_domain_creates_original_snapshot(tmp_path, minimal_grid_and_topo):
    from mom6_bathy import git_utils

    # --- Patch snapshot_action to no-op ---
    git_utils.snapshot_action = lambda *a, **k: "skipped"

    topo = minimal_grid_and_topo

    # --- Setup snapshot directory ---
    snapshot_dir = tmp_path / "Topos"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    topo.SNAPSHOT_DIR = str(snapshot_dir)
    topo._repo_root = None  # ensure Git is disabled

    # --- Create editor ---
    editor = PatchedTopoEditor(topo, build_ui=False, snapshot_dir=str(snapshot_dir))

    # --- Ensure files exist ---
    grid_name = getattr(topo._grid, "name", getattr(topo._grid, "_name", None))
    shape = topo.depth.data.shape
    shape_str = f"{shape[0]}x{shape[1]}"
    original_topo_path = snapshot_dir / f"original_topo_{grid_name}_{shape_str}.nc"
    original_min_depth_path = (
        snapshot_dir / f"original_min_depth_{grid_name}_{shape_str}.json"
    )
    original_json_path = snapshot_dir / f"original_{grid_name}_{shape_str}.json"

    assert original_topo_path.exists(), "Original topo NetCDF file should exist"
    assert original_min_depth_path.exists(), "Original min depth JSON file should exist"
    assert original_json_path.exists(), "Original snapshot JSON file should exist"

    with open(original_min_depth_path) as f:
        min_depth = json.load(f)
    assert "min_depth" in min_depth

    with open(original_json_path) as f:
        snapshot_data = json.load(f)
    assert snapshot_data is not None


def test_ui_state_after_domain_switch(tmp_path):
    import os
    import numpy as np
    from ipywidgets import HBox, FloatText, Dropdown, Button, ToggleButton
    import git

    snapshot_dir = str(tmp_path / "Topos")
    os.makedirs(snapshot_dir, exist_ok=True)

    # Initialize Git repo
    repo = git.Repo.init(tmp_path)
    dummy_file = tmp_path / "README.md"
    dummy_file.write_text("dummy")
    repo.index.add([str(dummy_file)])
    repo.git.checkout("-b", "master")
    repo.index.commit("initial commit")

    # -----------------------
    # First domain (resolution 0.1)
    # -----------------------
    grid1 = Grid(
        resolution=0.1, xstart=278.0, lenx=0.5, ystart=7.0, leny=0.5, name="testpanama"
    )
    topo1 = Topo(grid=grid1, min_depth=10.0)
    topo1.set_flat(100.0)

    domain_dir1 = os.path.join(
        snapshot_dir, f"domain_{grid1.name}_{grid1.qlon.shape[0]}x{grid1.qlon.shape[1]}"
    )
    os.makedirs(domain_dir1, exist_ok=True)
    topo1.SNAPSHOT_DIR = domain_dir1
    topo1.repo_root = str(tmp_path)

    editor = PatchedTopoEditor(topo1, build_ui=False, snapshot_dir=domain_dir1)
    patch_all_widgets(editor)

    # Patch GUI plotting & control panel to avoid errors
    editor.construct_interactive_plot = lambda: setattr(
        editor, "_interactive_plot", HBox()
    )
    editor.construct_control_panel = lambda: setattr(editor, "_control_panel", HBox())
    editor.refresh_commit_dropdown = lambda: None

    # Patch all widgets required by construct_observances
    editor._display_mode_toggle = ToggleButton(value=False)
    editor._min_depth_specifier = FloatText(value=topo1.min_depth)
    editor._basin_specifier_toggle = Button()
    editor._basin_specifier_delete_selected = Button()
    editor._depth_specifier = FloatText(value=topo1.min_depth)
    editor._undo_button = Button()
    editor._redo_button = Button()
    editor._reset_button = Button()
    editor._save_button = Button()
    editor._load_button = Button()
    editor._switch_domain_button = Button()
    editor._git_create_branch_button = Button()
    editor._git_delete_branch_button = Button()
    editor._git_checkout_button = Button()
    editor._git_merge_button = Button()
    editor._snapshot_name = FloatText(value=0.0)
    editor._commit_dropdown = Dropdown(options=[])
    editor.fig = type(
        "DummyFig",
        (),
        {
            "canvas": type(
                "DummyCanvas", (), {"mpl_connect": lambda self, *a, **k: None}
            )()
        },
    )()

    # -----------------------
    # Second domain (resolution 0.2)
    # -----------------------
    grid2 = Grid(
        resolution=0.2, xstart=278.0, lenx=0.5, ystart=7.0, leny=0.5, name="testpanama"
    )
    topo2 = Topo(grid=grid2, min_depth=10.0)
    topo2.set_flat(50.0)

    domain_dir2 = os.path.join(
        snapshot_dir, f"domain_{grid2.name}_{grid2.qlon.shape[0]}x{grid2.qlon.shape[1]}"
    )
    os.makedirs(domain_dir2, exist_ok=True)
    topo2.SNAPSHOT_DIR = domain_dir2
    topo2.repo_root = str(tmp_path)

    # -----------------------
    # Load new topo (simulate domain switch)
    # -----------------------
    editor.load_new_topo(topo2)
    patch_all_widgets(editor)

    # Manually sync min_depth widget with topo2
    if hasattr(editor, "_min_depth_specifier"):
        editor._min_depth_specifier.value = topo2.min_depth

    # -----------------------
    # Assertions
    # -----------------------
    assert editor.topo._grid.resolution == 0.2
    assert editor.topo.depth.data.shape == topo2.depth.data.shape
    assert np.allclose(editor.topo.depth.data, topo2.depth.data)

    if hasattr(editor, "_min_depth_specifier"):
        assert editor._min_depth_specifier.value == 10.0
