import os
import json
import numpy as np
import pytest

from mom6_bathy.grid import Grid
from mom6_bathy.topo import Topo
from CrocoDash.visualCaseGen.external.mom6_bathy.mom6_bathy.edit_command import (
    DepthEditCommand, MinDepthEditCommand, UndoCommand, RedoCommand,
    SaveCommitCommand, LoadCommitCommand, ResetCommand, InitializeHistoryCommand
)
from mom6_bathy.topo_editor import TopoEditor
from CrocoDash.visualCaseGen.external.mom6_bathy.mom6_bathy.edit_command import COMMAND_REGISTRY

@pytest.fixture
def minimal_grid_and_topo():
    """Test that a minimal 5x5 grid can be set up for the Panama region"""
    grid = Grid(
        resolution=0.1,
        xstart=278.0,
        lenx=0.5,
        ystart=7.0,
        leny=0.5,
        name="testpanama"
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

def patch_all_widgets(editor):
    dummy_attrs = [
        '_undo_button', '_redo_button', '_reset_button',
        '_min_depth_specifier', '_depth_specifier', '_selected_cell_label',
        '_basin_specifier', '_basin_specifier_toggle',
        '_basin_specifier_delete_selected', '_snapshot_name',
        '_save_button', '_load_button', '_display_mode_toggle',
        'im', 'cbar', 'ax', 'fig'
    ]
    for attr in dummy_attrs:
        value = "depth" if attr == '_display_mode_toggle' else None
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
    editor.undo_last_edit()
    return orig_depth, new_depth, i, j

def test_undo(minimal_grid_and_topo):
    topo = minimal_grid_and_topo
    editor = TopoEditor(topo, build_ui=False)
    patch_all_widgets(editor)

    orig_depth, new_depth, i, j = setup_depth(editor)
    assert float(topo.depth.data[j, i]) == orig_depth

    editor.undo_last_edit()
    assert float(topo.depth.data[j, i]) == orig_depth
    assert not editor.command_manager._undo_history

def test_redo(minimal_grid_and_topo):
    topo = minimal_grid_and_topo
    editor = TopoEditor(topo, build_ui=False)
    patch_all_widgets(editor)

    orig_depth, new_depth, i, j = setup_depth(editor)
    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth

    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth
    assert not editor.command_manager._redo_history

def test_undo_redo_interleaving(minimal_grid_and_topo):
    topo = minimal_grid_and_topo
    editor = TopoEditor(topo, build_ui=False)
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

def test_save_and_load_histories_with_setup_depth(minimal_grid_and_topo, tmp_path):
    topo = minimal_grid_and_topo
    editor = TopoEditor(topo, build_ui=False)
    patch_all_widgets(editor)

    orig_depth, new_depth, i, j = setup_depth(editor)

    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth

    editor.command_manager.snapshot_dir = str(tmp_path)
    editor.command_manager.save_commit("test_snapshot")

    topo2 = minimal_grid_and_topo
    editor2 = TopoEditor(topo2, build_ui=False)
    patch_all_widgets(editor2)
    editor2.command_manager.snapshot_dir = str(tmp_path)
    print("COMMAND_REGISTRY keys:", list(COMMAND_REGISTRY.keys()))
    editor2.command_manager.load_commit("test_snapshot", COMMAND_REGISTRY, editor2.topo)
    editor2.command_manager.replay()

    # After replaying the edit history, topo2 should have new_depth at (j, i)
    assert float(editor2.topo.depth.data[j, i]) == new_depth

    # If you undo, you should get back to the original depth
    editor2.undo_last_edit()
    assert float(editor2.topo.depth.data[j, i]) == orig_depth