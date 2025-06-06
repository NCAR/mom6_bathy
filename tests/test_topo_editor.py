import os
import json
import numpy as np
import pytest

from mom6_bathy.grid import Grid
from mom6_bathy.topo import Topo
from mom6_bathy.topo_editor import TopoEditor

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
        # Return self so any attribute access works
        return self

    def __call__(self, *args, **kwargs):
        # Dummy is callable, returns self or None as needed
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
        # Some widgets expect an initial value; set depth for display_mode_toggle
        value = "depth" if attr == '_display_mode_toggle' else None
        setattr(editor, attr, Dummy(value=value))

def setup_depth(editor, i=2, j=2, new_depth=777.0):
    """Helper to perform a depth change and register it in undo_history."""
    orig_depth = float(editor.topo.depth.data[j, i])
    editor._select_cell(i, j)
    edit = {
        'action': 'depth_change',
        'i': i,
        'j': j,
        'old_value': orig_depth,
        'new_value': new_depth,
        "timestamp": "now",
    }
    editor.apply_edit(edit)
    editor.undo_last_edit()
    return orig_depth, new_depth, i, j

def test_undo(minimal_grid_and_topo):
    topo = minimal_grid_and_topo
    editor = TopoEditor(topo, build_ui=False)
    patch_all_widgets(editor)

    orig_depth, new_depth, i, j = setup_depth(editor)
    assert float(topo.depth.data[j, i]) == orig_depth

    # Test for undo when there are no more edits to undo
    editor.undo_last_edit()
    assert float(topo.depth.data[j, i]) == orig_depth
    assert not editor._undo_history

def test_redo(minimal_grid_and_topo):
    topo = minimal_grid_and_topo
    editor = TopoEditor(topo, build_ui=False)
    patch_all_widgets(editor)

    orig_depth, new_depth, i, j = setup_depth(editor)
    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth

    # Test for redo when there are no more edits to redo
    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth
    assert not editor._redo_history

def test_undo_redo_interleaving(minimal_grid_and_topo):
    topo = minimal_grid_and_topo
    editor = TopoEditor(topo, build_ui=False)
    patch_all_widgets(editor)

    i, j = 2, 2
    orig = float(editor.topo.depth.data[j, i])

    # First change: 100
    edit_100 = {
        'action': 'depth_change',
        'i': i,
        'j': j,
        'old_value': orig,
        'new_value': 100.0,
        "timestamp": "now",
    }
    editor.apply_edit(edit_100)

    # Second change: 200
    edit_200 = {
        'action': 'depth_change',
        'i': i,
        'j': j,
        'old_value': 100.0,
        'new_value': 200.0,
        "timestamp": "now",
    }
    editor.apply_edit(edit_200)

    # Third change: 300
    edit_300 = {
        'action': 'depth_change',
        'i': i,
        'j': j,
        'old_value': 200.0,
        'new_value': 300.0,
        "timestamp": "now",
    }
    editor.apply_edit(edit_300)

    # Undo twice (should go from 300 -> 200 -> 100)
    editor.undo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == 200.0
    editor.undo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == 100.0

    # Make a new edit (should clear redo)
    edit_999 = {
        'action': 'depth_change',
        'i': i,
        'j': j,
        'old_value': 100.0,
        'new_value': 999.0,
        "timestamp": "now",
    }
    editor.apply_edit(edit_999)
    assert float(editor.topo.depth.data[j, i]) == 999.0

    # Redo should do nothing now
    editor.redo_last_edit()
    assert float(editor.topo.depth.data[j, i]) == 999.0

    import json
import os

import json
import os

def test_save_and_load_histories_with_setup_depth(minimal_grid_and_topo, tmp_path):
    topo = minimal_grid_and_topo
    editor = TopoEditor(topo, build_ui=False)
    patch_all_widgets(editor)

    orig_depth, new_depth, i, j = setup_depth(editor)

    editor.redo_last_edit()
    assert float(topo.depth.data[j, i]) == new_depth

    # Save histories to a file
    save_path = tmp_path / "edit_history_test.json"
    editor.save_histories(str(save_path))

    # Create a new editor instance and load the histories
    topo2 = minimal_grid_and_topo
    editor2 = TopoEditor(topo2, build_ui=False)
    patch_all_widgets(editor2)
    editor2.load_histories(str(save_path))

    # Check that the history was loaded correctly
    with open(save_path) as f:
        data = json.load(f)
        assert data["undo_history"] == editor2._undo_history
        assert data["redo_history"] == editor2._redo_history
        assert data["topo_id"] == editor2.get_topo_id()

    # After replaying the edit history, topo2 should have new_depth at (j, i)
    editor2.replay_edit_history()
    assert float(editor2.topo.depth.data[j, i]) == new_depth

    # If you undo, you should get back to the original depth
    editor2.undo_last_edit()
    assert float(editor2.topo.depth.data[j, i]) == orig_depth