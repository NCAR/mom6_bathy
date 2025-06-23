import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import ipywidgets as widgets
import os
import json
from matplotlib.ticker import MaxNLocator
from mom6_bathy.command_manager import TopoCommandManager
from mom6_bathy.edit_command import (
    DepthEditCommand, MinDepthEditCommand, COMMAND_REGISTRY
)
from mom6_bathy.git_utils import (
    git_snapshot_action, git_commit_info, git_get_current_branch, git_list_branches,
    git_create_branch_and_switch, git_delete_branch_and_switch, git_merge_branch, git_safe_checkout_branch
)

CROCODASH_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))

class TopoEditor(widgets.HBox):
    
    def __init__(self, topo, build_ui=True, snapshot_dir="snapshots", golden_dir="original_topo"):

        self.topo = topo
        self.ny = self.topo.depth.data.shape[0]
        self.nx = self.topo.depth.data.shape[1]

        self.SNAPSHOT_DIR = snapshot_dir
        self.GOLDEN_DIR = golden_dir

        os.makedirs(self.SNAPSHOT_DIR, exist_ok=True)
        os.makedirs(self.GOLDEN_DIR, exist_ok=True)
        self.current_branch = git_get_current_branch(CROCODASH_REPO_ROOT)

        self.command_manager = TopoCommandManager(domain_id=self.get_topo_id, topo=self.topo, command_registry=COMMAND_REGISTRY)

        self._selected_cell = None

        self._original_depth = self.topo.depth.data.copy()
        self._original_min_depth = self.topo.min_depth

        # Ensure golden/original topo exists for resets
        self._ensure_golden_topo()

        if build_ui:
            # Setup UI controls, plot, and observers
            self.construct_control_panel()
            self.construct_interactive_plot()
            self.construct_observances()
            self.initialize_history()
            self.refresh_commit_dropdown()
            super().__init__([self._control_panel, self._interactive_plot])
        else:
            super().__init__([])

    def initialize_history(self):
        self.command_manager.initialize()
        self.update_undo_redo_buttons()

    def get_topo_id(self):
        grid = self.topo._grid
        grid_name = getattr(grid, "name", getattr(grid, "_name", None))
        shape = [int(v) for v in self.topo.depth.data.shape]
        return {"grid_name": grid_name, "shape": shape}

    def _ensure_golden_topo(self):
        topo_id = self.get_topo_id()
        grid_name = topo_id['grid_name']
        shape = topo_id['shape']
        shape_str = f"{shape[0]}x{shape[1]}"
        
        golden_dir = self.GOLDEN_DIR
        os.makedirs(golden_dir, exist_ok=True)
        golden_topo_path = os.path.join(golden_dir, f"golden_topo_{grid_name}_{shape_str}.npy")
        golden_min_depth_path = os.path.join(golden_dir, f"golden_min_depth_{grid_name}_{shape_str}.json")
        if not os.path.exists(golden_topo_path):
            np.save(golden_topo_path, self.topo.depth.data)
            with open(golden_min_depth_path, "w") as f:
                json.dump({"min_depth": float(self.topo.min_depth)}, f)

        golden_name = f"golden_{grid_name}_{shape_str}"
        golden_path = os.path.join(self.SNAPSHOT_DIR, f"{golden_name}.json")
        try:
            self.command_manager.load_commit(golden_name, COMMAND_REGISTRY, self.topo)
        except FileNotFoundError:
            self.command_manager.save_commit(golden_name)

        status = git_snapshot_action('ensure_tracked', CROCODASH_REPO_ROOT, 
                                     file_path=golden_path, commit_msg=f"Update golden snapshot {golden_name}")

    def save_commit(self, _btn=None):
        name = self._snapshot_name.value.strip()
        if not name:
            print("Enter a name!")
            return
        self.command_manager.save_commit(name)
        print(f"Saved snapshot '{name}'.")
        self.refresh_commit_dropdown()

    def load_commit(self, _btn=None, commit_sha=None, file_path=None):
        if not commit_sha or not file_path:
            print("No commit or file selected.")
            return
        rel_path = file_path
        abs_path = os.path.join(CROCODASH_REPO_ROOT, rel_path)
        success, msg = git_snapshot_action('restore', CROCODASH_REPO_ROOT, file_path=rel_path, commit_sha=commit_sha)
        if success:
            print(msg)
        else:
            if ("does not exist in" in msg or "exists on disk, but not in" in msg) and "golden_" in os.path.basename(file_path):
                if os.path.exists(abs_path):
                    print(f"Golden snapshot '{os.path.basename(file_path)}' was never committed, loading from disk.")
                else:
                    print(f"Golden snapshot '{os.path.basename(file_path)}' not found on disk.")
                    return
            else:
                print(f"Failed to load commit: {msg}")
                return
        self._snapshot_name.value = os.path.splitext(os.path.basename(file_path))[0]
        self.command_manager.load_commit(self._snapshot_name.value, COMMAND_REGISTRY, self.topo, reset_to_golden=True)
        self.update_undo_redo_buttons()
        self.trigger_refresh()
        print(f"Loaded snapshot '{self._snapshot_name.value}'.")

    def apply_edit(self, cmd, record_history=True):
        self.command_manager.execute(cmd)
        self.update_undo_redo_buttons()
        self.trigger_refresh()

    def undo_last_edit(self, b=None):
        self.command_manager.undo()
        self.update_undo_redo_buttons()
        self.trigger_refresh()
    
    def redo_last_edit(self, b=None):
        self.command_manager.redo()
        self.update_undo_redo_buttons()
        self.trigger_refresh()

    def on_reset(self, b=None):
        self.command_manager.reset(
            self.topo,
            self._original_depth,
            self._original_min_depth,
            self.get_topo_id,
            min_depth_specifier=self._min_depth_specifier,
            trigger_refresh=self.trigger_refresh
        )
        self.update_undo_redo_buttons()
        print("Topo reset to original state.")

    def update_undo_redo_buttons(self):
        if hasattr(self, "_undo_button"):
            self._undo_button.disabled = not bool(self.command_manager._undo_history)
        if hasattr(self, "_redo_button"):
            self._redo_button.disabled = not bool(self.command_manager._redo_history)

    def refresh_commit_dropdown(self):
        rel_snapshots_dir = os.path.relpath(self.SNAPSHOT_DIR, CROCODASH_REPO_ROOT)
        options = git_commit_info(CROCODASH_REPO_ROOT, rel_dir=rel_snapshots_dir, mode='list')
        self._commit_dropdown.options = options
        if options:
            self._commit_dropdown.value = options[0][1]
        self.update_commit_details()

    def update_commit_details(self, change=None):
        val = self._commit_dropdown.value
        if not val:
            self._commit_details.value = ""
            return
        commit_sha, file_path = val
        self._commit_details.value = git_commit_info(CROCODASH_REPO_ROOT, commit_sha=commit_sha, file_path=file_path, mode='details')

    def construct_interactive_plot(self):
        # Ensure we are in interactive mode
        # This is default but if this notebook is executed out of order it may have been turned off
        plt.ioff()
        self.fig = plt.figure()
        plt.ion()
        self.ax = self.fig.gca()
        self.ax.clear()

        def format_coord(x, y):
            j, i = self.topo._grid.get_indices(y, x)
            return f'x={x:.2f}, y={y:.2f}, i={i}, j={j} depth={self.topo.depth.data[j, i]:.2f}'
        self.ax.format_coord = format_coord

        # Minor ticks
        self.ax.set_xticks(np.arange(-.5, self.nx, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.ny, 1), minor=True)

        # Remove minor ticks
        self.ax.tick_params(which='minor', bottom=False, left=False)

        # Create a new colormap where the values below self.topo.min_depth are white
        # and the rest is the default colormap
        self.cmap = plt.get_cmap('viridis')
        self.cmap.set_under('w')

        # plot
        self.im = self.ax.pcolormesh(
            self.topo._grid.qlon.data,
            self.topo._grid.qlat.data,
            self.topo.depth.data, 
            vmin=self.topo.min_depth, 
            cmap=self.cmap, 
        )

        # title
        self.ax.set_title('Double click on a cell to change its depth.')

        # colorbar
        self.cbar = self.fig.colorbar(self.im)

        # colorbar title
        self.cbar.set_label(f'Depth ({self.topo.depth.units})')

        # Enforce Cbar integer values for easier mask/basinmask colorbar
        self.cbar.set_ticks(MaxNLocator(integer=True))  # Ensure ticks are integers 
        # x and y labels
        self.ax.set_xlabel(f'x ({self.topo._grid.qlon.units})')
        self.ax.set_ylabel(f'y ({self.topo._grid.qlat.units})')

        # Show navigation toolbar
        self.fig.canvas.toolbar_visible = True
        self.fig.canvas.toolbar_position = 'top'

        # set canvas title:
        self._interactive_plot = widgets.HBox(
            children=(self.fig.canvas,),
            layout = {'border_left':'1px solid grey'}
        )

    def construct_control_panel(self):
        self._min_depth_specifier = widgets.BoundedFloatText(
            value=self.topo.min_depth,
            min=-1000.0,
            max=self.topo.depth.max(skipna=True).item(),
            step=10.0,
            description='Min depth (m):',
            disabled=False,
            layout={'width': '80%'},
            style={'description_width': 'auto'}
        )

        self._display_mode_toggle = widgets.ToggleButtons(
            options=['depth', 'mask', 'basinmask'],
            description='Field:',
            disabled=False,
            tooltips=['Display depth values', 'Display mask values', 'Display Basins'],
            layout={'width': '90%', 'display': 'flex'},
            style={'description_width': '40px', 'button_width': '85px'}
        )

        self._selected_cell_label = widgets.Label("Selected cell: None (double click to select a cell).")

        self._depth_specifier = widgets.FloatText(
            value=None,
            step=10.0,
            description='Depth (m):',
            disabled=True,
            placeholder='Select a cell first.',
            layout={'width': '80%'},
            style={'description_width': 'auto'}
        )

        self._basin_specifier_toggle = widgets.Button(
            description="Erase Disconnected Basins",
            disabled=True,
            layout={'width': '90%', 'display': 'flex'},
            style={'description_width': '100px'}
        )
        
        self._basin_specifier_delete_selected = widgets.Button(
            description="Erase Selected Basin",
            disabled=True,
            layout={'width': '90%', 'display': 'flex'},
            style={'description_width': '100px'}
        )
        
        self._basin_specifier = widgets.Label(
            value='Basin Label Number: None',
            layout={'width': '80%'},
            style={'description_width': 'auto'}
        )

        # Undo/Redo
        self._undo_button = widgets.Button(description='Undo', disabled=True, layout={'width': '44%'})
        self._redo_button = widgets.Button(description='Redo', disabled=True, layout={'width': '44%'})
        # Reset
        self._reset_button = widgets.Button(description='Reset', layout={'width': '44%'}, button_style='danger')
        # Snapshots
        self._snapshot_name = widgets.Text(value='', placeholder='Enter snapshot name', description='Snapshot:', layout={'width': '90%'})
        self._commit_dropdown = widgets.Dropdown(
            options=[],
            description='Commit:',
            layout={'width': '90%'}
        )
        self._commit_details = widgets.HTML(
            value="",
            layout={'width': '90%', 'min_height': '2em'}
        )
        self._save_button = widgets.Button(description='Save State', layout={'width': '44%'})
        self._load_button = widgets.Button(description='Load State', layout={'width': '44%'})
        # Git Version Control
        self._git_commit_msg = widgets.Text(
            value='',
            placeholder='Enter Git commit message',
            description='Git Msg:',
            layout={'width': '90%'}
        )
        self._git_commit_button = widgets.Button(description='Git Commit', layout={'width': '44%'})
        self._git_branch_name = widgets.Text(
            value='',
            placeholder='New branch name',
            description='Branch:',
            layout={'width': '90%'}
        )
        self._git_create_branch_button = widgets.Button(description='Create Branch', layout={'width': '44%'})
        self._git_delete_branch_button = widgets.Button(
            description='Delete Branch',
            layout={'width': '44%'},
            button_style='danger'  # Makes it red
        )
        self._git_branch_dropdown = widgets.Dropdown(
            options=git_list_branches(CROCODASH_REPO_ROOT),
            description='Checkout:',
            layout={'width': '90%'}
        )
        self._git_checkout_button = widgets.Button(description='Checkout', layout={'width': '44%'})
       
        self._git_merge_source_dropdown = widgets.Dropdown(
            options=git_list_branches(CROCODASH_REPO_ROOT),
            description='Merge from:',
            layout={'width': '90%'}
        )
        self._git_merge_button = widgets.Button(description='Merge Branch', layout={'width': '44%'})

         # Group related controls
        display_section = widgets.VBox([
            widgets.HTML("<h3>Display</h3>"),
            self._display_mode_toggle,
        ])

        global_settings_section = widgets.VBox([
            widgets.HTML("<h3>Global Settings</h3>"),
            self._min_depth_specifier,
        ])

        cell_editing_section = widgets.VBox([
            widgets.HTML("<h3>Cell Editing</h3>"),
            self._selected_cell_label,
            self._depth_specifier,
        ])

        basin_section = widgets.VBox([
            widgets.HTML("<h3>Basin Selector</h3>"),
            self._basin_specifier,
            self._basin_specifier_toggle,      
            self._basin_specifier_delete_selected,  
        ])

        history_section = widgets.VBox([
            widgets.HTML("<h3>Edit History</h3>"),
            widgets.HBox([self._undo_button, self._redo_button, self._reset_button]),
        ])

        snapshot_section = widgets.VBox([
            self._snapshot_name,
            self._commit_dropdown,
            self._commit_details,
            widgets.HBox([self._save_button, self._load_button]),
        ])

        git_section = widgets.VBox([
            self._git_commit_msg,
            self._git_commit_button,
            self._git_branch_name,
            widgets.HBox([self._git_create_branch_button, self._git_delete_branch_button]),
            self._git_branch_dropdown,
            self._git_checkout_button,
            self._git_merge_source_dropdown,
            self._git_merge_button,
        ])

        # Always-visible controls
        main_controls = widgets.VBox([
            display_section,
            global_settings_section,
            cell_editing_section,
            basin_section,
            history_section,
        ])

        # Each advanced section in its own Accordion (so both can be open at once)
        snapshot_accordion = widgets.Accordion(children=[snapshot_section])
        snapshot_accordion.set_title(0, 'Snapshots')
        snapshot_accordion.selected_index = None  # collapsed by default

        git_accordion = widgets.Accordion(children=[git_section])
        git_accordion.set_title(0, 'Git Version Control')
        git_accordion.selected_index = None  # collapsed by default

        # Combine everything
        self._control_panel = widgets.VBox([
            widgets.HTML("<h2>Topo Editor</h2>"),
            main_controls,
            snapshot_accordion,
            git_accordion,
        ], layout={'width': '30%', 'height': '100%', 'overflow_y': 'auto'})

        current_branch = git_get_current_branch(CROCODASH_REPO_ROOT)
        if current_branch in self._git_branch_dropdown.options:
            self._git_branch_dropdown.value = current_branch

    def refresh_display_mode(self, change):
        mode = change['new']
        if mode == 'depth':
            self.im.set_clim(vmin = self.topo.min_depth, vmax = self.topo.depth.max(skipna=True).item())
            self.im.set_array(self.topo.depth.data)
            self.im.set_clim(vmin = self.topo.min_depth, vmax = self.topo.depth.max(skipna=True).item()) 
            self.cbar.set_label(f'Depth ({self.topo.depth.units})')
        elif mode == 'mask':
            self.im.set_array(self.topo.tmask.data)
            self.im.set_clim((0, 1))
            self.cbar.set_label('Land Mask')
        elif mode == 'basinmask':
            self.im.set_array(self.topo.basintmask.data)
            self.im.set_clim((0,self.topo.basintmask.data.max()))
            self.cbar.set_label('Basin Mask')
        else:
            raise ValueError(f"Unknown display mode: {mode}")
                
    def trigger_refresh(self):
        self.refresh_display_mode({'new': self._display_mode_toggle.value})

    def _select_cell(self, i, j):

        # Remove patch if exists (UI mode)
        if self._selected_cell is not None and len(self._selected_cell) > 2 and self._selected_cell[2] is not None and hasattr(self, "ax"):
            self._selected_cell[2].remove()

        # Try to create and add the patch (UI mode)
        polygon = None
        if hasattr(self, "ax"):
            vertices = np.array([
                [self.topo._grid.qlon.data[j, i], self.topo._grid.qlat.data[j, i]],
                [self.topo._grid.qlon.data[j, i+1], self.topo._grid.qlat.data[j, i+1]],
                [self.topo._grid.qlon.data[j+1, i+1], self.topo._grid.qlat.data[j+1, i+1]],
                [self.topo._grid.qlon.data[j+1, i], self.topo._grid.qlat.data[j+1, i]],
            ])
            polygon = patches.Polygon(vertices, edgecolor='r', facecolor='none', alpha=0.8, linewidth=2, label='Selected cell')
            self.ax.add_patch(polygon)

        # Update the selected cell reference
        self._selected_cell = (i, j, polygon)

        # Update label and specifiers if in UI mode
        if hasattr(self, "_selected_cell_label"):
            self._selected_cell_label.value = f"Selected cell: {i}, {j}"

        if hasattr(self, "_depth_specifier"):
            self._depth_specifier.disabled = False
            self._depth_specifier.value = self.topo.depth.data[j, i]

        if hasattr(self, "_basin_specifier"):
            label = self.topo.basintmask.data[j, i]
            self._basin_specifier.value = f"Basin Label Number: {str(label)}"
            # Enable/disable basin erase buttons if they exist
            if hasattr(self, "_basin_specifier_toggle") and hasattr(self, "_basin_specifier_delete_selected"):
                if label != 0:
                    self._basin_specifier_toggle.disabled = False
                    self._basin_specifier_delete_selected.disabled = False
                else:
                    self._basin_specifier_toggle.disabled = True
                    self._basin_specifier_delete_selected.disabled = True

    def construct_observances(self):
        # Display mode toggle
        self._display_mode_toggle.observe(self.refresh_display_mode, names='value', type='change')

        # Double click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_double_click)

        # Min depth change
        self._min_depth_specifier.observe(self.on_min_depth_change, names='value', type='change')

        # Basin erase buttons
        self._basin_specifier_toggle.on_click(self.erase_disconnected_basins)
        self._basin_specifier_delete_selected.on_click(self.erase_selected_basin)

        # Depth change
        self._depth_specifier.observe(self.on_depth_change, names='value', type='change')

        # Undo/Redo/Reset
        self._undo_button.on_click(self.undo_last_edit)
        self._redo_button.on_click(self.redo_last_edit)
        self._reset_button.on_click(self.on_reset)

        # Snapshots
        self._save_button.on_click(self.save_commit)
        self._load_button.on_click(self.on_load_button_clicked)
        self._snapshot_name.observe(lambda change: self.refresh_commit_dropdown(), names='value')
        self._commit_dropdown.observe(self.update_commit_details, names='value')

        # Git
        self._git_commit_button.on_click(self.on_git_commit)
        self._git_create_branch_button.on_click(self.on_git_create_branch)
        self._git_delete_branch_button.on_click(self.on_git_delete_branch)
        self._git_checkout_button.on_click(self.on_git_checkout)
        self._git_merge_button.on_click(self.on_git_merge)

        self._display_mode_toggle.observe(
            self.refresh_display_mode,
            names='value',
            type='change'
        )

    # --- UI Callback Methods ---

    def on_double_click(self, event):
        if event.dblclick:
            y, x = event.ydata, event.xdata
            j, i = self.topo._grid.get_indices(y, x)
            self._select_cell(i, j)

    def on_min_depth_change(self, change):
        old_val = self.topo.min_depth
        new_val = change['new']
        if old_val != new_val:
            cmd = MinDepthEditCommand(self.topo, attr='min_depth', new_value=new_val, old_value=old_val)
            self.apply_edit(cmd)
            self.update_undo_redo_buttons()

    def erase_disconnected_basins(self, b):
        if self._selected_cell is None:
            return
        i, j, _ = self._selected_cell
        label = self.topo.basintmask.data[j, i]
        affected = np.where(self.topo.basintmask.data != label)
        indices = list(zip(affected[0], affected[1]))
        if not indices:
            return
        old_values = [self.topo.depth.data[jj, ii] for jj, ii in indices]
        new_values = [0] * len(indices)
        cmd = DepthEditCommand(self.topo, indices, new_values, old_values=old_values)
        self.apply_edit(cmd)
        self.update_undo_redo_buttons()
    
    def erase_selected_basin(self, b):
        if self._selected_cell is None:
            return
        i, j, _ = self._selected_cell
        label = self.topo.basintmask.data[j, i]
        affected = np.where(self.topo.basintmask.data == label)
        indices = list(zip(affected[0], affected[1]))
        if not indices:
            return
        old_values = [self.topo.depth.data[jj, ii] for jj, ii in indices]
        new_values = [0] * len(indices)
        cmd = DepthEditCommand(self.topo, indices, new_values, old_values=old_values)
        self.apply_edit(cmd)
        self.update_undo_redo_buttons()

    def on_depth_change(self, change):
        if self._selected_cell is None:
            return
        i, j, _ = self._selected_cell
        old_val = self.topo.depth.data[j, i]
        new_val = change['new']
        if old_val == new_val:
            return
        cmd = DepthEditCommand(self.topo, [(j, i)], [new_val], old_values=[old_val])
        self.apply_edit(cmd)
        self.update_undo_redo_buttons()

    def on_load_button_clicked(self, b):
        val = self._commit_dropdown.value
        if not val:
            print("No commit selected.")
            return
        commit_sha, file_path = val
        self.load_commit(commit_sha=commit_sha, file_path=file_path)
        
    # --- Git Callbacks ---

    def on_git_commit(self, b):
        msg = self._git_commit_msg.value.strip()
        if not msg:
            print("Please enter a Git commit message.")
            return
        name = self._snapshot_name.value.strip()
        if not name:
            print("Please enter a snapshot name to commit.")
            return

        # Save snapshot before commit
        self.command_manager.save_commit(name)
        snapshot_path = os.path.join(self.SNAPSHOT_DIR, f"{name}.json")
        result = git_snapshot_action('commit', CROCODASH_REPO_ROOT, file_path=snapshot_path, commit_msg=msg)
        print(result)
        self.refresh_commit_dropdown()

    def on_git_create_branch(self, b):
        name = self._git_branch_name.value.strip()
        if not name:
            print("Please enter a branch name.")
            return
        try:
            branch = git_create_branch_and_switch(name, CROCODASH_REPO_ROOT)
            print(f"Created and switched to branch '{branch}'.")
            self._git_branch_dropdown.options = git_list_branches(CROCODASH_REPO_ROOT)
            self._git_branch_dropdown.value = git_get_current_branch(CROCODASH_REPO_ROOT)
            self._git_merge_source_dropdown.options = git_list_branches(CROCODASH_REPO_ROOT)
        except Exception as e:
            print(f"Error creating branch: {str(e)}")


    def on_git_delete_branch(self, b):
        name = self._git_branch_name.value.strip()
        if not name:
            print("Please enter the branch name to delete.")
            return
        try:
            current = git_get_current_branch(CROCODASH_REPO_ROOT)
            if current == name:
                print(f"Cannot delete the currently checked-out branch '{name}'.")
                return
            git_delete_branch_and_switch(name, CROCODASH_REPO_ROOT)
            print(f"Deleted branch '{name}'.")
            self._git_branch_dropdown.options = git_list_branches(CROCODASH_REPO_ROOT)
            self._git_branch_dropdown.value = git_get_current_branch(CROCODASH_REPO_ROOT)
            self._git_merge_source_dropdown.options = git_list_branches(CROCODASH_REPO_ROOT)
        except Exception as e:
            print(f"Error deleting branch: {str(e)}")

    def on_git_checkout(self, b):
        target = self._git_branch_dropdown.value
        if not target:
            print("Please select a branch to checkout.")
            return
        try:
            rel_snapshot_dir = os.path.relpath(os.path.abspath(self.SNAPSHOT_DIR), CROCODASH_REPO_ROOT)
            success, untracked, changed, staged = git_safe_checkout_branch(CROCODASH_REPO_ROOT, target, rel_snapshot_dir)
            if not success:
                print("Cannot checkout: You have unsaved or uncommitted changes in 'snapshots'. Please save and commit them before switching branches.")
                if untracked:
                    print("Untracked files:", untracked)
                if changed:
                    print("Unstaged changes:", changed)
                if staged:
                    print("Staged but uncommitted:", staged)
                return
            print(f"Checked out to branch '{target}'.")

            self._git_branch_dropdown.options = git_list_branches(CROCODASH_REPO_ROOT)
            self._git_branch_dropdown.value = git_get_current_branch(CROCODASH_REPO_ROOT)
            self._git_merge_source_dropdown.options = git_list_branches(CROCODASH_REPO_ROOT)

            snapshots = [f for f in os.listdir(self.SNAPSHOT_DIR) if f.endswith('.json') and not f.startswith('golden_')]
            if snapshots:
                snapshots.sort(key=lambda f: os.path.getmtime(os.path.join(self.SNAPSHOT_DIR, f)), reverse=True)
                latest_snapshot = snapshots[0]
                latest_name = latest_snapshot.replace(".json", "")
                self._snapshot_name.value = latest_name
                self.load_commit()
                print(f"Loaded latest snapshot '{latest_name}' from new branch.")
            else:
                # No user snapshots, load golden
                topo_id = self.get_topo_id()
                grid_name = topo_id['grid_name']
                shape = topo_id['shape']
                shape_str = f"{shape[0]}x{shape[1]}"
                golden_name = f"golden_{grid_name}_{shape_str}"
                self._snapshot_name.value = golden_name
                self.load_commit()
                print(f"No user snapshots found, loaded golden snapshot '{golden_name}'.")

        except Exception as e:
            print(f"Error checking out branch: {str(e)}")

    def on_git_merge(self, b):
        source = self._git_merge_source_dropdown.value
        if not source:
            print("Select a branch to merge from.")
            return
        success, msg = git_merge_branch(CROCODASH_REPO_ROOT, source)
        print(msg)
        self._git_branch_dropdown.options = git_list_branches(CROCODASH_REPO_ROOT)
        self._git_branch_dropdown.value = git_get_current_branch(CROCODASH_REPO_ROOT)
        self._git_merge_source_dropdown.options = git_list_branches(CROCODASH_REPO_ROOT)
