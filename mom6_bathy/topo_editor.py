import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import ipywidgets as widgets
import os
import json
from matplotlib.ticker import MaxNLocator
from datetime import datetime
 

class TopoEditor(widgets.HBox):
    
    def __init__(self, topo):
        
        self.topo = topo
        self.ny = self.topo.depth.data.shape[0]
        self.nx = self.topo.depth.data.shape[1]

        self._ensure_golden_topo()
        
        # State tracking
        self._undo_history = []
        self._redo_history = []
        self._field_mode = 'depth' # 'depth' or 'mask'
        self._selected_cell = None # none or (i, j, patch)

        # Reset references
        self._original_depth = self.topo.depth.data.copy()
        self._original_min_depth = self.topo.min_depth

        # Directory
        self.SNAPSHOT_DIR = "topo_snapshots"

        # Build UI
        self.construct_control_panel()
        self.construct_interactive_plot()
        self.construct_observances()

        # History
        self.initialize_history()

        super().__init__([
            self._control_panel,
            self._interactive_plot,
        ])

    def initialize_history(self):

        # Initialize edit history system and apply changes to current state
        self.load_histories()
        self.replay_edit_history()

        # Button states
        self._undo_button.disabled = not bool(self._undo_history)
        self._redo_button.disabled = not bool(self._redo_history)
        self._save_button.on_click(self.save_snapshot)
        self._load_button.on_click(self.load_snapshot)

    def _ensure_golden_topo(self):
        
        # Ensure original topo upon reset
        topo_id = self.get_topo_id()
        grid_name = topo_id['grid_name']
        shape = topo_id['shape']
        shape_str = f"{shape[0]}x{shape[1]}"
        golden_dir = "original_topo"
        os.makedirs(golden_dir, exist_ok=True)
        golden_topo_path = os.path.join(golden_dir, f"golden_topo_{grid_name}_{shape_str}.npy")
        golden_min_depth_path = os.path.join(golden_dir, f"golden_min_depth_{grid_name}_{shape_str}.json")
        if not os.path.exists(golden_topo_path):
            np.save(golden_topo_path, self.topo.depth.data)
            with open(golden_min_depth_path, "w") as f:
                json.dump({"min_depth": float(self.topo.min_depth)}, f)

    def reset_topo(self):
        # Compose the path based on domain
        topo_id = self.get_topo_id()
        grid_name = topo_id['grid_name']
        shape = topo_id['shape']
        shape_str = f"{shape[0]}x{shape[1]}"
        golden_dir = "original_topo"
        golden_topo_path = os.path.join(golden_dir, f"golden_topo_{grid_name}_{shape_str}.npy")
        golden_min_depth_path = os.path.join(golden_dir, f"golden_min_depth_{grid_name}_{shape_str}.json")
        
        if os.path.exists(golden_topo_path):
            self.topo.depth.data[:] = np.load(golden_topo_path)
            if os.path.exists(golden_min_depth_path):
                with open(golden_min_depth_path, "r") as f:
                    d = json.load(f)
                    self.topo.min_depth = d.get("min_depth", self.topo.min_depth)
            print(f"Topo reset to golden/original topo for {grid_name} {shape_str} from disk.")
        else:
            # Fallback to in-memory
            self.topo.depth.data[:] = np.copy(self._original_depth)
            self.topo.min_depth = self._original_min_depth
            print("Topo reset to first-in-memory state.")
        # Change min_depth value in text box back to default
        self._min_depth_specifier.value = self.topo.min_depth
        self.trigger_refresh()

    def get_topo_id(self):

        # Get a unique identifier for edit state based on name and shape
        grid = self.topo._grid
        grid_name = getattr(grid, "name", getattr(grid, "_name", None))
        shape = [int(v) for v in self.topo.depth.data.shape]
        return {"grid_name": grid_name, "shape": shape}

    def save_snapshot(self, _btn=None):

        # Save the current editing state (undo and redo histories)
        name = self._snapshot_name.value.strip()
        if not name:
            print("Enter a name!")
            return
        data = {
            'topo_id': self.get_topo_id(),
            'undo_history': self._undo_history,
            'redo_history': self._redo_history
        }

        # Create snapshot directory if it doesn't already exist
        if not os.path.exists(self.SNAPSHOT_DIR):
            os.makedirs(self.SNAPSHOT_DIR)
        fname = os.path.join(self.SNAPSHOT_DIR, f"{name}.json")
        with open(fname, 'w') as f:
            json.dump(data, f, default=str)
        print(f"Saved snapshot '{name}'.")

    def load_snapshot(self, _btn=None):

        # Load previously saved editing snapshot from a JSON file
        name = self._snapshot_name.value.strip()
        if not name:
            print("Enter the name of a snapshot to load!")
            return
        fname = os.path.join(self.SNAPSHOT_DIR, f"{name}.json")
        if not os.path.exists(fname):
            print(f"No snapshot found with name '{name}'.")
            return
        with open(fname, 'r') as f:
            data = json.load(f)
            if data['topo_id'] != self.get_topo_id():
                print("Error: Snapshot is for a different domain.")
                return
            self._undo_history = data['undo_history']
            self._redo_history = data['redo_history']

            # Reset topography to original state then applies changes
            self.reset_topo()  
            self.replay_edit_history()
            self._undo_button.disabled = not bool(self._undo_history)
            self._redo_button.disabled = not bool(self._redo_history)
            self.save_histories()
            print(f"Loaded snapshot '{name}'.")

    def get_history_path(self):

        # Generate and return file path of topography edit history for current domain
        topo_id = self.get_topo_id()
        grid_name = topo_id["grid_name"]
        shape = topo_id["shape"]
        shape_str = f"{shape[0]}x{shape[1]}"
        return f"topo_histories/edit_history_{grid_name}_{shape_str}.json"

    def save_histories(self, filepath=None):
        if filepath is None:
            filepath = self.get_history_path()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save undo and redo histories
        with open(filepath, "w") as f:
            json.dump({
                "topo_id": self.get_topo_id(),
                "undo_history": self._undo_history,
                "redo_history": self._redo_history
            }, f, default=str)

    def load_histories(self, filepath=None):
        if filepath is None:
            filepath = self.get_history_path()

        # Load the undo and redo histories from the correct topo_id
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                file_topo_id = data.get("topo_id")
                if file_topo_id is not None and file_topo_id != self.get_topo_id():
                    print("Edit history file does not match current domain! (this should never happen)")
                    self._undo_history = []
                    self._redo_history = []
                else:
                    self._undo_history = data.get("undo_history", [])
                    self._redo_history = data.get("redo_history", [])
        else:
            self._undo_history = []
            self._redo_history = []

    def replay_edit_history(self):

        # Reapply all edits from the undo history to the current state
        for edit in self._undo_history:
            self.apply_edit(edit, record_history=False)
        self.trigger_refresh()

    def apply_edit(self, edit, record_history=True):

        # Apply a single edit operation to the object's state based on the action
        action = edit['action']
        if action == 'depth_change':
            i, j = edit['i'], edit['j']
            self.topo.depth.data[j, i] = edit['new_value']
        elif action == 'erase_disconnected_basins' or action == 'erase_selected_basin':
            indices = edit['affected_indices']
            for (j, i) in indices:
                self.topo.depth.data[j, i] = 0
        elif action == 'min_depth_change':
            self.topo.min_depth = edit['new_value']
        # Additional actions...
        if record_history:
            self._undo_history.append(edit)
            self._redo_history.clear()
            if hasattr(self, '_redo_button'):
                self._redo_button.disabled = True

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
            style={'description_width': '100px', 'button_width': '90px'}
        )

        self._selected_cell_label = widgets.Label("Selected cell: None (double click on canvas to select a cell).")

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
        self._save_button = widgets.Button(description='Save State', layout={'width': '44%'})
        self._load_button = widgets.Button(description='Load State', layout={'width': '44%'})

        self._control_panel = widgets.VBox([
            widgets.HTML("<h2>Topo Editor</h2>"),
            widgets.HTML("<hr><h3>Display</h3>"),
            self._display_mode_toggle,
            widgets.HTML("<hr><h3>Global Settings</h3>"),
            self._min_depth_specifier,
            widgets.HTML("<hr><h3>Cell Editing</h3>"),
            self._selected_cell_label,
            self._depth_specifier,
            widgets.HTML("<hr><h3>Basin Selector</h3>"),
            self._basin_specifier,
            self._basin_specifier_toggle,
            self._basin_specifier_delete_selected,
            widgets.HTML("<hr><h3>Edit History</h3>"),
            widgets.HBox([self._undo_button, self._redo_button, self._reset_button]),
            widgets.HTML("<hr><h3>Snapshots</h3>"),
            widgets.HBox([self._snapshot_name]),
            widgets.HBox([self._save_button, self._load_button]),
          ], layout= {'width': '30%', 'height': '100%'})


    def refresh_display_mode(self, change):

        mode = change['new']

        if mode == 'depth':
            self.im.set_clim(vmin = self.topo.min_depth, vmax = self.topo.depth.max(skipna=True).item())
            self.im.set_array(self.topo.depth.data)
            self.im.set_clim(vmin = self.topo.min_depth, vmax = self.topo.depth.max(skipna=True).item()) # For some reason, this needs to be set twice to get the correct minimum bound
            
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

        # If a cell was previously selected, remove the patch
        if self._selected_cell is not None:
            self._selected_cell[2].remove()
        # Create a patch to highlight the selected cell
        vertices = np.array([
            [self.topo._grid.qlon.data[j,i], self.topo._grid.qlat.data[j,i]],
            [self.topo._grid.qlon.data[j,i+1], self.topo._grid.qlat.data[j,i+1]],
            [self.topo._grid.qlon.data[j+1,i+1], self.topo._grid.qlat.data[j+1,i+1]],
            [self.topo._grid.qlon.data[j+1,i], self.topo._grid.qlat.data[j+1, i]],
        ])
        polygon = patches.Polygon(vertices, edgecolor='r', facecolor='none', alpha=0.8, linewidth=2, label='Selected cell')
        self.ax.add_patch(polygon)

        # Update the selected cell label
        self._selected_cell_label.value = f"Selected cell: {i}, {j}"

        # store the selected cell
        self._selected_cell = (i, j, polygon)
        
        # Enable the depth specifier
        self._depth_specifier.disabled = False
        self._depth_specifier.value = self.topo.depth.data[j, i]

        # Update Basin Specifier
        label = self.topo.basintmask.data[j,i]
        self._basin_specifier.value = "Basin Label Number: " + str(label)

        # If not land, manifest button
        if label != 0:
            self._basin_specifier_toggle.disabled = False
            self._basin_specifier_delete_selected.disabled = False
        else:
            self._basin_specifier_toggle.disabled = True
            self._basin_specifier_delete_selected.disabled = True


    def construct_observances(self):

        def on_double_click(event):

            if event.dblclick:

                # Get the coordinates of the click and retrieve the cell indices
                y, x = event.ydata, event.xdata
                j, i = self.topo._grid.get_indices(y, x)
                self._select_cell(i, j)
                
        self.fig.canvas.mpl_connect('button_press_event', on_double_click)

        def on_min_depth_change(change):
            old_val = self.topo.min_depth
            new_val = change['new']
            self._undo_history.append({
                'action': 'min_depth_change',
                'old_value': float(old_val),
                'new_value': float(new_val),
                'timestamp': datetime.now().isoformat(),
            })
            self._undo_button.disabled = False
            self._redo_history.clear()
            self._redo_button.disabled = True
            self.topo.min_depth = new_val
            self.trigger_refresh()
            self.save_histories()

        self._min_depth_specifier.observe(
            on_min_depth_change,
            names='value',
            type='change'
        )

        self._display_mode_toggle.observe(
            self.refresh_display_mode,
            names='value',
            type='change'
        )

        def erase_disconnected_basins(b):
            if self._selected_cell is not None:
                i, j, _ = self._selected_cell
                label = self.topo.basintmask.data[j, i]
                ocean_mask_changed = np.where(self.topo.basintmask == label, 1, 0)

                affected = np.where(ocean_mask_changed == 0)
                old_depths = self.topo.depth.data[affected]
                self._undo_history.append({
                    'action': 'erase_disconnected_basins',
                    'basin_label': int(label),
                    'retained_cell': (int(i), int(j)),
                    'affected_indices': list(zip(affected[0].tolist(), affected[1].tolist())),
                    'old_depths': old_depths.tolist(),
                    'timestamp': datetime.now().isoformat(),
                })
                self._undo_button.disabled = False
                self._redo_history.clear()
                self._redo_button.disabled = True
                self.topo.depth = np.where(ocean_mask_changed == 0, 0, self.topo.depth)
                self.trigger_refresh()
                self.save_histories()

        self._basin_specifier_toggle.on_click(erase_disconnected_basins)
        
        def erase_selected_basin(b):
            if self._selected_cell is not None:
                i, j, _ = self._selected_cell
                label = self.topo.basintmask.data[j, i]
                ocean_mask_changed = np.where(self.topo.basintmask == label, 1, 0)
                affected = np.where(ocean_mask_changed == 1)
                old_depths = self.topo.depth.data[affected]
                self._undo_history.append({
                    'action': 'erase_selected_basin',
                    'basin_label': int(label),
                    'selected_cell': (int(i), int(j)),
                    'affected_indices': list(zip(affected[0].tolist(), affected[1].tolist())),
                    'old_depths': old_depths.tolist(),
                    'timestamp': datetime.now().isoformat(),
                })
                self._undo_button.disabled = False
                self._redo_history.clear()
                self._redo_button.disabled = True
                self.topo.depth = np.where(ocean_mask_changed == 1, 0, self.topo.depth)
                self.trigger_refresh()
                self.save_histories()
                
        self._basin_specifier_delete_selected.on_click(erase_selected_basin)
        
        def on_depth_change(change):
            if self._selected_cell is not None:
                i, j, _ = self._selected_cell
                old_val = self.topo.depth.data[j, i]
                new_val = change['new']
                self._undo_history.append({
                    'action': 'depth_change',
                    'i': i,
                    'j': j,
                    'old_value': old_val,
                    'new_value': new_val,
                    "timestamp": datetime.now().isoformat(),
                })
                self._undo_button.disabled = False
                self._redo_history.clear()
                self._redo_button.disabled = True
                self.topo.depth.data[j, i] = new_val
                self.trigger_refresh()
                self.save_histories()

        self._depth_specifier.observe(
            on_depth_change,
            names='value',
            type='change'
        )

        def undo_last_edit(b):
            if not self._undo_history:
                self._undo_button.disabled = True
                return
            last_action = self._undo_history.pop()

            if last_action['action'] == 'depth_change':
                i, j = last_action['i'], last_action['j']
                self.topo.depth.data[j, i] = last_action['old_value']
            elif last_action['action'] == 'erase_disconnected_basins':
                indices = last_action['affected_indices']
                old_depths = last_action['old_depths']
                depth_arr = self.topo.depth.data.copy()
                for (j, i), old_val in zip(indices, old_depths):
                    depth_arr[j, i] = old_val
                self.topo.depth = depth_arr 
            elif last_action['action'] == 'erase_selected_basin':
                indices = last_action['affected_indices']
                old_depths = last_action['old_depths']
                depth_arr = self.topo.depth.data.copy()
                for (j, i), old_val in zip(indices, old_depths):
                    depth_arr[j, i] = old_val
                self.topo.depth = depth_arr 
            elif last_action['action'] == 'min_depth_change':
                self.topo.min_depth = last_action['old_value']
            else:
                print(f"Undo not implemented for action: {last_action['action']}")

            # Add the undone action to redo history
            self._redo_history.append(last_action)

            self.save_histories()

            # Disable if now empty
            if not self._undo_history:
                self._undo_button.disabled = True
            # Enable redo button if available
            self._redo_button.disabled = not bool(self._redo_history)

            # Refresh display
            self.trigger_refresh()

        self._undo_button.on_click(undo_last_edit)

        def redo_last_edit(b):
            if not self._redo_history:
                self._redo_button.disabled = True
                return
            next_action = self._redo_history.pop()

            self.apply_edit(next_action, record_history=False)
            self._undo_history.append(next_action)

            self.save_histories()

            self._undo_button.disabled = not bool(self._undo_history)
            self._redo_button.disabled = not bool(self._redo_history)

            self.trigger_refresh()
        
        self._redo_button.on_click(redo_last_edit)

        def on_reset(b):
            # Restore to original values
            self.reset_topo()
            # Clear histories to prevent reapplying old edits
            self._undo_history.clear()
            self._redo_history.clear()
            self._undo_button.disabled = True
            self._redo_button.disabled = True
            self.save_histories()
            self.trigger_refresh()
            print("Topo reset to original state.")


        self._reset_button.on_click(on_reset)

        self._save_button.on_click(self.save_snapshot)
        self._load_button.on_click(self.load_snapshot)
