import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import ipywidgets as widgets
from matplotlib.ticker import MaxNLocator
from edit_command import CellEditCommand, ScalarEditCommand, COMMAND_REGISTRY
from history_manager import EditHistory

class SetDepthCommand(CellEditCommand):

    def _get_value(self, topo, j, i):
        return topo.depth.data[j, i]
    
    def _set_value(self, topo, j, i, value):
        topo.depth.data[j, i] = value

class SetMinDepthCommand(ScalarEditCommand):
    pass

class EraseSelectedBasinCommand(SetDepthCommand):
    # We are just setting the depth of many indices to zero.
    pass

class EraseDisconnectedBasinsCommand(SetDepthCommand):
    # We are just setting the depth of many indices to zero.
    pass

class TopoEditor(widgets.HBox):
    
    def __init__(self, topo, build_ui=True):
        self.topo = topo
        self.ny = self.topo.depth.data.shape[0]
        self.nx = self.topo.depth.data.shape[1]

        self.history = EditHistory(domain_id_func=self.get_topo_id)

        self._selected_cell = None

        # Save original state for resetting
        self._original_depth = self.topo.depth.data.copy()
        self._original_min_depth = self.topo.min_depth

        # Directory for saving snapshots
        self.SNAPSHOT_DIR = "topo_snapshots"

        # Ensure golden/original topo exists for resets
        self._ensure_golden_topo()

        if build_ui:
            # Setup UI controls, plot, and observers
            self.construct_control_panel()
            self.construct_interactive_plot()
            self.construct_observances()
            self.initialize_history()
            super().__init__([self._control_panel, self._interactive_plot])
        else:
            super().__init__([])


    def initialize_history(self):
        self.history.load_histories(COMMAND_REGISTRY)
        self.history.replay(self.topo)
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
        golden_name = f"golden_{grid_name}_{shape_str}"
        try:
            self.history.load_snapshot(golden_name, COMMAND_REGISTRY)
        except FileNotFoundError:
            self.history.save_snapshot(golden_name)
    
    def reset_topo(self):
        topo_id = self.get_topo_id()
        grid_name = topo_id['grid_name']
        shape = topo_id['shape']
        shape_str = f"{shape[0]}x{shape[1]}"
        golden_name = f"golden_{grid_name}_{shape_str}"
        self.history.load_snapshot(golden_name, COMMAND_REGISTRY)
        self.history.replay(self.topo)
        self.trigger_refresh()
    
    def save_snapshot(self, _btn=None):
        name = self._snapshot_name.value.strip()
        if not name:
            print("Enter a name!")
            return
        self.history.save_snapshot(name)
        print(f"Saved snapshot '{name}'.")

    def load_snapshot(self, _btn=None):
        name = self._snapshot_name.value.strip()
        if not name:
            print("Enter the name of a snapshot to load!")
            return
        try:
            self.history.load_snapshot(name, COMMAND_REGISTRY)
            self.history.replay(self.topo)
            self.update_undo_redo_buttons()
            self.trigger_refresh()
            print(f"Loaded snapshot '{name}'.")
        except FileNotFoundError:
            print(f"No snapshot found with name '{name}'.")

    def apply_edit(self, cmd, record_history=True):
        cmd.execute(self.topo)
        if record_history:
            self.history.push(cmd)
            self.history.save_histories()
        self.update_undo_redo_buttons()
        self.trigger_refresh()

    def undo_last_edit(self, b=None):
        self.history.undo(self.topo)
        self.history.save_histories()
        self.update_undo_redo_buttons()
        self.trigger_refresh()
    
    def redo_last_edit(self, b=None):
        self.history.redo(self.topo)
        self.history.save_histories()
        self.update_undo_redo_buttons()
        self.trigger_refresh()

    def on_reset(self, b=None):
        self.reset_topo()
        self.history.save_histories()
        self.update_undo_redo_buttons()
        print("Topo reset to original state.")

    def update_undo_redo_buttons(self):
        if hasattr(self, "_undo_button"):
            self._undo_button.disabled = not bool(self.history._undo_history)
        if hasattr(self, "_redo_button"):
            self._redo_button.disabled = not bool(self.history._redo_history)

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

        self._display_mode_toggle.observe(
            self.refresh_display_mode,
            names='value',
            type='change'
        )

        def on_double_click(event):
            if event.dblclick:
                y, x = event.ydata, event.xdata
                j, i = self.topo._grid.get_indices(y, x)
                self._select_cell(i, j)
                
        self.fig.canvas.mpl_connect('button_press_event', on_double_click)

        def on_min_depth_change(change):
            old_val = self.topo.min_depth
            new_val = change['new']
            if old_val != new_val:
                cmd = SetMinDepthCommand(attr='min_depth', new_value=new_val, old_value=old_val)
                self.apply_edit(cmd)
                self.update_undo_redo_buttons()
        
        self._min_depth_specifier.observe(
            on_min_depth_change,
            names='value',
            type='change'
        )

        def erase_disconnected_basins(b):
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
            cmd = EraseDisconnectedBasinsCommand(indices, new_values, old_values=old_values)
            self.apply_edit(cmd)
            self.update_undo_redo_buttons()

        self._basin_specifier_toggle.on_click(erase_disconnected_basins)
        
        def erase_selected_basin(b):
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
            cmd = EraseSelectedBasinCommand(indices, new_values, old_values=old_values)
            self.apply_edit(cmd)
            self.update_undo_redo_buttons()
        
        self._basin_specifier_delete_selected.on_click(erase_selected_basin)

        def on_depth_change(change):
            if self._selected_cell is None:
                return
            i, j, _ = self._selected_cell
            old_val = self.topo.depth.data[j, i]
            new_val = change['new']
            if old_val == new_val:
                return
            cmd = SetDepthCommand([(j, i)], [new_val], old_values=[old_val])
            self.apply_edit(cmd)
            self.update_undo_redo_buttons()
       
        self._depth_specifier.observe(
            on_depth_change,
            names='value',
            type='change'
        )

        # Callback connections
        self._undo_button.on_click(self.undo_last_edit)
        self._redo_button.on_click(self.redo_last_edit)
        self._reset_button.on_click(self.on_reset)
        self._save_button.on_click(self.save_snapshot)
        self._load_button.on_click(self.load_snapshot)
