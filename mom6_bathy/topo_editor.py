import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import ipywidgets as widgets
from matplotlib.ticker import MaxNLocator

class TopoEditor(widgets.HBox):
    
    def __init__(self, topo):
        
        self.topo = topo
        self.ny = self.topo.depth.data.shape[0]
        self.nx = self.topo.depth.data.shape[1]

        self._field_mode = 'depth' # 'depth' or 'mask'
        self._selected_cell = None # none or (i, j, patch)

        self.construct_control_panel()
        self.construct_interactive_plot()
        self.construct_observances()

        super().__init__([
            self._control_panel,
            self._interactive_plot,
        ])

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
            self._basin_specifier_delete_selected
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
            self.topo.min_depth = change['new']
            self.trigger_refresh()

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
                ocean_mask_changed = np.where(self.topo.basintmask == self.topo.basintmask[j,i], 1, 0)
                self.topo.depth = self.topo.depth.where(ocean_mask_changed != 0, 0)
                self.trigger_refresh()

        self._basin_specifier_toggle.on_click(erase_disconnected_basins)
        
        def erase_selected_basin(b):
            if self._selected_cell is not None:
                
                i, j, _ = self._selected_cell
                ocean_mask_changed = np.where(self.topo.basintmask == self.topo.basintmask[j,i], 1, 0)
                self.topo.depth = np.where(ocean_mask_changed == 1, 0, self.topo.depth)
                self.trigger_refresh()
                
        self._basin_specifier_delete_selected.on_click(erase_selected_basin)
        
        def on_depth_change(change):
            if self._selected_cell is not None:
                i, j, _ = self._selected_cell
                self.topo.depth.data[j, i] = change['new']
                self.trigger_refresh()

        self._depth_specifier.observe(
            on_depth_change,
            names='value',
            type='change'
        )

