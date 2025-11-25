import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import ipywidgets as widgets
import os
import json
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator
from mom6_bathy.command_manager import TopoCommandManager
from mom6_bathy.edit_command import *
from mom6_bathy.git_utils import *


class TopoEditor(widgets.HBox):
    def __init__(self, topo, build_ui=True):
        self.topo = topo
        self.ny = self.topo.depth.data.shape[0]
        self.nx = self.topo.depth.data.shape[1]

        # Load TCM Logic
        self.tcm = TopoCommandManager("dummy", topo, COMMAND_REGISTRY)

        # --- Command Manager ---
        self.current_branch = get_current_branch(self.repo_root)
        self._selected_cell = None
        self._original_depth = np.array(self.topo.depth.data)
        self._original_min_depth = self.topo.min_depth

        # --- Build UI controls, plot, and observers ---
        self.construct_control_panel()
        self.construct_interactive_plot()
        self.construct_observances()
        self.initialize_history()
        self.refresh_tag_dropdown()

        # --- Initialize the widget layout ---
        super().__init__([self._control_panel, self._interactive_plot])

    def initialize_history(self):
        """Initialize the command manager's history and update button states."""
        self.update_undo_redo_buttons()

    def apply_edit(self, cmd):
        """Apply an edit command, update the UI, and autosave the working state."""
        self.command_manager.execute(cmd)
        self.update_undo_redo_buttons()
        self.trigger_refresh()

    def undo_last_edit(self, b=None):
        """Undo the last edit command and update the UI."""
        self.command_manager.undo()
        self.update_undo_redo_buttons()
        self._min_depth_specifier.value = self.topo.min_depth
        self.trigger_refresh()

    def redo_last_edit(self, b=None):
        """Redo the last undone edit command and update the UI."""
        self.command_manager.redo()
        self.update_undo_redo_buttons()
        self._min_depth_specifier.value = self.topo.min_depth
        self.trigger_refresh()

    def reset(self, b=None):
        """Reset the topo to its original state and update the UI."""
        self.command_manager.reset(
            self.topo,
            self._original_depth,
            self._original_min_depth,
            self.topo.get_domain_id,
            min_depth_specifier=self._min_depth_specifier,
            trigger_refresh=self.trigger_refresh,
        )
        self.update_undo_redo_buttons()
        print("Topo reset to original state.")

    def update_undo_redo_buttons(self):
        """Enable or disable the undo/redo buttons based on command history."""
        if hasattr(self, "_undo_button"):
            self._undo_button.disabled = not (
                hasattr(self.command_manager, "_undo_history")
                and bool(self.command_manager._undo_history)
            )
        if hasattr(self, "_redo_button"):
            self._redo_button.disabled = not (
                hasattr(self.command_manager, "_redo_history")
                and bool(self.command_manager._redo_history)
            )

    def refresh_tag_dropdown(self):
        """Refresh the list of available commits/snapshots in the dropdown menu."""
        current_branch = get_current_branch(self.repo_root)
        self.update_commit_details()

    def construct_interactive_plot(self):
        """
        Construct the interactive matplotlib plot for the topography editor.

        This sets up the main map display, colorbar, and coordinate formatting.
        The plot is embedded in a widget for use in the Jupyter interface.
        """
        # Close any existing figure to avoid memory leaks
        if hasattr(self, "fig") and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        plt.ioff()  # Turn off interactive mode for setup

        # Create the figure and axis with Cartopy projection
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set map extent based on grid longitude/latitude
        lon_min = float(self.topo._grid.qlon.data.min())
        lon_max = float(self.topo._grid.qlon.data.max())
        lat_min = float(self.topo._grid.qlat.data.min())
        lat_max = float(self.topo._grid.qlat.data.max())
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Custom coordinate formatter for mouse hover
        def format_coord(x, y):
            j, i = self.topo._grid.get_indices(y, x)
            return f"x={x:.2f}, y={y:.2f}, i={i}, j={j} depth={self.topo.depth.data[j, i]:.2f}"

        self.ax.format_coord = format_coord

        # Set up colormap and plot the depth field
        self.cmap = plt.get_cmap("viridis")
        self.cmap.set_under("w")
        self.im = self.ax.pcolormesh(
            self.topo._grid.qlon.data,
            self.topo._grid.qlat.data,
            self.topo.depth.data,
            vmin=self.topo.min_depth,
            cmap=self.cmap,
            transform=ccrs.PlateCarree(),
        )

        # Axis labels and title
        self.ax.set_title("Double click on a cell to change its depth.")
        self.ax.set_xlabel(
            f'x ({self.topo._grid.qlon.attrs.get("units", "degrees_east")})'
        )
        self.ax.set_ylabel(
            f'y ({self.topo._grid.qlat.attrs.get("units", "degrees_north")})'
        )

        # Add colorbar for depth
        self.cbar = self.fig.colorbar(
            self.im, ax=self.ax, orientation="vertical", pad=0.02
        )
        self.cbar.set_label(f"Depth ({self.topo.depth.units})")
        self.cbar.set_ticks(MaxNLocator(integer=True))

        # Enable toolbar and layout
        self.fig.canvas.toolbar_visible = True
        self.fig.canvas.toolbar_position = "top"
        self.fig.tight_layout()
        plt.ion()  # Restore interactive mode

        # Wrap the figure in a widget for display
        self._interactive_plot = widgets.HBox(
            children=(self.fig.canvas,), layout={"border_left": "1px solid grey"}
        )

    def construct_control_panel(self):
        """
        Construct the control panel widgets for the topography editor.

        This includes controls for display mode, cell editing, undo/redo,
        snapshots, and git/domain management. The controls are grouped
        into logical sections for clarity.
        """
        # --- Display and global settings ---
        self._min_depth_specifier = widgets.BoundedFloatText(
            value=self.topo.min_depth,
            min=-1000.0,
            max=float(np.nanmax(self.topo.depth.data)),
            step=10.0,
            description="Min depth (m):",
            disabled=False,
            layout={"width": "80%"},
            style={"description_width": "auto"},
        )
        self._display_mode_toggle = widgets.ToggleButtons(
            options=["depth", "mask", "basinmask"],
            description="Field:",
            disabled=False,
            tooltips=["Display depth values", "Display mask values", "Display Basins"],
            layout={"width": "90%", "display": "flex"},
            style={"description_width": "40px", "button_width": "85px"},
        )

        # --- Cell editing widgets ---
        self._selected_cell_label = widgets.Label(
            "Selected cell: None (double click to select a cell)."
        )
        self._depth_specifier = widgets.FloatText(
            value=None,
            step=10.0,
            description="Depth (m):",
            disabled=True,
            placeholder="Select a cell first.",
            layout={"width": "80%"},
            style={"description_width": "auto"},
        )

        # --- Basin editing widgets ---
        self._basin_specifier_toggle = widgets.Button(
            description="Erase Disconnected Basins",
            disabled=True,
            layout={"width": "90%", "display": "flex"},
            style={"description_width": "100px"},
        )
        self._basin_specifier_delete_selected = widgets.Button(
            description="Erase Selected Basin",
            disabled=True,
            layout={"width": "90%", "display": "flex"},
            style={"description_width": "100px"},
        )
        self._basin_specifier = widgets.Label(
            value="Basin Label Number: None",
            layout={"width": "80%"},
            style={"description_width": "auto"},
        )

        # --- Undo/Redo/Reset ---
        self._undo_button = widgets.Button(
            description="Undo", disabled=True, layout={"width": "44%"}
        )
        self._redo_button = widgets.Button(
            description="Redo", disabled=True, layout={"width": "44%"}
        )
        self._reset_button = widgets.Button(
            description="Reset", layout={"width": "44%"}, button_style="danger"
        )

        # --- Snapshot controls ---
        self._tag_name = widgets.Text(
            value="",
            placeholder="Enter tag name",
            description="Name:",
            layout={"width": "90%"},
        )
        self._tag_msg = widgets.Text(
            value="",
            placeholder="Enter tag message",
            description="Message:",
            layout={"width": "90%"},
        )
        self._tag_dropdown = widgets.Dropdown(
            options=[], description="Tags:", layout={"width": "90%"}
        )
        self._tag_details = widgets.HTML(
            value="", layout={"width": "90%", "min_height": "2em"}
        )
        self._save_button = widgets.Button(
            description="Save Tag", layout={"width": "44%"}
        )
        self._load_button = widgets.Button(
            description="Load Tag", layout={"width": "44%"}
        )

        self._git_branch_name = widgets.Text(
            value="",
            placeholder="New branch name",
            description="Branch:",
            layout={"width": "90%"},
        )
        self._git_create_branch_button = widgets.Button(
            description="Create Branch", layout={"width": "44%"}
        )
        self._git_delete_branch_button = widgets.Button(
            description="Delete Branch", layout={"width": "44%"}, button_style="danger"
        )
        self._git_branch_dropdown = widgets.Dropdown(
            options=list_branches(self.repo_root),
            description="Checkout:",
            layout={"width": "90%"},
        )
        self._git_checkout_button = widgets.Button(
            description="Checkout", layout={"width": "44%"}
        )
        # --- Group controls into logical sections ---
        display_section = widgets.VBox(
            [
                widgets.HTML("<h3>Display</h3>"),
                self._display_mode_toggle,
            ]
        )
        global_settings_section = widgets.VBox(
            [
                widgets.HTML("<h3>Global Settings</h3>"),
                self._min_depth_specifier,
            ]
        )
        cell_editing_section = widgets.VBox(
            [
                widgets.HTML("<h3>Cell Editing</h3>"),
                self._selected_cell_label,
                self._depth_specifier,
            ]
        )
        basin_section = widgets.VBox(
            [
                widgets.HTML("<h3>Basin Selector</h3>"),
                self._basin_specifier,
                self._basin_specifier_toggle,
                self._basin_specifier_delete_selected,
            ]
        )
        history_section = widgets.VBox(
            [
                widgets.HTML("<h3>Edit History</h3>"),
                widgets.HBox(
                    [self._undo_button, self._redo_button, self._reset_button]
                ),
            ]
        )
        git_section = widgets.VBox(
            [
                # Domain controls
                self._domain_dropdown,
                self._switch_domain_button,
                widgets.HTML("<hr>"),
                # Snapshot controls
                self._tag_name,
                self._tag_msg,
                self._tag_dropdown,
                self._tag_details,
                widgets.HBox([self._save_button, self._load_button]),
                widgets.HTML("<hr>"),
                # Git controls
                self._git_branch_name,
                widgets.HBox(
                    [self._git_create_branch_button, self._git_delete_branch_button]
                ),
                self._git_branch_dropdown,
                self._git_checkout_button,
            ]
        )

        # --- Layout: always-visible controls and advanced accordions ---
        main_controls = widgets.VBox(
            [
                display_section,
                global_settings_section,
                cell_editing_section,
                basin_section,
                history_section,
            ]
        )
        git_accordion = widgets.Accordion(children=[git_section])
        git_accordion.set_title(0, "Git Version Control")
        git_accordion.selected_index = None  # collapsed by default

        # --- Combine everything into the control panel ---
        self._control_panel = widgets.VBox(
            [
                widgets.HTML("<h2>Topo Editor</h2>"),
                main_controls,
                git_accordion,
            ],
            layout={"width": "30%", "height": "100%", "overflow_y": "auto"},
        )

        # Set the current branch in the dropdown if available
        current_branch = get_current_branch(self.repo_root)
        if current_branch in self._git_branch_dropdown.options:
            self._git_branch_dropdown.value

    def refresh_display_mode(self, change):
        """Refresh the display mode of the topography plot based on the selected mode."""
        mode = change["new"]
        if mode == "depth":
            self.im.set_clim(
                vmin=self.topo.min_depth, vmax=float(np.nanmax(self.topo.depth.data))
            )
            self.im.set_array(self.topo.depth.data)
            self.im.set_clim(
                vmin=self.topo.min_depth, vmax=float(np.nanmax(self.topo.depth.data))
            )  # For some reason, this needs to be set twice to get the correct minimum bound
            self.cbar.set_label(f"Depth ({self.topo.depth.units})")
        elif mode == "mask":
            self.im.set_array(self.topo.tmask.data)
            self.im.set_clim((0, 1))
            self.cbar.set_label("Land Mask")
        elif mode == "basinmask":
            self.im.set_array(self.topo.basintmask.data)
            self.im.set_clim((0, self.topo.basintmask.data.max()))
            self.cbar.set_label("Basin Mask")
        else:
            raise ValueError(f"Unknown display mode: {mode}")
        self.fig.canvas.draw_idle()

    def trigger_refresh(self):
        """Trigger a refresh of the interactive plot."""
        self.refresh_display_mode({"new": self._display_mode_toggle.value})

    def _select_cell(self, i, j):
        """Select a cell in the topography grid and update the UI accordingly."""
        # Remove old patch if it exists
        if (
            self._selected_cell is not None
            and len(self._selected_cell) > 2
            and self._selected_cell[2] is not None
            and hasattr(self, "ax")
        ):
            try:
                self._selected_cell[2].remove()
            except Exception:
                pass

        polygon = None
        if hasattr(self, "ax"):
            try:
                qlon = self.topo._grid.qlon.data
                qlat = self.topo._grid.qlat.data
                if (j + 1 < qlon.shape[0]) and (i + 1 < qlon.shape[1]):
                    vertices = np.array(
                        [
                            [qlon[j, i], qlat[j, i]],
                            [qlon[j, i + 1], qlat[j, i + 1]],
                            [qlon[j + 1, i + 1], qlat[j + 1, i + 1]],
                            [qlon[j + 1, i], qlat[j + 1, i]],
                        ]
                    )
                    polygon = patches.Polygon(
                        vertices,
                        edgecolor="r",
                        facecolor="none",
                        alpha=0.8,
                        linewidth=2,
                        label="Selected cell",
                        transform=ccrs.PlateCarree(),
                    )
                    self.ax.add_patch(polygon)
                    self.fig.canvas.draw_idle()
            except Exception as e:
                print(f"Failed to draw polygon patch: {e}")

        self._selected_cell = (i, j, polygon)

        # UI updates
        if hasattr(self, "_selected_cell_label"):
            self._selected_cell_label.value = f"Selected cell: {i}, {j}"
        if hasattr(self, "_depth_specifier"):
            self._depth_specifier.disabled = False
            self._depth_specifier.value = self.topo.depth.data[j, i]
        if hasattr(self, "_basin_specifier"):
            label = self.topo.basintmask.data[j, i]
            self._basin_specifier.value = f"Basin Label Number: {str(label)}"
            if hasattr(self, "_basin_specifier_toggle") and hasattr(
                self, "_basin_specifier_delete_selected"
            ):
                if label != 0:
                    self._basin_specifier_toggle.disabled = False
                    self._basin_specifier_delete_selected.disabled = False
                else:
                    self._basin_specifier_toggle.disabled = True
                    self._basin_specifier_delete_selected.disabled = True

    def construct_observances(self):
        """Attach event observers and callbacks to all interactive widgets and plot elements."""
        # Display mode toggle
        self._display_mode_toggle.observe(
            self.refresh_display_mode, names="value", type="change"
        )

        # Double click event for cell selection on the plot
        self.fig.canvas.mpl_connect("button_press_event", self.on_double_click)

        # Min depth change observer
        self._min_depth_specifier.observe(
            self.on_min_depth_change, names="value", type="change"
        )

        # Basin erase buttons
        self._basin_specifier_toggle.on_click(self.erase_disconnected_basins)
        self._basin_specifier_delete_selected.on_click(self.erase_selected_basin)

        # Depth change observer for selected cell
        self._depth_specifier.observe(
            self.on_depth_change, names="value", type="change"
        )

        # Undo/Redo/Reset buttons
        self._undo_button.on_click(self.undo_last_edit)
        self._redo_button.on_click(self.redo_last_edit)
        self._reset_button.on_click(self.reset)

        # Snapshot controls
        self._save_button.on_click(self.on_save_and_commit)
        self._load_button.on_click(self.on_load_button_clicked)
        self._tag_name.observe(
            lambda change: self.refresh_tag_dropdown(), names="value"
        )
        self._tag_dropdown.observe(self.update_commit_details, names="value")

        # Git/domain controls
        self._git_create_branch_button.on_click(self.on_git_create_branch)
        self._git_delete_branch_button.on_click(self.on_git_delete_branch)
        self._git_checkout_button.on_click(self.on_git_checkout)
        self._display_mode_toggle.observe(
            self.refresh_display_mode, names="value", type="change"
        )

    # --- UI Callback Methods ---

    def on_save_and_commit(self, _btn=None):
        """Save the current state as a snapshot and commit it to the repository."""
        name = self._tag_name.value.strip()
        msg = self._tag_msg.value.strip()
        if not name:
            print("Enter a snapshot name!")
            return
        if not msg:
            print("Enter a snapshot message!")
            return

        self.command_manager.save_commit(name)  # TODO: Save a tag!
        print(f"Saved tag '{name}'.")
        self.refresh_tag_dropdown()
        return

    def on_double_click(self, event):
        """Handle double-click events on the plot to select a cell."""
        if event.dblclick and event.xdata is not None and event.ydata is not None:
            # Convert lon/lat to grid indices
            j, i = self.topo._grid.get_indices(event.ydata, event.xdata)
            if 0 <= i < self.nx and 0 <= j < self.ny:
                self._select_cell(i, j)

    def on_min_depth_change(self, change):
        """Handle changes to the minimum depth specifier."""
        old_val = self.topo.min_depth
        new_val = change["new"]
        if old_val != new_val:
            cmd = MinDepthEditCommand(
                self.topo, attr="min_depth", new_value=new_val, old_value=old_val
            )
            self.apply_edit(cmd)
            self.update_undo_redo_buttons()

    def erase_disconnected_basins(self, b):
        """Erase all disconnected basins in the topography."""
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
        """Erase the basin associated with the currently selected cell."""
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
        """Handle changes to the depth specifier for the selected cell."""
        if self._selected_cell is None:
            return
        i, j, _ = self._selected_cell
        old_val = self.topo.depth.data[j, i]
        new_val = change["new"]
        if old_val == new_val:
            return
        cmd = DepthEditCommand(self.topo, [(j, i)], [new_val], old_values=[old_val])
        self.apply_edit(cmd)
        self.update_undo_redo_buttons()

    def on_load_button_clicked(self, b):
        """Load a tag from the dropdown and update the editor state."""
        val = self._tag_dropdown.value
        if not val:
            print("No commit selected.")
            return
        commit_sha, file_path = val
        snapshot_name = os.path.splitext(os.path.basename(file_path))[0]
        self.reset()
        self.load_commit(name=snapshot_name)
        self.refresh_tag_dropdown()
        # Set dropdown to the just-loaded commit if present
        for label, value in self._tag_dropdown.options:
            if os.path.splitext(os.path.basename(value[1]))[0] == snapshot_name:
                self._tag_dropdown.value = value
                break
        print(f"Loaded tag '{snapshot_name}' for current grid.")

    def on_git_create_branch(self, b):
        """Create a new git branch and switch to it."""
        name = self._git_branch_name.value.strip()
        if not name:
            print("Please enter a branch name.")
            return
        try:
            branch = create_branch_and_switch(name, self.repo_root)
            print(f"Created and switched to branch '{branch}'.")
            self._git_branch_dropdown.options = list_branches(self.repo_root)
            self._git_branch_dropdown.value = get_current_branch(self.repo_root)
            self._git_merge_source_dropdown.options = list_branches(self.repo_root)
        except Exception as e:
            print(f"Error creating branch: {str(e)}")

    def on_git_delete_branch(self, b):
        """Delete the specified git branch."""
        name = self._git_branch_name.value.strip()
        if not name:
            print("Please enter the branch name to delete.")
            return
        try:
            current = get_current_branch(self.repo_root)
            if current == name:
                print(f"Cannot delete the currently checked-out branch '{name}'.")
                return
            delete_branch_and_switch(name, self.repo_root)
            print(f"Deleted branch '{name}'.")
            self._git_branch_dropdown.options = list_branches(self.repo_root)
            self._git_branch_dropdown.value = get_current_branch(self.repo_root)
            self._git_merge_source_dropdown.options = list_branches(self.repo_root)
        except Exception as e:
            print(f"Error deleting branch: {str(e)}")

    def on_git_checkout(self, b):
        """Checkout the specified git branch."""
        target = self._git_branch_dropdown.value
        if not target:
            print("Please select a branch to checkout.")
            return
        try:
            rel_snapshot_dir = os.path.relpath(
                os.path.abspath(self.SNAPSHOT_DIR), self.repo_root
            )
            success, _, _, _ = safe_checkout_branch(
                self.repo_root, target, rel_snapshot_dir
            )
            if not success:
                return
            print(f"Checked out to branch '{target}'.")

            # Update branch dropdowns
            self._git_branch_dropdown.options = list_branches(self.repo_root)
            self._git_branch_dropdown.value = get_current_branch(self.repo_root)
            self._git_merge_source_dropdown.options = list_branches(self.repo_root)

            # --- Reset domain dropdown/options to reflect new branch ---
            self._domain_dropdown.options = self._domain_options

            # Try to keep the same domain selected, or fallback to the first available
            selected = self._domain_dropdown.value
            if selected not in [v for l, v in self._domain_options]:
                if self._domain_options:
                    self._domain_dropdown.value = self._domain_options[0][1]
                    selected = self._domain_dropdown.value
                else:
                    print("No domains found in this branch.")
                    return

            # --- Always reload from original topo for the selected domain ---
            self.on_switch_domain(None)

            # --- Now load the latest user snapshot (if any) on top of the original topo ---
            self.refresh_tag_dropdown()
            user_snapshots = [
                os.path.basename(opt[1][1])
                for opt in self._tag_dropdown.options
                if opt[1][1].endswith(".json")
                and not os.path.basename(opt[1][1]).startswith("original_")
            ]
            if user_snapshots:
                # Find the most recent user snapshot (not original, not autosave/history)
                user_snapshots.sort(
                    key=lambda f: os.path.getmtime(os.path.join(self.SNAPSHOT_DIR, f)),
                    reverse=True,
                )
                latest_snapshot = user_snapshots[0]
                latest_name = latest_snapshot.replace(".json", "")
                self.load_commit(latest_name)
                print(f"Loaded latest snapshot '{latest_name}' from new branch.")
            else:
                print("No user snapshots found, using original topo.")

        except Exception as e:
            print(f"Error checking out branch: {str(e)}")
