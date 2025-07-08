import os
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class GridEditor(widgets.HBox):

    def __init__(self, grid, repo_root=None):
        self.grid = grid
        self.repo_root = repo_root if repo_root is not None else os.getcwd()
        self.grids_dir = os.path.join(self.repo_root, "Grids")
        self._initial_params = {
            "lenx": grid.lenx,
            "leny": grid.leny,
            "resolution": grid.resolution,
            "xstart": grid.xstart,
            "ystart": grid.ystart,
            "name": grid.name
        }

        self.construct_control_panel()
        self.construct_observances()

        # --- Plot ---
        plt.ioff()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        plt.ion()
        self.fig.canvas.layout.width = "100%"
        self.fig.canvas.layout.min_width = "0"
        self.fig.canvas.toolbar_visible = True
        self.fig.canvas.toolbar_position = 'top'

        super().__init__([self._control_panel, self.fig.canvas], layout=widgets.Layout(width="100%", align_items="flex-start"))

        self.refresh_commit_dropdown()
        self.plot_grid()
    
    def construct_control_panel(self):
        self._snapshot_name = widgets.Text(value='', placeholder='Enter grid name', description='Name:', layout={'width': '90%'})
        self._commit_msg = widgets.Text(value='', placeholder='Enter grid message', description='Message:', layout={'width': '90%'})
        self._commit_dropdown = widgets.Dropdown(options=[], description='Grids:', layout={'width': '90%'})
        self._commit_details = widgets.HTML(value="", layout={'width': '90%', 'min_height': '2em'})
        self._save_button = widgets.Button(description='Save Grid', layout={'width': '44%'})
        self._load_button = widgets.Button(description='Load Grid', layout={'width': '44%'})
        self._reset_button = widgets.Button(description='Reset', layout={'width': '100%'}, button_style='danger')

        # Use initial values for slider ranges
        initial_xstart = float(self.grid.xstart)
        initial_ystart = float(self.grid.ystart)

        self._resolution_slider = widgets.FloatSlider(
            value=self.grid.resolution, min=0.01, max=1.0, step=0.01, description="Resolution"
        )
        self._xstart_slider = widgets.FloatSlider(
            value=initial_xstart,
            min=max(initial_xstart - 30, 0),
            max=min(initial_xstart + 30, 360),
            step=0.01,
            description="xstart"
        )
        self._lenx_slider = widgets.FloatSlider(
            value=self.grid.lenx, min=0.01, max=50.0, step=0.01, description="lenx"
        )
        self._ystart_slider = widgets.FloatSlider(
            value=initial_ystart,
            min=max(initial_ystart - 30, -90),
            max=min(initial_ystart + 30, 90),
            step=0.01,
            description="ystart"
        )
        self._leny_slider = widgets.FloatSlider(
            value=self.grid.leny, min=0.01, max=50.0, step=0.01, description="leny"
        )

        controls = widgets.VBox([
            widgets.HTML("<h3>Grid Controls</h3>"),
            self._resolution_slider,
            self._xstart_slider,
            self._lenx_slider,
            self._ystart_slider,
            self._leny_slider,
            widgets.HBox([self._reset_button], layout=widgets.Layout(justify_content="flex-end", width="100%")),
        ], layout=widgets.Layout(width="100%", min_width="200px", max_width="400px", align_items="stretch", overflow_y="auto"))
        
        library_section = widgets.VBox([
            widgets.HTML("<h3>Library</h3>"),
            self._snapshot_name,
            self._commit_msg,
            self._commit_dropdown,
            self._commit_details,
            widgets.HBox([self._save_button, self._load_button]),
        ])

        self._control_panel = widgets.VBox([
            controls,
            library_section,
        ], layout={'width': '45%', 'height': '100%'})

    def construct_observances(self):
        self._save_button.on_click(self.save_grid)
        self._load_button.on_click(self.load_grid)
        self._reset_button.on_click(self.reset_grid)
        self._snapshot_name.observe(lambda change: self.refresh_commit_dropdown(), names='value')
        self._commit_dropdown.observe(self.update_commit_details, names='value')

        for slider in [
            self._resolution_slider,
            self._xstart_slider,
            self._lenx_slider,
            self._ystart_slider,
            self._leny_slider,
        ]:
            slider.observe(self._on_slider_change, names="value")

    def plot_grid(self):
        self.ax.clear()
        self.ax.coastlines(resolution='10m', linewidth=1)
        self.ax.add_feature(cfeature.LAND, facecolor='0.9')
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        n_jq, n_iq = self.grid.qlon.shape
        for i in range(n_iq):
            self.ax.plot(self.grid.qlon[:, i], self.grid.qlat[:, i], color='k', linewidth=0.1, transform=ccrs.PlateCarree())
        for j in range(n_jq):
            self.ax.plot(self.grid.qlon[j, :], self.grid.qlat[j, :], color='k', linewidth=0.1, transform=ccrs.PlateCarree())
        self.ax.set_title("Grid Editor")

        lon_min, lon_max = float(self.grid.qlon.min()), float(self.grid.qlon.max())
        lat_min, lat_max = float(self.grid.qlat.min()), float(self.grid.qlat.max())
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        gl = self.ax.gridlines(draw_labels=True, linewidth=0, color='none')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        self._draw_scale_bar(lon_min, lon_max, lat_min, lat_max)

        self.fig.canvas.draw_idle()
    
    def _draw_scale_bar(self, lon_min, lon_max, lat_min, lat_max):
        """Draw a fixed-length scale bar with dynamic label using geometric calculation."""
        try:
            frac = 0.2
            bar_lat = lat_min + 0.05 * (lat_max - lat_min)
            bar_lon_start = lon_min + 0.05 * (lon_max - lon_min)
            bar_lon_end = bar_lon_start + frac * (lon_max - lon_min)

            # Convert to radians
            R = 6371000  # meters
            lat_rad = np.deg2rad(bar_lat)
            dlon_rad = np.deg2rad(bar_lon_end - bar_lon_start)
            bar_length_m = abs(dlon_rad * np.cos(lat_rad) * R)

            # # Label: show the actual bar length in appropriate units
            # if bar_length_m >= 1609.34:
            #     label = f"{bar_length_m/1609.34:.1f} mi"
            if bar_length_m >= 1000:
                label = f"{bar_length_m/1000:.1f} km"
            else:
                label = f"{int(bar_length_m)} m"

            # Draw the scale bar
            self.ax.plot([bar_lon_start, bar_lon_end], [bar_lat, bar_lat], color='k', linewidth=3, transform=ccrs.PlateCarree())
            self.ax.text((bar_lon_start + bar_lon_end) / 2, bar_lat + 0.01 * (lat_max - lat_min),
                        label, ha='center', va='bottom', fontsize=10, transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"Failed to draw scale bar: {e}")

    def save_grid(self, _btn=None):
        name = self._snapshot_name.value.strip()
        msg = self._commit_msg.value.strip()
        if not name:
            print("Enter a grid name!")
            return
        if not msg:
            print("Enter a grid message!")
            return

        if name.lower().endswith('.json'):
            name = name[:-5]
        sanitized_name = self.grid.sanitize_name(name)
        self.grid.name = sanitized_name

        # Use Grid's path logic
        folder, nc_path, json_path = self.grid._get_folder_and_paths(self.repo_root)
        self.SNAPSHOT_DIR = folder

        self.grid.to_netcdf(nc_path)
        self.grid.save_metadata(json_path, msg, ncfile=nc_path)

        print(f"Saved grid '{os.path.basename(json_path)}' and NetCDF in '{folder}'.")
        self.refresh_commit_dropdown()
        return

    def load_grid(self, b=None):
        val = self._commit_dropdown.value
        if not val:
            print("No grid selected.")
            return
        # Use Grid's path logic to get the full path to the JSON
        snapshot_path = os.path.join(self.grids_dir, val)
        try:
            data = self.grid.load_metadata(snapshot_path)
            domain_id = data.get("domain_id", {})
            ncfile = data.get("ncfile")
            if not ncfile:
                print("No NetCDF file recorded in metadata. Cannot load grid.")
                return
            nc_path = os.path.join(os.path.dirname(snapshot_path), ncfile)
            self.grid = self.grid.from_netcdf(nc_path)
            # Set sliders using metadata
            self._resolution_slider.value = float(domain_id.get("resolution"))
            self._xstart_slider.value = float(domain_id.get("xstart"))
            self._lenx_slider.value = float(domain_id.get("lenx"))
            self._ystart_slider.value = float(domain_id.get("ystart"))
            self._leny_slider.value = float(domain_id.get("leny"))
            self.plot_grid()
            print(f"Loaded grid '{snapshot_path}' and NetCDF '{nc_path}'.")
        except Exception as e:
            print(f"Failed to load grid: {e}")

    def sync_sliders_to_grid(self):
        try:
            sliders = [
                self._resolution_slider,
                self._xstart_slider,
                self._lenx_slider,
                self._ystart_slider,
                self._leny_slider,
            ]
            for slider in sliders:
                try:
                    slider.unobserve(self._on_slider_change, names="value")
                except ValueError:
                    pass

            self._resolution_slider.value = float(self.grid.resolution)
            self._xstart_slider.value = float(self.grid.xstart)
            self._lenx_slider.value = float(self.grid.lenx)
            self._ystart_slider.value = float(self.grid.ystart)
            self._leny_slider.value = float(self.grid.leny)

            for slider in sliders:
                slider.observe(self._on_slider_change, names="value")
        except Exception as e:
            print(f"Error in sync_sliders_to_grid: {e}")
            
    def _on_slider_change(self, change):
        from mom6_bathy.grid import Grid
        self.grid = Grid(
            lenx=self._lenx_slider.value,
            leny=self._leny_slider.value,
            resolution=self._resolution_slider.value,
            xstart=self._xstart_slider.value,
            ystart=self._ystart_slider.value,
            name=self.grid.name,
            save_on_create=False  # Avoid saving on every slider change
        )
        self.plot_grid()

    def reset_grid(self, b=None):
        from mom6_bathy.grid import Grid
        params = self._initial_params
        name = self._snapshot_name.value.strip() or params["name"]
        sanitized_name = self.grid.sanitize_name(name)
        self.grid = Grid(
            lenx=params["lenx"],
            leny=params["leny"],
            resolution=params["resolution"],
            xstart=params["xstart"],
            ystart=params["ystart"],
            name=sanitized_name
        )
        self.sync_sliders_to_grid()
        self.plot_grid()

    def refresh_commit_dropdown(self):
        grid_jsons = self.grid.list_metadata_files(self.grids_dir)
        options = []
        for rel_path in grid_jsons:
            abs_path = os.path.join(self.grids_dir, rel_path)
            try:
                data = self.grid.load_metadata(abs_path)
                msg = data.get("message", "")
                date = data.get("date", "")
                domain_id = data.get("domain_id", {})
                # Extract shape from folder name if possible
                folder_name = os.path.basename(os.path.dirname(abs_path))
                # Remove 'grid_' prefix from file name
                base_name = os.path.basename(rel_path)
                if base_name.startswith("grid_"):
                    base_name = base_name[5:]
                # Compose label: name_shape (from folder), or fallback to name
                label = f"{folder_name}" if "_" in folder_name else f"{domain_id.get('name', '')}"
                options.append((label, rel_path))
            except Exception:
                continue

        # Sort by file modification time, newest first
        options.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.grids_dir, x[1])),
            reverse=True
        )

        self._commit_dropdown.options = options if options else []
        if options:
            option_values = [v for (l, v) in options]
            if self._commit_dropdown.value not in option_values:
                self._commit_dropdown.value = options[0][1]
        else:
            self._commit_dropdown.value = None
        self.update_commit_details()

    def update_commit_details(self, change=None):
        val = self._commit_dropdown.value
        if not val:
            self._commit_details.value = ""
            return
        file_path = val
        abs_path = os.path.join(self.grids_dir, file_path)
        try:
            data = self.grid.load_metadata(abs_path)
            msg = data.get("message", "")
            date = data.get("date", "")
            domain_id = data.get("domain_id", {})
            details = (
                f"<b>Name:</b> {domain_id.get('name', '')}<br>"
                f"<b>Date:</b> {date}<br>"
                f"<b>Message:</b> {msg}<br>"
            )
            self._commit_details.value = details
        except Exception:
            self._commit_details.value = ""