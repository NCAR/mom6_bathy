import os
import json
import datetime
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import threading
from mom6_bathy.vgrid import VGrid

class VGridEditor(widgets.HBox):
    """
    Interactive editor for vertical grids (VGrid).
    Allows creation, editing, saving, and loading of vertical grid profiles.
    """

    def __init__(self, vgrid=None, repo_root=None):
        self.repo_root = repo_root if repo_root is not None else os.getcwd()
        self.vgrids_dir = os.path.join(self.repo_root, "VGrids")
        os.makedirs(self.vgrids_dir, exist_ok=True)

        if vgrid is None:
            vgrid = VGrid.uniform(nk=10, depth=100.0)
        self.vgrid = vgrid
        self._initial_dz = np.copy(self.vgrid.dz)

        # Infer ratio and grid type from dz
        ratio_value, grid_type = self.infer_ratio_and_type(self.vgrid.dz)

        # --- Disable observers during setup ---
        self._observers_attached = False
        self.construct_control_panel(ratio_value=ratio_value, grid_type=grid_type)

        # Set widget values to match vgrid BEFORE attaching observers
        self._nk_slider.value = self.vgrid.nk
        self._depth_slider.value = float(self.vgrid.depth)
        self._ratio_slider.value = ratio_value
        self._type_toggle.value = grid_type

        self.plot_vgrid()
        super().__init__([self._control_panel, self.fig.canvas], layout=widgets.Layout(width="100%", align_items="flex-start"))
        self.construct_observances()  # Attach observers only after UI is set up and values are correct
        self._observers_attached = True
        self.refresh_commit_dropdown()

    @staticmethod
    def infer_ratio_and_type(dz, tol=1e-2):
        """Infer the top/bottom ratio and grid type from dz array."""
        dz0 = dz[0]
        dzbot = dz[-1]
        ratio = dzbot / dz0 if dz0 != 0 else 1.0
        if np.isclose(ratio, 1.0, atol=tol):
            return 1.0, "Uniform"
        else:
            return ratio, "Hyperbolic"

    def construct_control_panel(self, ratio_value=1.0, grid_type="Uniform"):
        label_style = {'description_width': '120px'}

        self._nk_slider = widgets.IntSlider(
            value=self.vgrid.nk, min=2, max=100, step=1, description="Levels",
            layout={'width': '98%'}, style=label_style
        )
        self._depth_slider = widgets.FloatSlider(
            value=self.vgrid.depth, min=1.0, max=10000.0, step=1.0, description="Depth (m)",
            layout={'width': '98%'}, style=label_style
        )
        self._ratio_slider = widgets.FloatSlider(
            value=ratio_value, min=0.1, max=20.0, step=0.01,
            description="Top/Bottom Ratio:",
            layout={'width': '98%'}, style=label_style
        )
        self.ratio_help = widgets.HTML(
            value="<span style='font-size: 90%; color: #888;'>Ratio of bottom layer thickness to top layer thickness (for hyperbolic grids)</span>",
            layout={'width': '98%', 'display': 'none'}
        )

        # Timer handle for hiding help
        self._ratio_help_timer = None

        def show_help(change=None):
            self.ratio_help.layout.display = 'block'
            # Cancel any previous timer
            if self._ratio_help_timer is not None:
                self._ratio_help_timer.cancel()
            # Start a new timer to hide after 3 seconds
            self._ratio_help_timer = threading.Timer(3.0, hide_help)
            self._ratio_help_timer.start()

        def hide_help():
            self.ratio_help.layout.display = 'none'

        self._ratio_slider.observe(show_help, names='value')

        self._type_toggle = widgets.ToggleButtons(
            options=["Uniform", "Hyperbolic"],
            value=grid_type,
            description="Type",
            layout={'width': '98%'}, style=label_style
        )
        self._snapshot_name = widgets.Text(
            value='', placeholder='Enter vgrid name', description='Name:',
            layout={'width': '98%'}, style=label_style
        )
        self._commit_msg = widgets.Text(
            value='', placeholder='Enter vgrid message', description='Message:',
            layout={'width': '98%'}, style=label_style
        )
        self._commit_dropdown = widgets.Dropdown(
            options=[], description='VGrids:', layout={'width': '98%'}, style=label_style
        )
        self._commit_details = widgets.HTML(
            value="", layout={'width': '98%', 'min_height': '2em'}
        )
        self._save_button = widgets.Button(
            description='Save VGrid', layout={'width': '49%'}
        )
        self._load_button = widgets.Button(
            description='Load VGrid', layout={'width': '49%'}
        )
        self._reset_button = widgets.Button(
            description='Reset', layout={'width': '98%'}, button_style='danger'
        )

        controls = widgets.VBox([
            widgets.HTML("<h3>Vertical Grid Controls</h3>"),
            self._type_toggle,
            self._nk_slider,
            self._depth_slider,
            self._ratio_slider,
            self.ratio_help,  # Only shows on focus
            self._reset_button,
        ], layout=widgets.Layout(width="100%", align_items="stretch", overflow_y="visible"))
        
        library_section = widgets.VBox([
            widgets.HTML("<h3>Library</h3>"),
            self._snapshot_name,
            self._commit_msg,
            self._commit_dropdown,
            self._commit_details,
            widgets.HBox([self._save_button, self._load_button], layout={'width': '100%'}),
        ], layout={'width': '100%'})

        self._control_panel = widgets.VBox([
            controls,
            library_section,
        ], layout={'width': '35%', 'height': '100%'})

    def construct_observances(self):
        # Only attach observers if not already attached
        if getattr(self, "_observers_attached", False):
            return
        self._save_button.on_click(self.save_vgrid)
        self._load_button.on_click(self.load_vgrid)
        self._reset_button.on_click(self.reset_vgrid)
        self._snapshot_name.observe(lambda change: self.refresh_commit_dropdown(), names='value')
        self._commit_dropdown.observe(self.update_commit_details, names='value')
        self._type_toggle.observe(self._on_param_change, names="value")
        self._nk_slider.observe(self._on_param_change, names="value")
        self._depth_slider.observe(self._on_param_change, names="value")
        self._ratio_slider.observe(self._on_param_change, names="value")

    def plot_vgrid(self):
        if not hasattr(self, "fig"):
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(6, 5))
            plt.ion()
        else:
            self.ax.clear()
        # Draw horizontal lines at each interface
        for depth in self.vgrid.z:
            self.ax.axhline(y=depth, color='steelblue')
        self.ax.set_ylim(max(self.vgrid.z) + 10, min(self.vgrid.z) - 10)  # Invert y-axis
        self.ax.set_ylabel("Depth")
        self.ax.set_title("Vertical Grid")
        self.fig.canvas.draw_idle()

    def _on_param_change(self, change):
        nk = self._nk_slider.value
        depth = self._depth_slider.value
        ratio = self._ratio_slider.value
        grid_type = self._type_toggle.value
        if grid_type == "Uniform":
            self.vgrid = VGrid.uniform(nk=nk, depth=depth)
            self._ratio_slider.disabled = True
        else:
            self.vgrid = VGrid.hyperbolic(nk=nk, depth=depth, ratio=ratio)
            self._ratio_slider.disabled = False
        self.plot_vgrid()

    def save_vgrid(self, _btn=None):
        name = self._snapshot_name.value.strip()
        msg = self._commit_msg.value.strip()
        if not name:
            print("Enter a vgrid name!")
            return
        if not msg:
            print("Enter a vgrid message!")
            return

        if name.lower().endswith('.nc'):
            name = name[:-3]
        sanitized_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        vgrid_name = f"vgrid_{sanitized_name}"
        vgrid_dir = os.path.join(self.vgrids_dir, vgrid_name)
        os.makedirs(vgrid_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "name": sanitized_name,
            "nk": self.vgrid.nk,
            "depth": float(self.vgrid.depth),
            "type": self._type_toggle.value,
            "ratio": self._ratio_slider.value,
            "message": msg,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = os.path.join(vgrid_dir, f"{vgrid_name}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved vgrid metadata in '{vgrid_dir}'.")
        self.refresh_commit_dropdown()
        rel_path = os.path.join(vgrid_name, f"{vgrid_name}.json")
        if rel_path in [v for (_, v) in self._commit_dropdown.options]:
            self._commit_dropdown.value = rel_path
        return

    def load_vgrid(self, b=None):
        val = self._commit_dropdown.value
        if not val:
            print("No vgrid selected.")
            return
        meta_path = os.path.join(self.vgrids_dir, val)
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
            nk = data.get("nk", 10)
            depth = data.get("depth", 100.0)
            ratio = data.get("ratio", 1.0)
            grid_type = data.get("type", "Uniform")

            # Temporarily remove observers to avoid triggering _on_param_change
            self._nk_slider.unobserve(self._on_param_change, names="value")
            self._depth_slider.unobserve(self._on_param_change, names="value")
            self._type_toggle.unobserve(self._on_param_change, names="value")
            self._ratio_slider.unobserve(self._on_param_change, names="value")

            if grid_type == "Uniform":
                self.vgrid = VGrid.uniform(nk=nk, depth=depth)
            else:
                self.vgrid = VGrid.hyperbolic(nk=nk, depth=depth, ratio=ratio)

            self._nk_slider.value = self.vgrid.nk
            self._depth_slider.value = float(self.vgrid.depth)
            self._type_toggle.value = grid_type
            self._ratio_slider.value = ratio

            # Reconnect observers
            self._nk_slider.observe(self._on_param_change, names="value")
            self._depth_slider.observe(self._on_param_change, names="value")
            self._type_toggle.observe(self._on_param_change, names="value")
            self._ratio_slider.observe(self._on_param_change, names="value")

            self.plot_vgrid()
            print(f"Loaded vgrid '{val}'.")
        except Exception as e:
            print(f"Failed to load vgrid: {e}")

    def reset_vgrid(self, b=None):
        self.vgrid = VGrid(self._initial_dz.copy())
        ratio_value, grid_type = self.infer_ratio_and_type(self.vgrid.dz)
        self._nk_slider.value = self.vgrid.nk
        self._depth_slider.value = float(self.vgrid.depth)
        self._type_toggle.value = grid_type
        self._ratio_slider.value = ratio_value
        self.plot_vgrid()

    def refresh_commit_dropdown(self):
        vgrid_jsons = []
        for root, dirs, files in os.walk(self.vgrids_dir):
            for fname in files:
                if fname.startswith("vgrid_") and fname.endswith(".json"):
                    rel_dir = os.path.relpath(root, self.vgrids_dir)
                    rel_path = os.path.join(rel_dir, fname) if rel_dir != "." else fname
                    vgrid_jsons.append(rel_path)
        options = []
        for rel_path in vgrid_jsons:
            abs_path = os.path.join(self.vgrids_dir, rel_path)
            try:
                with open(abs_path, "r") as f:
                    data = json.load(f)
                base = os.path.basename(rel_path)
                if base.startswith("vgrid_") and base.endswith(".json"):
                    display_name = base[len("vgrid_"):-len(".json")]
                else:
                    display_name = base
                options.append((display_name, rel_path))
            except Exception:
                continue

        options.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.vgrids_dir, x[1])),
            reverse=True
        )
        self._commit_dropdown.options = options if options else []
        # Only set value if it is already set and still valid
        if options:
            option_values = [v for (l, v) in options]
            if self._commit_dropdown.value not in option_values:
                self._commit_dropdown.value = None
        else:
            self._commit_dropdown.value = None
        self.update_commit_details()

    def update_commit_details(self, change=None):
        val = self._commit_dropdown.value
        if not val:
            self._commit_details.value = ""
            return
        abs_path = os.path.join(self.vgrids_dir, val)
        try:
            with open(abs_path, "r") as f:
                data = json.load(f)
            details = (
                f"<b>Name:</b> {data.get('name', '')}<br>"
                f"<b>Date:</b> {data.get('date', '')}<br>"
                f"<b>Message:</b> {data.get('message', '')}<br>"
            )
            self._commit_details.value = details
        except Exception:
            self._commit_details.value = ""