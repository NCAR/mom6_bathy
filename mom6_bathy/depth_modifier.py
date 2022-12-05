import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import ipywidgets as widgets

class DepthModifier(widgets.AppLayout):
        
    def __init__(self,bathy):
        
        self.bathy = bathy

        # depth and mask values (subject to change)
        self.depth_data = bathy.depth.data.copy()
        self.tmask_data = bathy.tmask.data.copy()
        
        self.depth_min, self.depth_max = self.depth_data.min(), self.depth_data.max()
        
        self.ny = self.tmask_data.shape[0]
        self.nx = self.tmask_data.shape[1]
        

        # ensure we are interactive mode 
        # this is default but if this notebook is executed out of order it may have been turned off
        plt.ioff()
        fig = plt.figure()
        plt.ion()

        self.ax = fig.gca()

        # Minor ticks
        self.ax.set_xticks(np.arange(-.5, self.nx, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.ny, 1), minor=True)

        # Gridlines based on minor ticks
        self.ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.3)

        # Remove minor ticks
        self.ax.tick_params(which='minor', bottom=False, left=False)

        # plot
        self.im = self.ax.imshow(self.depth_data)

        # colorbar
        self.cbar = plt.colorbar(self.im)
        #cbar_ticks = np.linspace(0., 1., num=6, endpoint=True)
        #self.cbar.set_ticks(cbar_ticks)
        #fig.colorbar(self.im)

        self.construct_header()
        self.construct_footer()

        self.selection = {}

        def on_cell_click(event):
            if not event.dblclick:
                pass # single click
            else: # double click
                if 'patch' in self.selection:
                    patch = self.selection.pop('patch')
                    patch.remove()
                self.selection['i'] = int(event.xdata + 0.5)
                self.selection['j'] = int(event.ydata  + 0.5)
                self.ax.set_title(f"Selection: i={self.selection['i']}, j={self.selection['j']}")

                self.modifier_widget.layout.display=''
                self.lbl_selected_cell.value = f"Selected cell at i={self.selection['i']}, j={self.selection['j']}."
                self.lbl_curr_depth.value = f"Current depth: {self.depth_data[self.selection['j'],self.selection['i']]}"

                self.selection['patch'] = patches.Circle((self.selection['i'], self.selection['j']), 0.3, fc='r', alpha=0.8)
                self.ax.add_patch(self.selection['patch'])

        cid = fig.canvas.mpl_connect('button_press_event', on_cell_click)

        self.ax.set_title('Double click on a cell to change its depth/mask.')

        super().__init__(
            header = self.tbtn_display,
            center=fig.canvas,
            footer=self.modifier_widget,
            pane_heights=[0, 5, 1],
            pane_widths=[0, 9, 0]
        )

    def construct_header(self):

        self.tbtn_display = widgets.ToggleButtons(
            value = 'Depth',
            options=['Depth','Mask'],
            description='Toggle display:',
            disabled=False,
        )
        self.tbtn_display.style.description_width = '200px'
        self.tbtn_display.style.button_width = '90px'

        def on_tbtn_display_click(change):
            self.refresh_display(mode=change['new'])

        self.tbtn_display.observe(
            on_tbtn_display_click,
            names='value',
            type='change'
        )

    def construct_footer(self):

        # Modify depth widget -------------------------------------------------------

        self.lbl_selected_cell = widgets.Label("")
        self.lbl_curr_depth = widgets.Label("")


        btn_apply = widgets.Button(
            description = "Apply"
        )

        btn_save = widgets.Button(
            description = "Save",
            icon='floppy-disk',
            button_style='success',
            disabled = True
        )

        lbl_save_status = widgets.Label("")

        txt_set_depth = widgets.FloatText(
            description="Enter new depth:"
        )
        txt_set_depth.style.description_width = '160px'

        self.modifier_widget = widgets.VBox([
            self.lbl_selected_cell,
            self.lbl_curr_depth,
            widgets.HBox([txt_set_depth, btn_apply, btn_save, lbl_save_status])
        ],
            layout=widgets.Layout(left='50px', display='none')
        )

        def apply_new_depth(b):

            j = self.selection['j']
            i = self.selection['i']
            new_depth = txt_set_depth.value
            curr_depth = self.depth_data[j,i]

            mask_changes = (self.depth_data[j,i] >= self.bathy.min_depth) != (new_depth >= self.bathy.min_depth)
            if mask_changes:
                self.tmask_data[j,i] = 1 if new_depth >= self.bathy.min_depth else 0
            self.depth_data[j,i] = new_depth

            if 'patch' in self.selection:
                patch = self.selection.pop('patch')
                patch.remove()

            if curr_depth == self.depth_min:
                # find the new min
                self.depth_min = self.depth_data.min()
            if curr_depth == self.depth_max:
                # find the new max
                self.depth_max = self.depth_data.max()

            self.depth_min = min(self.depth_min, new_depth)
            self.depth_max = max(self.depth_max, new_depth)

            self.refresh_display()

            if btn_save.disabled == True:
                btn_save.disabled = False
                lbl_save_status.value = ''

        btn_apply.on_click(apply_new_depth)

        def save(b):
            self.bathy.depth.data = self.depth_data
            self.bathy.tmask.data = self.tmask_data
            lbl_save_status.value = "Saved!"
            btn_save.disabled = True

        btn_save.on_click(save)

    def refresh_display(self, mode=None):

        if mode is None:
            mode = self.tbtn_display.value

        if mode == "Depth":
            self.ax.imshow(self.depth_data)
            self.im.set_clim((self.depth_min,self.depth_max))
            self.cbar.update_normal(self.im)
        else:
            self.ax.imshow(self.tmask_data)
            self.im.set_clim((0.0,1.0))
            self.cbar.update_normal(self.im)
