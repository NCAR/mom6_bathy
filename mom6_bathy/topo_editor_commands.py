from mom6_bathy.edit_command import CellEditCommand, ScalarEditCommand, register_command, COMMAND_REGISTRY

@register_command
class SetDepthCommand(CellEditCommand):

    def _get_value(self, topo, j, i):
        return topo.depth.data[j, i]
    
    def _set_value(self, topo, j, i, value):
        topo.depth.data[j, i] = value

@register_command
class SetMinDepthCommand(ScalarEditCommand):
    pass

@register_command
class EraseSelectedBasinCommand(SetDepthCommand):
    # We are just setting the depth of many indices to zero.
    pass

@register_command
class EraseDisconnectedBasinsCommand(SetDepthCommand):
    # We are just setting the depth of many indices to zero.
    pass