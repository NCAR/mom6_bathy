from CrocoDash.edit_command import EditCommand, register_command, COMMAND_REGISTRY
import numpy as np

def to_native(val):
    # Converts numpy scalars to Python native types, otherwise returns as-is.
    return val.item() if isinstance(val, (np.generic,)) else val

def to_native_tuple(t):
    # Converts a tuple of numpy ints to native ints
    return tuple(int(x) for x in t)

class CellEditCommand(EditCommand):
    """Define any edit that affects one or more elements of an array"""
    def __init__(self, affected_indices, new_values, old_values=None):
        # Convert indices and values to native types for consistency and serialization
        self.affected_indices = [to_native_tuple(idx) for idx in affected_indices]
        self.new_values = [to_native(v) for v in new_values]
        self.old_values = [to_native(v) for v in old_values] if old_values is not None else None

    def execute(self, topo):
        if self.old_values is None:
            self.old_values = [to_native(self._get_value(topo, j, i)) for j, i in self.affected_indices]
        for idx, (j, i) in enumerate(self.affected_indices):
            self._set_value(topo, j, i, self.new_values[idx])

    def undo(self, topo):
        for idx, (j, i) in enumerate(self.affected_indices):
            self._set_value(topo, j, i, self.old_values[idx])

    def serialize(self):
        return {
            'type': self.__class__.__name__,
            'affected_indices': [to_native_tuple(idx) for idx in self.affected_indices],
            'new_values': [to_native(v) for v in self.new_values],
            'old_values': [to_native(v) for v in self.old_values] if self.old_values is not None else None
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            affected_indices=[tuple(idx) for idx in data['affected_indices']],
            new_values=data['new_values'],
            old_values=data['old_values']
        )

    # These methods must be implemented by subclasses
    def _get_value(self, topo, j, i):
        raise NotImplementedError

    def _set_value(self, topo, j, i, value):
        raise NotImplementedError

class ScalarEditCommand(EditCommand):
    """Define any edit that affects a single scalar attribute of an object"""
    def __init__(self, attr, new_value, old_value=None):
        self.attr = attr
        self.new_value = to_native(new_value)
        self.old_value = to_native(old_value) if old_value is not None else None

    def execute(self, topo):
        if self.old_value is None:
            self.old_value = to_native(getattr(topo, self.attr))
        setattr(topo, self.attr, self.new_value)

    def undo(self, topo):
        setattr(topo, self.attr, self.old_value)

    def serialize(self):
        return {
            'type': self.__class__.__name__,
            'attr': self.attr,
            'new_value': to_native(self.new_value),
            'old_value': to_native(self.old_value),
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            attr=data['attr'],
            new_value=data['new_value'],
            old_value=data['old_value']
        )

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