from abc import ABC, abstractmethod

class EditCommand(ABC):

    @abstractmethod
    def execute(self, topo):
        pass

    @abstractmethod
    def undo(self, topo):
        pass

    @abstractmethod
    def serialize(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: dict):
        pass

class CellEditCommand(EditCommand):
    """Define any edit that affects one or more elements of an array"""
    def __init__(self, affected_indices, new_values, old_values=None):
        self.affected_indices = affected_indices 
        self.new_values = new_values            
        self.old_values = old_values             

    def execute(self, topo):
        if self.old_values is None:
            self.old_values = []
            self.old_values.extend(
                self._get_value(topo, j, i) for j, i in self.affected_indices
            )
        for idx, (j, i) in enumerate(self.affected_indices):
            self._set_value(topo, j, i, self.new_values[idx])

    def undo(self, topo):
        for idx, (j, i) in enumerate(self.affected_indices):
            self._set_value(topo, j, i, self.old_values[idx])

    def serialize(self):
        return {
            'type': self.__class__.__name__,
            'affected_indices': self.affected_indices,
            'new_values': self.new_values,
            'old_values': self.old_values
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            affected_indices=data['affected_indices'],
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
        self.new_value = new_value
        self.old_value = old_value

    def execute(self, topo):
        if self.old_value is None:
            self.old_value = getattr(topo, self.attr)
        setattr(topo, self.attr, self.new_value)

    def undo(self, topo):
        setattr(topo, self.attr, self.old_value)

    def serialize(self):
        return {
            'type': self.__class__.__name__,
            'attr': self.attr,
            'new_value': self.new_value,
            'old_value': self.old_value,
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            attr=data['attr'],
            new_value=data['new_value'],
            old_value=data['old_value']
        )

# Registry for type <-> class mapping
COMMAND_REGISTRY = {}
def register_command(cls):
    COMMAND_REGISTRY[cls.__name__] = cls
    return cls