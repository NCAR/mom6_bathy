import json
import os
import numpy as np
from abc import ABC, abstractmethod

class CommandManager(ABC):
    def __init__(self, domain_id, snapshot_dir="Topos"):
        self._undo_history = []
        self._redo_history = []
        self.snapshot_dir = snapshot_dir
        self.domain_id = domain_id

    def get_domain_id(self):
        """
        Return a unique identifier for the domain or context.

        This identifier is used to associate command histories and snapshots with a specific
        domain (such as a dataset, grid, or other context), and to prevent applying histories
        to incompatible domains.

        Returns
        -------
        domain_id : object
            A value or data structure (commonly a dict or string) that uniquely identifies
            the domain. If self.domain_id is callable, this method should call it and return
            the result; otherwise, it should return self.domain_id as-is.
        """
        return self.domain_id() if callable(self.domain_id) else self.domain_id
    
    def get_history_path(self):
        dom_id = self.get_domain_id()
        if isinstance(dom_id, dict):
            id_str = "_".join(f"{k}-{v}" for k, v in dom_id.items())
        else:
            id_str = str(dom_id)
        return os.path.join(self.snapshot_dir, f"history_{id_str}.json")

    def save_histories(self):
        path = self.get_history_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([cmd.serialize() for cmd in self._undo_history], f)

    def load_histories(self, command_registry, *args, **kwargs):
        path = self.get_history_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                self._undo_history = [
                    command_registry[d['type']].deserialize(d)(*args, **kwargs) for d in data
                ]
                self._redo_history = []

    def save_commit(self, name):
        os.makedirs(self.snapshot_dir, exist_ok=True)
        fname = os.path.join(self.snapshot_dir, f"{name}.json")
        data = {
            "domain_id": self.get_domain_id(),
            "undo_history": [cmd.serialize() for cmd in self._undo_history],
            "redo_history": [cmd.serialize() for cmd in self._redo_history]
        }
        with open(fname, "w") as f:
            json.dump(data, f)

    def load_commit(self, name, command_registry, *args, **kwargs):
        fname = os.path.join(self.snapshot_dir, f"{name}.json")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No snapshot named {name}")
        with open(fname, "r") as f:
            data = json.load(f)
        snapshot_domain = data.get("domain_id", {})
        current_domain = self.get_domain_id()
        if snapshot_domain != current_domain:
            # Accept domain change silently for TopoEditor
            pass
        self._undo_history = [
            command_registry[d['type']].deserialize(d)(*args, **kwargs) for d in data["undo_history"]
        ]
        self._redo_history = [
            command_registry[d['type']].deserialize(d)(*args, **kwargs) for d in data["redo_history"]
        ]
        self.replay()

    @abstractmethod
    def execute(self, cmd):
        """Execute a command, push it onto the undo stack, and clear the redo stack."""
        pass

    @abstractmethod
    def push(self, command):
        """Add a command to the history."""
        pass

    @abstractmethod
    def undo(self):
        """Undo the last command."""
        pass

    @abstractmethod
    def redo(self):
        """Redo the last undone command."""
        pass

    @abstractmethod
    def initialize(self, command_registry, *args, **kwargs):
        """Initialize with a given registry and context."""
        pass

    @abstractmethod
    def replay(self):
        """Replay all commands in the undo history."""
        pass

class TopoCommandManager(CommandManager):
    def __init__(self, domain_id, topo, command_registry, snapshot_dir="Topos"):
        super().__init__(domain_id, snapshot_dir)
        self._topo = topo
        self._command_registry = command_registry

    def execute(self, cmd):
        """
        Execute a command object. If it's a user edit, push to history and clear redo.
        For system commands (undo, redo, save, load, reset), the command object handles everything.
        """
        user_edit_types = ('DepthEditCommand', 'MinDepthEditCommand')  # Add more as needed
        if cmd.__class__.__name__ in user_edit_types:
            cmd()
            self.push(cmd)
            self._redo_history.clear()
            self.save_histories()
        else:
            cmd()

    def get_history_path(self):
        dom_id = self.get_domain_id()
        if isinstance(dom_id, dict):
            grid_name = dom_id.get("grid_name", "unknown")
            shape = dom_id.get("shape", ["?", "?"])
            shape_str = f"{shape[0]}x{shape[1]}"
            fname = f"history_{grid_name}_{shape_str}.json"
        else:
            fname = f"history_{str(dom_id)}.json"
        return os.path.join(self.snapshot_dir, fname)
    
    def push(self, command):
        self._undo_history.append(command)
        self._redo_history.clear()

    def load_histories(self):
        super().load_histories(self._command_registry, self._topo)

    def replay(self):
        for cmd in self._undo_history:
            cmd()

    def undo(self):
        if not self._undo_history:
            return False
        cmd = self._undo_history.pop()
        cmd.undo()
        self._redo_history.append(cmd)
        self.save_histories()
        return True

    def redo(self):
        if not self._redo_history:
            return False
        cmd = self._redo_history.pop()
        cmd()
        self._undo_history.append(cmd)
        self.save_histories()
        return True

    def load_commit(self, name, command_registry, topo, reset_to_original=False):
        """ Original (or original state) refers to the reference, or baseline state of the (topo) data before 
        any user edits or modifications have been applied.
        """
        if reset_to_original: 
            # If true, topo is reset to the original state. If false, in-memory state is used.
            topo_id = self.get_domain_id()
            grid_name = topo_id['grid_name']
            shape = topo_id['shape']
            shape_str = f"{shape[0]}x{shape[1]}"
            original_topo_path = os.path.join(self.snapshot_dir, f"original_topo_{grid_name}_{shape_str}.npy")
            original_min_depth_path = os.path.join(self.snapshot_dir, f"original_min_depth_{grid_name}_{shape_str}.json")

            if os.path.exists(original_topo_path):
                topo.depth.data[:] = np.load(original_topo_path)
                if os.path.exists(original_min_depth_path):
                    with open(original_min_depth_path, "r") as f:
                        d = json.load(f)
                        topo.min_depth = d.get("min_depth", topo.min_depth)
            else:
                print("Warning: original topo not found, cannot reset before loading snapshot.")

        # Call the base class implementation for the rest
        super().load_commit(name, command_registry, topo)

    def reset(self, topo, original_depth, original_min_depth, get_topo_id, min_depth_specifier=None, trigger_refresh=None):
        topo_id = get_topo_id()
        grid_name = topo_id['grid_name']
        shape = topo_id['shape']
        shape_str = f"{shape[0]}x{shape[1]}"
        original_topo_path = os.path.join(self.snapshot_dir, f"original_topo_{grid_name}_{shape_str}.npy")
        original_min_depth_path = os.path.join(self.snapshot_dir, f"original_min_depth_{grid_name}_{shape_str}.json")

        if os.path.exists(original_topo_path):
            topo.depth.data[:] = np.load(original_topo_path)
            if os.path.exists(original_min_depth_path):
                with open(original_min_depth_path, "r") as f:
                    d = json.load(f)
                    topo.min_depth = d.get("min_depth", topo.min_depth)
        else:
            topo.depth.data[:] = np.copy(original_depth)
            topo.min_depth = original_min_depth

        original_name = f"original_{grid_name}_{shape_str}"
        from mom6_bathy.edit_command import COMMAND_REGISTRY
        try:
            self.load_commit(original_name, COMMAND_REGISTRY, topo)
        except FileNotFoundError:
            print("Original command snapshot not found, edit history not reset.")

        if min_depth_specifier is not None:
            min_depth_specifier.value = topo.min_depth
        if trigger_refresh is not None:
            trigger_refresh()

        self._undo_history.clear()
        self._redo_history.clear()
        self.save_histories()

    def initialize(self):
        self.load_histories()
        self.replay()
        