import json
import os
import numpy as np
from CrocoDash.command_manager import CommandManager

class TopoCommandManager(CommandManager):
    def __init__(self, domain_id, snapshot_dir="snapshots"):
        super().__init__(domain_id, snapshot_dir)

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

    def get_domain_id(self):
        return self.domain_id() if callable(self.domain_id) else self.domain_id

    def get_history_path(self):
        dom_id = self.get_domain_id()
        if isinstance(dom_id, dict):
            id_str = "_".join(f"{k}-{v}" for k, v in dom_id.items())
        else:
            id_str = str(dom_id)
        return os.path.join("edit_histories", f"history_{id_str}.json")

    def push(self, command):
        self._undo_history.append(command)
        self._redo_history.clear()

    def save_histories(self):
        path = self.get_history_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([cmd.serialize() for cmd in self._undo_history], f)

    def load_histories(self, command_registry, topo):
        path = self.get_history_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                self._undo_history = [
                    command_registry[d['type']].deserialize(d)(topo) for d in data
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

    def load_commit(self, name, command_registry, topo):
        fname = os.path.join(self.snapshot_dir, f"{name}.json")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No snapshot named {name}")
        with open(fname, "r") as f:
            data = json.load(f)
        self._undo_history = [
            command_registry[d['type']].deserialize(d)(topo) for d in data["undo_history"]
        ]
        self._redo_history = [
            command_registry[d['type']].deserialize(d)(topo) for d in data["redo_history"]
        ]

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

    def load_commit(self, name, command_registry, topo):
        fname = os.path.join(self.snapshot_dir, f"{name}.json")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No snapshot named {name}")
        with open(fname, "r") as f:
            data = json.load(f)
        self._undo_history = [
            command_registry[d['type']].deserialize(d)(topo) for d in data["undo_history"]
        ]
        self._redo_history = [
            command_registry[d['type']].deserialize(d)(topo) for d in data["redo_history"]
        ]

    def reset(self, topo, original_depth, original_min_depth, get_topo_id, min_depth_specifier=None, trigger_refresh=None):
        topo_id = get_topo_id()
        grid_name = topo_id['grid_name']
        shape = topo_id['shape']
        shape_str = f"{shape[0]}x{shape[1]}"
        golden_dir = "original_topo"
        golden_topo_path = os.path.join(golden_dir, f"golden_topo_{grid_name}_{shape_str}.npy")
        golden_min_depth_path = os.path.join(golden_dir, f"golden_min_depth_{grid_name}_{shape_str}.json")

        if os.path.exists(golden_topo_path):
            topo.depth.data[:] = np.load(golden_topo_path)
            if os.path.exists(golden_min_depth_path):
                with open(golden_min_depth_path, "r") as f:
                    d = json.load(f)
                    topo.min_depth = d.get("min_depth", topo.min_depth)
            print(f"Topo reset to golden/original topo for {grid_name} {shape_str} from disk.")
        else:
            topo.depth.data[:] = np.copy(original_depth)
            topo.min_depth = original_min_depth
            print("Topo reset to first-in-memory state.")

        golden_name = f"golden_{grid_name}_{shape_str}"
        from CrocoDash.edit_command import COMMAND_REGISTRY
        try:
            self.load_commit(golden_name, COMMAND_REGISTRY, topo)
        except FileNotFoundError:
            print("Golden command snapshot not found, edit history not reset.")

        if min_depth_specifier is not None:
            min_depth_specifier.value = topo.min_depth
        if trigger_refresh is not None:
            trigger_refresh()

        self._undo_history.clear()
        self._redo_history.clear()
        self.save_histories()

    def initialize(self, command_registry, topo):
        self.load_histories(command_registry, topo)
        self.replay()
