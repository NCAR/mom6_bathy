import json
import os
from CrocoDash.history_manager import EditHistory

class TopoEditHistory(EditHistory):
    def __init__(self, domain_id, snapshot_dir="snapshots"):
        super().__init__(domain_id, snapshot_dir)
        self._undo_history = []
        self._redo_history = []
        self.snapshot_dir = snapshot_dir
        self.domain_id = domain_id

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

    def undo(self, topo):
        if not self._undo_history:
            return False
        cmd = self._undo_history.pop()
        cmd.undo(topo)
        self._redo_history.append(cmd)
        return True

    def redo(self, topo):
        if not self._redo_history:
            return False
        cmd = self._redo_history.pop()
        cmd.execute(topo)
        self._undo_history.append(cmd)
        return True

    def save_histories(self):
        path = self.get_history_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([cmd.serialize() for cmd in self._undo_history], f)

    def load_histories(self, command_registry):
        path = self.get_history_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                self._undo_history = [
                    command_registry[d['type']].deserialize(d) for d in data
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

    def load_commit(self, name, command_registry):
        fname = os.path.join(self.snapshot_dir, f"{name}.json")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No snapshot named {name}")
        with open(fname, "r") as f:
            data = json.load(f)
        self._undo_history = [
            command_registry[d['type']].deserialize(d) for d in data["undo_history"]
        ]
        self._redo_history = [
            command_registry[d['type']].deserialize(d) for d in data["redo_history"]
        ]

    def replay(self, topo):
        for cmd in self._undo_history:
            cmd.execute(topo)