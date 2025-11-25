import json
import os
import numpy as np
from abc import ABC, abstractmethod
from mom6_bathy.git_utils import get_repo
from pathlib import Path
import tempfile


class CommandManager(ABC):
    def __init__(self, directory, command_registry):
        self.directory = Path(directory)
        self.repo = get_repo(directory)
        self.command_registry = command_registry
        self.history_file_path = self.directory / f"command_history.json"
        if not self.history_file_path.exists():
            with self.history_file_path.open("w") as f:
                json.dump({"Description": "Command historys"}, f)

    @abstractmethod
    def execute(self, cmd, message=None):
        """Execute a command, push it onto the undo stack, and clear the redo stack."""
        pass

    def push(self, command, message=None):
        """Add a command to the history."""

        # The command must be serializable to JSON
        command_data = command.serialize()

        self.add_to_history("head", json.dumps(command_data))
        # git add it
        rel_path = os.path.relpath(self.history_file_path, self.repo.working_tree_dir)

        self.repo.git.add(rel_path)
        # git commit it
        if message is not None:
            self.repo.git.commit("-m", f"{message}")
        else:
            self.repo.git.commit("-m", f"COMMAND")

    def parse_commit_message(self, sha):
        """
        Parse a commit message to extract command type and data.
        Expected format: "<CommandType>"
        """
        try:
            commit = self.repo.commit(sha)

            # Access the commit message
            commit_msg = commit.message
            # Split at the first colon
            cmd_type, affected_sha = commit_msg.split("-", 1)
            cmd_type = cmd_type.strip()
            affected_sha = affected_sha.strip()
            # Open history, use the shaw to get command data
            with self.history_file_path.open("r") as f:
                history = json.load(f)
            cmd_data = history[affected_sha]

            return cmd_type, affected_sha, cmd_data
        except Exception as e:
            raise ValueError(f"Invalid commit message format: {commit_msg}") from e

    def add_to_history(self, sha, command_data: str):
        """
        Ensure the command unoffical history file exists (true history is generated from commit messages), append a line to it.
        """
        # Path to history file
        # 1. Load existing history (if file exists)
        if self.history_file_path.exists():
            with self.history_file_path.open("r") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = {}
        else:
            history = {}

        # Move previous head entry to sha entry of current head
        if "head" in history:
            history[self.repo.head.commit.hexsha] = history["head"]

        # 2. Add/overwrite the SHA entry
        history[sha] = command_data

        # 3. Write back to the file
        with self.history_file_path.open("w") as f:
            json.dump(history, f, indent=2)

        return self.history_file_path

    @abstractmethod
    def undo(self):
        """Undo the last command."""
        pass

    @abstractmethod
    def redo(self):
        """Redo the last undone command."""
        pass


class TopoCommandManager(CommandManager):
    """
    The unique part here is we need to handle the topo object seperately!
    """

    def __init__(self, topo, command_registry):
        super().__init__(topo.domain_dir, command_registry)
        self._topo = topo
        self._original_topo_path = self.directory / "original_topog.nc"
        if not self._original_topo_path.exists():  # I.E. first time init
            self._topo.write_topo(self._original_topo_path)
        # Override history path to temporary because topo editing should exist across session
        self.history_file_path = self.directory / f"temp_command_history.json"

        # Initialize temp history file if it doesn't exist
        if not self.history_file_path.exists():
            with self.history_file_path.open("w") as f:
                json.dump({"Description": "Command historys"}, f)

        # If permanent history is not synced with temporary, raise an error, and copy the permanent into temp
        permanent_history_path = self.directory / f"command_history.json"
        if permanent_history_path.exists():
            with permanent_history_path.open(
                "r"
            ) as f_perm, self.history_file_path.open("r") as f_temp:
                perm_history = json.load(f_perm)
                temp_history = json.load(f_temp)
                if perm_history != temp_history:
                    print(
                        "Warning: Permanent history and temporary history are out of sync. Syncing temporary history with permanent."
                    )
                    with self.history_file_path.open("w") as f_temp_write:
                        json.dump(perm_history, f_temp_write)

    def save(self, file_name="topog.nc"):
        """Save the current topo state, and make history permanent as a git tag with the given name."""
        # First, copy over the temp history into real
        # Copy temp history to permanent history
        permanent_history_path = self.directory / f"command_history.json"
        with self.history_file_path.open("r") as src, permanent_history_path.open(
            "w"
        ) as dst:
            dst.write(src.read())
        # Now write out topo
        self._topo.write_topo(self.directory / file_name)

    def tag(self, tag_name):
        """Tag the current state of the topo."""
        self.repo.git.tag(tag_name)
        self.save(file_name=f"{tag_name}_topog.nc")

    def retrieve_tag(self, tag_name):
        """Retrieve a tagged topo state."""
        # Checkout the tag
        self.repo.git.checkout(tag_name)
        # Load the topo file
        tagged_topo_path = self.directory / f"{tag_name}_topog.nc"
        if not tagged_topo_path.exists():
            raise FileNotFoundError(
                f"Tagged topo file {tagged_topo_path} does not exist."
            )
        self._topo.set_depth_via_topog_file(tagged_topo_path)

    def checkout(self, branch_name):
        """Switch to a branch, loading the topo state from that branch."""
        self.repo.git.checkout(branch_name)

        # Replay New History
        self.reapply_changes()

    def reapply_changes(self):
        """Reapply all changes from history to the topo."""
        # Load the original topo file
        branch_topo_path = self.directory / "original_topog.nc"
        self._topo.set_depth_via_topog_file(branch_topo_path)
        for commit in self.repo.iter_commits():
            commit_sha = commit.hexsha
            try:
                cmd_type, affected_sha, cmd_data = self.parse_commit_message(commit_sha)
            except ValueError:
                continue  # skip malformed messages
            # Reconstruct and execute the command
            command_class = self.command_registry[cmd_data["type"]]
            cmd = command_class.deserialize(cmd_data)(self._topo)
            cmd()  # Execute the command without pushing

    def execute(self, cmd, message=None):
        """
        Execute a command object. If it's a user edit, push to history and clear redo.
        For system commands (undo, redo, save, load, reset), the command object handles everything.
        """
        user_edit_types = (
            "DepthEditCommand",
            "MinDepthEditCommand",
        )  # Add more as needed
        if cmd.__class__.__name__ in user_edit_types:
            cmd()
            self.push(cmd, message=message)
        else:
            raise ValueError(
                "Unsupported command type for execute. {}".format(
                    cmd.__class__.__name__
                )
            )

    def undo(self):
        # Find first commit that isn't an undo
        for commit in self.repo.iter_commits():
            commit_sha = commit.hexsha
            try:
                cmd_type, affected_sha, cmd_data = self.parse_commit_message(commit_sha)
            except ValueError:
                continue  # skip malformed messages
            if "REVERT" not in cmd_type:
                break

        command_class = self.command_registry[cmd_data["type"]]
        cmd = command_class.reverse_deserialize(cmd_data)(self._topo)
        self.execute(
            cmd, message=f"REVERT-{commit_sha}"
        )  # This is the revert right here, a revert commit

    def redo(self):
        # Redo needs to find the first revert commit and only runs if it doesn't hit a COMMAND cmd_type in the backwards iteration then takes the revert commit and reverse desearlies and has the message "REDO-<original commit sha>"
        redo_possible = True
        already_redone = {}
        for commit in self.repo.iter_commits():
            commit_sha = commit.hexsha
            try:
                cmd_type, affected_sha, cmd_data = self.parse_commit_message(commit_sha)
            except ValueError:
                continue  # skip malformed messages

            # You can't redo something if nothing is undone
            if cmd_type == "COMMAND":
                redo_possible = False
                break

            # At least one undo was found and already redone
            elif "REDO" in cmd_type:
                # We've already redone a commit
                already_redone[affected_sha] = True

            # We found an undo commit, we check if it's already redone. If not, break, If so, continue searching
            elif "REVERT" in cmd_type:
                # Parse commit sha
                if commit.hexsha in already_redone:
                    continue
                else:
                    commit_sha = commit.hexsha
                break

        if not redo_possible:
            print("No redo available.")
            return
        else:
            command_class = self.command_registry[cmd_data["type"]]
            cmd = command_class.reverse_deserialize(cmd_data)(self._topo)
            self.execute(cmd, message=f"REDO-{commit_sha}")

    def reset(self):
        """Reset the bathymetry to its original state by replaying commands from the beginning."""

        for commit in self.repo.iter_commits():
            commit_sha = commit.hexsha
            try:
                cmd_type, affected_sha, cmd_data = self.parse_commit_message(commit_sha)
            except ValueError:
                continue  # skip malformed messages

            if cmd_type == "COMMAND":
                # Reconstruct and execute the command
                command_class = self.command_registry[cmd_data["type"]]
                cmd = command_class.reverse_deserialize(cmd_data)(self._topo)
                self.execute(cmd, message=None)  # no need to commit again

    def __del__(self):
        # This runs when the object is garbage collected
        print("TopoCommandManagers is being destroyed! Writing out topo!")
        self._topo.write_topo(self.directory / "topog.nc")
