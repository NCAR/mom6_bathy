import json
import os
import numpy as np
from abc import ABC, abstractmethod
from mom6_bathy.git_utils import get_repo
from pathlib import Path


class CommandManager(ABC):
    def __init__(self, domain_id, directory, command_registry):
        self.directory = Path(directory)
        self.domain_id = domain_id
        self.repo = get_repo(directory)
        self.command_registry = command_registry

    @abstractmethod
    def execute(self, cmd, message=None):
        """Execute a command, push it onto the undo stack, and clear the redo stack."""
        pass

    def push(self, command, message=None):
        """Add a command to the history."""

        # The command must be serializable to JSON
        command_data = command.serialize()
        history_file_path = add_to_unofficial_history(self, command_data)

        # git add it
        rel_path = os.path.relpath(history_file, self.repo.working_tree_dir)
        self.repo.git.add(rel_path)
        # git commit it
        if message is not None:
            self.repo.git.commit("-m", f"{message}: {json.dumps(command_data)}")
        else:
            self.repo.git.commit("-m", f"COMMAND: {json.dumps(command_data)}")

    def parse_commit_message(self, commit_msg: str):
        """
        Parse a commit message to extract command type and data.
        Expected format: "<CommandType>: <JSON data>"
        """
        try:
            if ":" not in commit_msg:
                raise ValueError("Commit message missing ':' separator")

            # Split at the first colon
            cmd_type, json_str = commit_msg.split(":", 1)
            cmd_type = cmd_type.strip()
            json_str = json_str.strip()

            # Parse JSON
            data = json.loads(json_str)
            return cmd_type, data
        except Exception as e:
            raise ValueError(f"Invalid commit message format: {commit_msg}") from e

    def add_to_unofficial_history(self, json_line: str):
        """
        Ensure the command unoffical history file exists (true history is generated from commit messages), append a line to it.
        """
        import os

        # Path to history file
        history_file = self.directory / f"command_history_{self.domain_id}.json"

        # 1. Create file if missing
        if not history_file.exists():
            history_file.touch()

        # 2. Append the line
        with history_file.open("a") as f:
            f.write(json.dumps(json_obj) + "\n")

        return history_file

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
    def __init__(self, domain_id, topo, command_registry):
        super().__init__(domain_id, topo.domain_dir, command_registry)
        self._topo = topo

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
            message = commit.message.strip()
            try:
                cmd_type, cmd_data = self.parse_commit_message(message)
            except ValueError:
                continue  # skip malformed messages
            if "REVERT" not in cmd_type:
                commit_sha = commit.hexsha  # <- the SHA of this commit
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
            message = commit.message.strip()
            try:
                cmd_type, cmd_data = self.parse_commit_message(message)
            except ValueError:
                continue  # skip malformed messages

            # You can't redo something if nothing is undone
            if cmd_type == "COMMAND":
                redo_possible = False
                break

            # At least one undo was found and already redone
            elif "REDO" in cmd_type:
                # We've already redone a commit
                original_sha = cmd_type.split("-")[1]
                already_redone[original_sha] = True

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

        for commit in commits:
            message = commit.message.strip()
            try:
                cmd_type, cmd_data = self.parse_commit_message(message)
            except ValueError:
                continue  # skip malformed messages

            if cmd_type == "COMMAND":
                # Reconstruct and execute the command
                command_class = self.command_registry[cmd_data["type"]]
                cmd = command_class.reverse_deserialize(cmd_data)(self._topo)
                self.execute(cmd, message=None)  # no need to commit again
