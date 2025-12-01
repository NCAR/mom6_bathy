import json
import os
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from mom6_bathy.git_utils import get_repo
from pathlib import Path
import tempfile


class CommandType(Enum):
    """Enumeration for command types in the history."""

    INITIAL = "INITIAL"
    COMMAND = "COMMAND"
    UNDO = "UNDO"
    REDO = "REDO"


class CommandManager(ABC):
    def __init__(self, directory, command_registry):
        self.directory = Path(directory)
        self.repo = get_repo(directory)
        self.command_registry = command_registry

        self.history_file_path = self.directory / f"command_history.json"
        if not self.history_file_path.exists():
            self.history_dict = {"Description": "Command history"}
            self.write_history()
        else:
            with self.history_file_path.open("r") as f:
                self.history_dict = json.load(f)

    def write_history(self):
        """Write the current history dictionary to the history file."""
        with self.history_file_path.open("w") as f:
            json.dump(self.history_dict, f, indent=2)

    def load_history(self):
        """Load the history dictionary from the history file."""
        with self.history_file_path.open("r") as f:
            history_dict = json.load(f)
        self.history_dict = history_dict

    @abstractmethod
    def execute(self, cmd, message=None):
        """Execute a command, push it onto the undo stack, and clear the redo stack."""
        pass

    def push(self, command, cmd_type: CommandType, command_message=None):
        """Add a command to the history."""

        # The command must be serializable to JSON
        command_data = command.serialize()

        self.add_to_history("head", json.dumps(command_data))
        # git add it
        rel_path = os.path.relpath(self.history_file_path, self.repo.working_tree_dir)

        self.repo.git.add(rel_path)

        self.repo.git.commit(
            "-m",
            f"{cmd_type.value}-{command_message if command_message is not None else ''}",
        )

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
            if "-" in commit_msg:
                cmd_type, affected_sha = commit_msg.split("-", 1)
                affected_sha = affected_sha.strip()
            else:
                cmd_type = commit_msg.strip()
                affected_sha = None
            cmd_type = cmd_type.strip()

            # write  acheck that if the shaw it's head instead we use key head
            if sha == self.repo.head.commit.hexsha:
                sha = "head"
            cmd_data = json.loads(self.history_dict[sha])

            return cmd_type, affected_sha, cmd_data
        except Exception as e:
            raise ValueError(f"Invalid commit message format: {commit_msg}") from e

    def add_to_history(self, sha, command_data: str):
        """
        Ensure the command unoffical history file exists (true history is generated from commit messages), append a line to it.
        """

        # Move previous head entry to sha entry of current head
        if "head" in self.history_dict:
            self.history_dict[self.repo.head.commit.hexsha] = self.history_dict["head"]

        # 2. Add/overwrite the SHA entry
        self.history_dict[sha] = command_data

        # 3. Write back to file
        self.write_history()

        # Return path to history file
        return self.history_file_path

    @abstractmethod
    def undo(self):
        """Undo the last command."""
        pass

    @abstractmethod
    def redo(self):
        """Redo the last undone command."""
        pass

    def _history_state(self):
        """
        Build a linear logical history of commands and their undo/redo state.
        Returns a dict: sha -> state dict.
        """
        self.load_history()
        state = {}  # sha -> {cmd_data, applied, undone, redone}
        state["undone_order"] = []

        for commit in self.repo.iter_commits(reverse=True):
            sha = commit.hexsha

            try:
                cmd_type, affected_sha, cmd_data = self.parse_commit_message(sha)
            except ValueError:
                continue

            if cmd_type == "COMMAND":
                state[sha] = dict(
                    cmd_data=cmd_data,
                    applied=True,
                )

            elif cmd_type == "UNDO":
                # this UNDO targets affected_sha
                state["undone_order"].append(affected_sha)
                if affected_sha in state:
                    state[affected_sha]["applied"] = False

            elif cmd_type == "REDO":
                if affected_sha in state:
                    state[affected_sha]["applied"] = True
                state["undone_order"].remove(affected_sha)

        return state

    def get_current_branch(self):
        return self.repo.active_branch.name

    def list_branches(self):
        return [head.name for head in self.repo.heads]

    def create_branch(self, branch):
        self.repo.create_head(branch)

    def get_tag_names(self):
        return [tag.name for tag in self.repo.tags]
    
    def cleanup_history(self):
        """Drop all commits newer than the last commit recorded in history_dict."""
        self.load_history()

        # history_dict keys = SHAs of all valid commits
        valid_shas = set(self.history_dict.keys())

        # Walk newest → oldest until we find the newest valid SHA
        last_valid_sha = None
        for commit in self.repo.iter_commits():  # default: newest → oldest
            if commit.hexsha in valid_shas:
                last_valid_sha = commit.hexsha
                break

        # Reset to the last valid commit (deletes all newer ones)
        self.repo.git.reset("--hard", last_valid_sha)


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
            self.history_dict = {"Description": "Temporary Command History"}
            self.write_history()
        self.load_history()

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
        self._topo.set_depth_via_topog_file(branch_topo_path, quietly=True)
        state = self._history_state()
        for commit in state:
            if type(state[commit]) == dict and state[commit]["applied"]:
                command_class = self.command_registry[state[commit]["cmd_data"]["type"]]
                cmd = command_class.deserialize(state[commit]["cmd_data"])(self._topo)
                cmd()  # No need to execute again, just replay

    def execute(self, cmd, cmd_type: CommandType = CommandType.COMMAND, message=None):
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
            self.push(
                cmd,
                cmd_type=cmd_type,
                command_message=message if message is not None else cmd.message,
            )
        else:
            raise ValueError(
                "Unsupported command type for execute. {}".format(
                    cmd.__class__.__name__
                )
            )

    def undo(self, check_only=False):
        # Find first commit that isn't an undo
        state = self._history_state()
        can_undo = False
        for commit in self.repo.iter_commits():
            commit_sha = commit.hexsha
            if (
                commit_sha in state
            ):  # i.e. it is a command (undo and redo's are not stored)
                if state[commit_sha]["applied"]:
                    cmd_data = state[commit_sha]["cmd_data"]
                    can_undo = True
                    break

        if can_undo and not check_only:
            command_class = self.command_registry[cmd_data["type"]]
            cmd = command_class.reverse_deserialize(cmd_data)(self._topo)
            self.execute(
                cmd, cmd_type=CommandType.UNDO, message=f"{commit_sha}"
            )  # This is the revert right here, a revert commit
        return can_undo

    def redo(self, check_only=False):
        # Redo needs to find the first revert commit and only runs if it doesn't hit a COMMAND cmd_type in the backwards iteration then takes the revert commit and reverse desearlies and has the message "REDO-<original commit sha>"
        can_redo = False
        state = self._history_state()
        if len(state["undone_order"]) > 0:
            can_redo = True

        if can_redo and not check_only:
            cmd_data = state[state["undone_order"][-1]]["cmd_data"]
            command_class = self.command_registry[cmd_data["type"]]
            cmd = command_class.deserialize(cmd_data)(self._topo)
            self.execute(
                cmd, cmd_type=CommandType.REDO, message=f"{state['undone_order'][-1]}"
            )
        return can_redo

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
                cmd()  # No need to execute again, just replay
