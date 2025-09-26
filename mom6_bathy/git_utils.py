import os
import git

def get_domain_dir(grid, base_dir="Topos"):
    """
    Returns a unique directory path for a given grid object.
    Works for both rectilinear and curvilinear grids.
    """
    name = getattr(grid, "name", "unknown")
    ny = getattr(grid, "ny", None)
    nx = getattr(grid, "nx", None)
    if ny is not None and nx is not None:
        shape = f"{ny}x{nx}"
    else:
        shape = "unknownshape"
    return os.path.join(base_dir, f"domain_{name}_{shape}")

def list_domain_dirs(base_dir="Topos"):
    """
    List all domain directories in the base_dir.
    """
    if not os.path.exists(base_dir):
        return []
    return [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("domain_")
    ]

def get_repo(path):
    """
    Ensure a git repository exists at the given path. If not, initialize one.
    Returns the repo object.
    """
    if not os.path.exists(os.path.join(path, ".git")):
        repo = git.Repo.init(path)
    else:
        repo = git.Repo(path)
    return repo

def snapshot_action(action, repo_root, file_path=None, commit_msg=None, commit_sha=None):
    """
    Perform snapshot-related git actions.
    action: 'commit', 'ensure_tracked', 'restore'
    """

    repo = git.Repo(repo_root)
    if action == 'commit':
        rel_path = os.path.relpath(file_path, repo_root)
        if os.path.exists(file_path):
            repo.git.add(rel_path)
            if repo.is_dirty(untracked_files=True):
                repo.index.commit(commit_msg)
                return f"Committed snapshot '{os.path.basename(file_path)}' to Git: {commit_msg}"
            else:
                return "Nothing to commit."
        else:
            if rel_path in repo.git.ls_files().splitlines():
                repo.git.rm(rel_path)
                if repo.is_dirty(untracked_files=True):
                    repo.index.commit(commit_msg)
                    return f"Committed deletion of snapshot '{os.path.basename(file_path)}' to Git: {commit_msg}"
                else:
                    return "Nothing to commit."
            else:
                return f"Snapshot file '{file_path}' does not exist and is not tracked by git."
    elif action == 'ensure_tracked':
        rel_path = os.path.relpath(file_path, repo_root)
        try:
            repo.head.commit.tree / rel_path
            in_head = True
        except KeyError:
            in_head = False
        if not in_head:
            repo.git.add(rel_path)
            repo.index.commit(commit_msg)
            return f"Added and committed {rel_path}"
        else:
            diff = repo.git.diff('HEAD', rel_path)
            if diff:
                repo.git.add(rel_path)
                repo.index.commit(commit_msg)
                return f"Updated and committed {rel_path}"
            else:
                return f"{rel_path} already committed and up to date."
    elif action == 'restore':
        abs_path = os.path.join(repo_root, file_path)
        try:
            file_contents = repo.git.show(f"{commit_sha}:{file_path}")
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w") as f:
                f.write(file_contents)
            return True, f"Restored '{os.path.basename(file_path)}' to commit {commit_sha}."
        except git.exc.GitCommandError as e:
            return False, str(e)
    else:
        raise ValueError(f"Unknown action: {action}")

def commit_info(
    repo_root,
    file_pattern="*.json",
    root_only=True,
    change_types=("D"),
    commit_sha=None,
    file_path=None,
    mode='list',
    branch=None,
):
    """
    Retrieves and formats commit information from a Git repository for the
    Topo Editor.

    This function operates in two primary modes. In 'list' mode, it scans a
    branch's history to find commits that match specific file patterns and
    change types, returning a filtered list. This mode is particularly useful
    for populating UI elements like dropdowns for restoring files. In 'details'
    mode, it provides a formatted HTML string with detailed information about a
    single commit, for the display panels in Topo Editor.

    """
    import fnmatch
    repo_root = os.path.abspath(repo_root)
    repo = git.Repo(repo_root)
    def get_null_tree(repo):
        return repo.tree(repo.git.hash_object('-t', 'tree', '/dev/null'))
    if mode == 'list':
        if branch is None:
            branch = repo.active_branch.name
        options = []
        for commit in repo.iter_commits(branch):
            parents = commit.parents or []
            if parents:
                diffs = commit.diff(parents[0])
            else:
                diffs = commit.diff(get_null_tree(repo))
            for diff in diffs:
                path = diff.b_path
                if root_only and "/" in path:
                    continue
                if diff.change_type in change_types:
                    if fnmatch.fnmatch(path, file_pattern):
                        label = f"{commit.hexsha[:7]} - {os.path.basename(path)} - {commit.message.strip().splitlines()[0]}"
                        options.append((label, (commit.hexsha, path)))
        # Only keep files that still exist on disk
        filtered = []
        for label, (sha, path) in options:
            abs_path = os.path.abspath(os.path.join(repo_root, path))
            if os.path.exists(abs_path):
                filtered.append((label, (sha, path)))
        return filtered
    elif mode == 'details':
        commit = repo.commit(commit_sha)
        return (
            f"<b>SHA:</b> {commit.hexsha[:7]}<br>"
            f"<b>File:</b> {os.path.basename(file_path)}<br>"
            f"<b>Author:</b> {commit.author.name}<br>"
            f"<b>Date:</b> {commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"<b>Message:</b> {commit.message.strip().splitlines()[0]}"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
def get_current_branch(repo_root):
    return git.Repo(repo_root).active_branch.name

def list_branches(repo_root):
    return [head.name for head in git.Repo(repo_root).heads]

def create_branch_and_switch(branch, repo_root):
    repo = git.Repo(repo_root)
    repo.git.checkout('-b', branch)
    return repo.active_branch.name

def delete_branch_and_switch(branch, repo_root):
    repo = git.Repo(repo_root)
    if repo.active_branch.name == branch:
        raise Exception("Cannot delete the currently checked out branch. Please checkout another branch first.")
    repo.git.branch('-D', branch)
    return repo.active_branch.name

def merge_branch(repo_root, source_branch):
    """Needs to be tested further."""
    repo = git.Repo(repo_root)
    current = repo.active_branch.name
    if source_branch == current:
        return False, "Cannot merge a branch into itself."
    try:
        repo.git.merge(source_branch)
        return True, f"Merged branch '{source_branch}' into '{current}'."
    except Exception as e:
        return False, f"Merge failed: {e}"

def safe_checkout_branch(repo_root, branch, rel_dir):
    repo = git.Repo(repo_root)
    # Ignore .last_domain.json in untracked files
    untracked = [
        f for f in repo.untracked_files
        if f.startswith(rel_dir) and not f.endswith('.last_domain.json')
    ]
    changed = [item.a_path for item in repo.index.diff(None) if item.a_path.startswith(rel_dir)]
    staged = [item.a_path for item in repo.index.diff('HEAD') if item.a_path.startswith(rel_dir)]
    if untracked or changed or staged:
        print("Cannot checkout: You have unsaved or uncommitted changes in 'snapshots'. Please save and commit them before switching branches.")
        if untracked:
            print("Untracked files:", untracked)
        if changed:
            print("Unstaged changes:", changed)
        if staged:
            print("Staged but uncommitted:", staged)
        return False, untracked, changed, staged
    repo.git.checkout(branch)
    return True, [], [], []