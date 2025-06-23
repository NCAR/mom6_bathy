import os
import git

def is_path_within_repo(repo_root, file_path):
    abs_repo = os.path.abspath(repo_root)
    abs_file = os.path.abspath(file_path)
    try:
        return os.path.commonpath([abs_repo, abs_file]) == abs_repo
    except ValueError:
        return False

def git_snapshot_action(action, repo_root, file_path=None, commit_msg=None, commit_sha=None):
    """
    Perform snapshot-related git actions.
    action: 'commit', 'ensure_tracked', 'restore'
    """
    # SKIP git operations if file_path is outside the repo
    if file_path is not None and not is_path_within_repo(repo_root, file_path):
        print(f"[git_utils] Skipping git operation for file outside repo: {file_path}")
        if action == 'restore':
            return False, "File is outside repo"
        return "Skipped git operation (file outside repo)."

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

def git_commit_info(repo_root, rel_dir=None, commit_sha=None, file_path=None, mode='list', branch=None):
    """
    mode: 'list' for listing commits affecting a dir, 'details' for commit details of a file.
    """
    repo = git.Repo(repo_root)
    if mode == 'list':
        if branch is None:
            branch = repo.active_branch.name
        commits = list(repo.iter_commits(branch, paths=rel_dir))
        options = []
        for commit in commits:
            files = [f for f in commit.stats.files if f.startswith(rel_dir)]
            for f in files:
                options.append((
                    f"{commit.hexsha[:7]} - {os.path.basename(f)} - {commit.message.strip().splitlines()[0]}",
                    (commit.hexsha, f)
                ))
        return options
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

def git_get_current_branch(repo_root):
    return git.Repo(repo_root).active_branch.name

def git_list_branches(repo_root):
    return [head.name for head in git.Repo(repo_root).heads]

def git_create_branch_and_switch(branch, repo_root):
    repo = git.Repo(repo_root)
    repo.git.checkout('-b', branch)
    return repo.active_branch.name

def git_delete_branch_and_switch(branch, repo_root):
    repo = git.Repo(repo_root)
    if repo.active_branch.name == branch:
        raise Exception("Cannot delete the currently checked out branch. Please checkout another branch first.")
    repo.git.branch('-D', branch)
    return repo.active_branch.name

def git_merge_branch(repo_root, source_branch):
    repo = git.Repo(repo_root)
    current = repo.active_branch.name
    if source_branch == current:
        return False, "Cannot merge a branch into itself."
    try:
        repo.git.merge(source_branch)
        return True, f"Merged branch '{source_branch}' into '{current}'."
    except Exception as e:
        return False, f"Merge failed: {e}"

def git_safe_checkout_branch(repo_root, branch, rel_dir):
    repo = git.Repo(repo_root)
    untracked = [f for f in repo.untracked_files if f.startswith(rel_dir)]
    changed = [item.a_path for item in repo.index.diff(None) if item.a_path.startswith(rel_dir)]
    staged = [item.a_path for item in repo.index.diff('HEAD') if item.a_path.startswith(rel_dir)]
    if untracked or changed or staged:
        return False, untracked, changed, staged
    repo.git.checkout(branch)
    return True, [], [], []