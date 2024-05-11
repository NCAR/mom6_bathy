import subprocess

def get_git_short_hash():
    """Get the short hash of the current git commit."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return "unknown"