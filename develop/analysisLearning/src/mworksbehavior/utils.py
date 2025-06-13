import re
from pathlib import Path


def git_info(module):
    """Return some info about git repository status of module"""
    repo = git.Repo(module.__file__, search_parent_directories=True)
    return str(repo.active_branch)
