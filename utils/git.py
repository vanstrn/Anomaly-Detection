#!/usr/bin/env python
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
"""
File for monitoring git status of repository while running experiments.
Required packages:
    GitPython - `pip install gitpython`
"""
# ---------------------------------------------------------------------------

import git
from shutil import copyfile


def GetCurrentCommitHash(repo):
    return repo.head.object.hexsha


def GetCurrentBranch(repo):
    return repo.active_branch


def SaveCurrentGitState(saveLocation,saveUntrackedFiles=False):
    #Creating a path to sace untracked files.
    repo = git.Repo()
    repoDifferences,untrackedFiles = GetCurrentGitState(repo)
    file = open(saveLocation+"/git_status.txt", "w")
    file.write("Current branch: "+str(GetCurrentBranch(repo))+"\n")
    file.write("Current commit: "+GetCurrentCommitHash(repo)+"\n\n")
    if len(untrackedFiles)>0:
        file.write("There are untracked files in the repository:\n")
        for untrackedFile in untrackedFiles:
            file.write(untrackedFile+"\n")
    else:
        file.write("All files are tracked in the repository:\n")
    #Creating file to stor list of untracked files and git differences.
    file.write("Repository Differences:\n")
    file.write(repoDifferences)
    #Copying untracked files
    if saveUntrackedFiles:
        dst = saveLocation+"/untrackedFiles"
        if not os.path.exists(dst):
                os.makedirs(dst)
        for file in untrackedFiles:
            copyfile(file, dst)
    file.close()


def GetCurrentGitState(repo):
    """Returns:
     str: detailed description of a tracked differences
     list: list of all untracked files. """

    #Getting list of untracked files:
    untrackedFiles = repo.untracked_files

    #Getting a summary of the tracked file differences.
    repoDifferences = repo.git.diff()

    return repoDifferences,untrackedFiles


if __name__ == "__main__":
    SaveCurrentGitState("test")
