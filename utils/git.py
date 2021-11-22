
"""Required packages:
    GitPython - `pip install gitpython`
"""
import git
from .utils import CreatePath
from shutil import copyfile

def GetCurrentCommitHash(repo):
    return repo.head.object.hexsha

def GetCurrentBranch(repo):
    return repo.active_branch


def SaveCurrentGitState(saveLocation,saveUntrackedFiles=False):
    #Creating a path to sace untracked files.
    repo = git.Repo()
    CreatePath(saveLocation)
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
        CreatePath(saveLocation+"/untrackedFiles")
        for files in untrackedFiles:
            copyfile(files, dst)
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

def GetCurrentGitSummary(repo):
    """Returns a two lists showing the names of dirty and untracked files."""

    #Getting list of untracked files:
    untrackedFiles = repo.untracked_files

    #Getting a summary of the tracked file differences.
    dirtyFiles  = []
    for file in repo.index.diff(None):
        dirtyFiles.append(file.a_path)

    return dirtyFiles, untrackedFiles


"""
Other useful git commands that I've found:

    `repo.index.diff(None)`
        Returns a list of the files which contain differences
"""
if __name__ == "__main__":
    SaveCurrentGitState("test")
