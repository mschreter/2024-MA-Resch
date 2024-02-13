# Basic instructions

## Installation
 
Build code in release configuration
```
git clone git@github.com:mschreter/2024-MA-Resch
cd 2024-MA-Resch
mkdir build
cd build
cmake -DDEAL_II_DIR=/home/my_deal_ii_build_folder -DCMAKE_BUILD_TYPE=Release ..
make
```

Switch to `-DCMAKE_BUILD_TYPE=Debug` for debug mode.

Run application with n=6 processes
```
mpirun -np 6 level-set_advection_reinitialization
```

## Testing

To run the tests in verbose mode execute:
```
ctest -V
```


## Code formatting

Make sure that `clang-format` is installed. To format all *.cc files in the application folder, go to the root directory of the project and execute

```
clang-format -i my_folder/*.cc
```

## Make changes to the code and push them to the online repository

Get the current master
```
git fetch origin 
git rebase origin/master
```

Change to a branch, which you would like to use for your developments
```
git checkout -b "my_branch"
```

Make your changes accordingly. If you are finished and want to commit the changes into the repo, perform the following steps: Format the files according to the section code formatting. Second, verify your changes via
```
git diff
git status
```

If you want to add all files that are marked as changed within `git status`, which have been already tracked before, you may simply execute
```
git add -u
```

If you want to add a new or any other specific file, e.g. `new.cc`, execute
```
git add new.cc
```

If you are done with adding the files to be committed, you could run `git status` once again and the relevant files should be highlighted in green now. Commit and push your added files back to the repo via
```
git commit -m "Add a useful description"
git push origin my_branch
```

Open up a pull request via the online GitHub mask.


## Rebase your changes to the current `main` branch

Let us assume you currently work on a branch `my_branch` and want to rebase your changes to the current `main` branch. **WARNING: In case you are an unexperienced git user, make sure that you make a back-up of your code before doing the following steps.**

0) Make sure you are on the correct local branch you want to rebase
```
git branch
```

1) Commit your local changes to your branch
```
git add -u
git commit -m "my changes"
```

2) Fetch the remote repository `origin`
```
git fetch origin
```

3) Rebase your code
```
git rebase -i origin/main
```
If there are no merge conflicts, you may forced push the rebased commit history of your local branch `my_branch` to the repo.
```
git push -f origin my_branch
```

In case there are merge conflicts in a file, e.g. `my_file_with_merge_conflicts.hpp` , first resolve them by changing the files and editing all the lines between`<<<<<<< HEAD` and `>>>>>>>` to obtain a clean code base. Subsequently, continue rebasing
```
git add my_file_with_merge_conflicts.hpp
git rebase --continue
```

Repeat this step up to a notifiation on successful rebasing. Finally, perform a forced push the rebased commit history of your local branch `my_branch` to the repo.
```
git push -f origin my_branch
```

