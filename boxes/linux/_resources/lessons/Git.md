# Linux : Git
Git is a program that keeps track of changes to your files. It is very useful when developing code. This entire course is stored as a git "repository" on GitHub.

## [Video](https://vimeo.com/1036825331)

## Concepts
- version control
- cloning
- repository
- working with local files
- sync a live change during video
- forking(?)
- git config and logging in

## Lesson

- **TASK**: "Clone" (copy) all of the contents in the LastBlackBox GitHub repository directly to your NB3's midbrain. **It is very important that you clone the LBB repo to a specific folder ("NoBlackBoxes") in your NB3's home directory**.
> *Why?* Most of the code examples assume that the repo is stored in this location. If you prefer to put it somewhere else, then you must be comfortable modifying the "repo root" path used in the examples.
 - *code*
```bash
cd $HOME            # Navigate to your "Home" directory
mkdir NoBlackBoxes  # Create NoBlackBoxes directory
cd NoBlackBoxes     # Change to NoBlackBoxes directory  

# Clone LBB repo (only the most recent version)
git clone --depth 1 https://github.com/NoBlackBoxes/LastBlackBox
```

> You should now have a complete copy of the LBB repo on your NB3.