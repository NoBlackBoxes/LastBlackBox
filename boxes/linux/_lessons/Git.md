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

## Connections

## Lesson

- [ ] **Task**: "Clone" (copy) all of the code in the LastBlackBox GitHub repository directly to your NB3's midbrain. It will help with later exercises if we all put this example code at the same location on the Raspberry Pi (the "home" directory).

```bash
cd ~                # Navigate to "home" directory
mkdir NoBlackBoxes  # Create NoBlackBoxes directory
cd NoBlackBoxes     # Change to NoBlackBoxes directory

# Clone LBB repo (only the most recent version)
git clone --depth 1 https://github.com/NoBlackBoxes/LastBlackBox
```
