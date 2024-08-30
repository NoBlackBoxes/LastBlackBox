# Linux : Arch : Freeing Space

Linux gives you lots of tools for freeing up space on a crowded hard drive.

## NCDU

Simply the best tool ever. This command will tell you where on your drive (from wherever you run the command) the largest folders/files are stored.

```bash
ncdu
```

## Cleaning Cache

Many files are "cached" by package managers. This allows reverting to previous versions if something goes wrong and skipping redundant downloads. However, these cached files can accumulate and take up *tons* of space on your hard drive. If everything is running smoothly, then you are often safe to purge these cached files and free up space.

On Arch, we can keep only the latest cached versions for pacman packages using the following command. *Note*: you will have to install the "pacman-contrib" package to get this command.

```bash
paccache -rk1
```

Python's PIP also caches packages. You can remove the cached versions (completely) using the following command.

```bash
pip cache purge
# You will also need to do this inside any large virtual environments you may be using
```
