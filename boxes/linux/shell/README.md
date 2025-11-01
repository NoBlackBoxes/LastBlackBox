# Linux : Shell
The "shell" is the program you interact with when you open a new terminal window. You can type commands at the prompt and receive feedback in beautiful (sometimes even colorful!) text.

Typing long commands can be tedious and error prone. There are many built-in shortcuts for making your life easier (tab completion, command history, etc.), but you can go even further! 

Linux shells allow adding custom *variables* and creating short *aliases* that allow you to run your most common long complex commands with just a few keystrokes.

## Creating a custom Shell
Custom variables and commands can be placed in a file called **.bashrc**. The "rc" stands for "run commands" and simply lists the commands the shell will run for you every time you open a new terminal window.

```bash
# Default Commands
# ----------------
[[ $- != *i* ]] && return # If not running interactively, don't do anything

alias ls='ls --color=auto'
alias grep='grep --color=auto'
PS1='[\u@\h \W]\$ '

# LBB Environment Variables
# -------------------------
export LBB="${HOME}/NoBlackBoxes/LastBlackBox"
# This creates a variable named LBB that you can use to access the root directory of the repo
# from anywhere via the command line: e.g. "cd $LBB"

# LBB Aliases
# -----------
alias Activate="source ${LBB}/_tmp/LBB/bin/activate" # Activate LBB python virtual environment

# Cosmetics
# ---------
PS1='\[\e[1;32m\]\u@\[\e[0;33m\]\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\$ ' # Fancy colored prompt
```

## Installing a custom Shell profile
Create a **.bash_profile** file in your "HOME" folder with content similar to shown above or simply copy the example here to correct location using the command below.

```bash
cd ${HOME}/NoBlackBoxes/LastBlackBox/boxes/linux/shell
cp .bashrc ${HOME}/.
source ${HOME}/.bashrc
```
