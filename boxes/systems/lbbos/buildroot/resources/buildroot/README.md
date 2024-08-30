# LastBlackBox : systems : OS : resources : buildroot

*buildroot* is a make-based system for building embedded Linux operating systems

## Step-by-step

- Clone the repo or use a stable branch/tarball
- Use the external buildroot tree located in the LBB repo
- Set output directory and run *_defconfig
- Change to output directory and build

## Notes

- Using glibc because ulibc has some issue with openssl (ucontext?)
  - Fortran support required for building lapacke
- Adding "user_tables" requires a newline after the entries, there may be some issue of LF vs CRLF line endings
- Users are defined in the /etc/passwd file
- Groups are defined in the /etc/groups file
- Passwords are currently stored in plain-text, but should be encoded and/or set to expire on first use
  - Password can be encoded using SHA-256 using python script (google it) and then stored in the /etc/shadow file

## To Do

- Init scripts for kernel module loading amongst other things
- Set date/time
- WiFi setup and other DHCP or static network config
