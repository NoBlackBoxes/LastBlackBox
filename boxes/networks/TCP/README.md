# Setting up a basic TCP server

Two .c scripts have been provided to set up a basic server/client relationship.  piServerBasic.c should be compiled and run on your Robot Pi (or whatever machine you are sending instructions too), while piClientBasic.c should be compiled and run on your Computer Pi (which evern machine you will be sencind instructions from).  These programs only allow for sending a single instruction from the client to the server before the connection is shut down.  You will therefore want to edit these scripts to make them fir for purpose, but the basic structure (and the establishing of sockets) will hold.

Use gcc or g++ to compile the script - both should already be installed on your Pi s, otherwise use `sudo apt-get install g++` to install them.

```g++ -o piServerBasic piServerBasic.c```

Will compile the file `piServerBasic.c` to the output file `piServerBasic`.  This output file is then executed by typing `/path/to/executable/piServerBasic` into the terminal.  If you are in the same directory, this is `./piServerBasic`.  If there is an error in your code that you wish to debug, make sure you include the `-g` flag when compiling your script.  This includes debug symbols which can then be used by [`gdb`](https://www.tutorialspoint.com/gnu_debugger/index.htm "gdb")

`piServerBasic.c` takes one argument - the socket number you wish to open.  Run this script first on your server.  Then run `piClientBasic.c` on your client machine with two arguments - `<hostname> <socket_number>`.  Make sure the socket number is the same as the server script.  You can find the hostname of your server with the `hostname` command in terminal (on your images it should be `LBB`).  Alternately, you can provide the IP address of the server.