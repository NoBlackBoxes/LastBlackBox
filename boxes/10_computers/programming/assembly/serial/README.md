# computers : programming : assembly : Serial

Send some bytes over the serial port...and get a reply

- Note: Must turn OFF hardware flow control for host terminal program

```bash
picocom -b 9600 -f x -r /dev/ttyUSB0
```

- Quit picocom with Ctl-a (the escape char), then Ctl-q

----
