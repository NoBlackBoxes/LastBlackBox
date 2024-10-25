# Linux : SSH

Secure shell terminal.

## SSH Tunnel Server

***THIS DOES NOT WORK***

You will need a virtual private server (VPS) in the cloud, basically a computer somewhere with a public IP address.

On your VPS, check that A

```bash
systemctl start sshd # ...if SSH service is not running
sshd -T
```

```bash
sudo nano /etc/ssh/sshd_config
# Add "GatewayPorts yes" to end of file
sudo systemctl restart ssh


# On Pi
ssh -nNTv -R 0.0.0.0:1122:localhost:22 my-username@my-vps-hostname-or-ip
```

