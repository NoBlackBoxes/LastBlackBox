# Linux : SSH

Secure shell.

## SSH Tunnel Server

You will need a virtual private server (VPS) in the cloud, basically a computer you can use that has a public IP address.

Connect to your VPS and check that your sshd config has "AllowTcpForwarding yes" and "GatewayPorts clientspecified".

```bash
sudo vim /etc/ssh/sshd_config
```

```text
# Uncomment and/or modify the file such that the following lines exist
AllowTcpForwarding yes
GatewayPorts clientspecified
```

```bash
sudo systemctl restart ssh
```

Create a (sudo) user account on your VPS

```bash
sudo adduser <your usenamer>
sudo usermod -aG sudo <your usenamer>
mkdir ~/.ssh
chmod 700 ~/.ssh
```

On your NB3, create a new security key for **your user** and copy it to your VPS

```bash
ssh-keygen -t ed25519 -C <your usenamer>        # Generate Key
cat /home/<your usenamer>/.ssh/id_ed25519.pub   # Print key       
# Copy to server's ~/.ssh/authorized_keys file
```

On your local client, create a new security key for your user and copy it to your VPS

```bash
ssh-keygen -t ed25519 -C <your usenamer>        # Generate Key
cat /home/<your usenamer>/.ssh/id_ed25519.pub   # Print key       
# Copy to server's ~/.ssh/authorized_keys file
```

On your NB3, connect to the tunnel (which uses port 1122) on your VPS
```bash
ssh -nNTv -R 0.0.0.0:1122:localhost:22 <your username>@<your VPS IP address>
```

You now should be able to connect to your NB3 from ***outside*** your local network (via SSH on port 1122).

```bash
ssh <your usenamer>@<your VPS IP address>
```

To make this persistent (to reconnect the NB3 to the VPS server's tunnel if it ever disconnects), then you can use "autossh".

```bash
# Install autossh (NB3)
sudo apt-get install autossh

# Run autossh (-f option runs autossh in background), include unused monitoring port
autossh -f -M 33344 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -nNTv -R 0.0.0.0:1122:localhost:22 <your VPS IP address>
```

> Note: you can add more tunnel ports with additional -R options(e.g. -R 0.0.0.0:1234:localhost:80)

To automate running autossh on startup, use systemd.

Create an autossh systemd unit file
```bash
sudo vim /lib/systemd/system/autossh.service
```

..with this content
```text
[Unit]
Description=Keeps a tunnel to kampff.org for SSH on port 1122 to 22 and HTTP on port 1234 to 1234
Wants=network-online.target
After=network-online.target

[Service]
User=kampff
Type=simple
Restart=on-failure
RestartSec=3
Environment="AUTOSSH_GATETIME=0"
ExecStart=/usr/bin/autossh -M 33344 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -nNTv -R 0.0.0.0:1122:localhost:22 -R 0.0.0.0:1234:localhost:1234 <your-VPS-IP or Domain>

[Install]
WantedBy=multi-user.target
```

Copy Unit (service) file systemd can find it (not sure this is correct)
```bash
sudo cp /lib/systemd/system/autossh.service /etc/systemd/system/autossh.service
```

Check that autossh service is OK (running)
```bash
sudo systemctl daemon-reload
sudo systemctl enable autossh.service
sudo systemctl start autossh
```

Check logs if something goes wrong
```bash
journalctl -u autossh
```