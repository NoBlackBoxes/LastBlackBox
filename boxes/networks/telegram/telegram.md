## Using the Telegram bot to send your IP address

In this folder there is a shell script called `send_telegram.sh`. It can be used to send your NB3's IP address (or any other message) to the Telegram group using the Telegram bot that Eirinn created (@bootcamp_message_bot)

Copy the file to your home directory, and use chmod to make it executable:

    cp send_telegram.sh ~/
    cp send_telegram.service ~/
    cd ~/
    chmod +x send_telegram.sh

To send your IP address to the Telegram group:

    ./send_telegram.sh

To send a different message:

    ./send_telegram.sh "Hello world!"

To change your bot's name (used in the IP address message), you can set an environment variable:

    export MYNAME=MyRobotName

To make this name persistent after rebooting, add that export line to the end of your bashrc file (via `nano ~/.bashrc`)

## Making this script run on startup

That's what the service file is for. It contains info to turn this script into a service which runs 30 seconds after startup (the delay is to allow the internet to connect).

    sudo systemctl enable send_telegram.service

## Sending messages to a specific Telegram user

Currently, the chatID of the message is hard-coded in the script file (send_telegram.sh) but it can be easily modified to send to a different chatID.

### Get your chat ID

Send a message to @get_id_bot on Telegram. It will respond with your chatID.

### Authorise the Bootcamp Telegram bot to talk to your chatID

This is for spam prevention. You need to message @bootcamp_message_bot once. Doesn't matter what you said. It won't respond, but you will be authorised.

That's it: your modified send_telegram.sh can send a message to you.