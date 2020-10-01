#! /bin/bash
TOKEN=1259800835:AAFNOTBhoP-6sY14_AKnOUawGU7EBMAjrP8 # the telegram bot's token
CHAT_ID=-1001477028322 # the LBB Telegram group to send the message to
MESSAGE=${1-"${MYNAME:-Anonymous machine} is online at `hostname -I`"}
URL="https://api.telegram.org/bot$TOKEN/sendMessage"

curl -s -X POST $URL -o /dev/null -d chat_id=$CHAT_ID -d text="$MESSAGE" -d disable_notification="true"
exit 0