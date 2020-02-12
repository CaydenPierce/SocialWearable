#!/usr/bin/env bash

time=`date +%s`
#socat UNIX-CONNECT:/home/pi/emex/emex.sock - | nc caydenpierce.com 1234
socat - udp4-listen:5001 | nc caydenpierce.com 1234
