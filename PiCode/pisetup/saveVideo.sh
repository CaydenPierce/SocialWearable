#!/usr/bin/env bash

time=`date +%s`
#raspivid -t 0 -w 1280 -h 720 -fps 24 -b 3000000 -o - | tee /home/pi/glog/$time.h264 | socat UNIX-LISTEN:/home/pi/emex/emex.sock,reuseaddr,fork -
#raspivid -ih -fl -t 0 -w 1280 -h 720 -fps 24 -b 3000000 -o - | tee /home/pi/glog/$time.h264 | socat UNIX-LISTEN:/home/pi/emex/emex.sock,reuseaddr,fork -
raspivid -ih -fl -t 0 -w 1280 -h 720 -fps 24 -b 3000000 -o - | tee /home/pi/glog/$time.h264 | socat - udp-sendto:127.0.0.1:5001
