#!/usr/bin/env bash

time=`date +%s`
arecord -D hw:1,0 -c2 -r 48000 -f S32_LE -t wav -V mono -v /home/pi/glog/$time.wav
