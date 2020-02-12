#!/usr/bin/env bash

time=`date +%s` #get current time
echo "Saving to: $1/$time.h264" #echo the name of the file
netcat -l 1234 | tee $1/$time.h264 #| cat > emex #stream it in
MP4Box -add $1/$time.h264:fps=24 $1/$time.mp4 #save an mp4 version
