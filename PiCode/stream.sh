#!/usr/bin/env bash

time=`date +%s`
raspivid -t 0 -w 1280 -h 720 -fps 24 -b 5000000 -o - | tee ~/glog/$time.h264 | nc caydenpierce.com 1234
