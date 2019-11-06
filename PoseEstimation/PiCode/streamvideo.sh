#!/usr/bin/env bash
cd ~/mjpg-streamer/mjpg-streamer-experimental
./mjpg_streamer -o "output_http.so -w ./www" -i "input_raspicam.so -fps 15" #make sure to point this to your git repo of mjpg_streamer
