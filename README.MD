# Wearable Social Intelligence Cognitive Extension

Enhance your social intelligence by living in a human-machine feedback loop where the computer co-processor provides you with insights into the non-verbal communication being displayed around you. The system currently does body language and facial expression decoding to provide you with heads up notifications about the social situation that's going on around you. This can also be used to emotionally tag mmemories.

This runs on a setup where the ProcessorCode folder are the server scripts, and the PiCode are the scripts to run on a Pi based wearable. I have my domain name hardcoded here, feel free to move that to a config.py and setup your own server.

## Setup

-Install the "Lightweight OpenPose" git submodule and follow instructions on the github page [1] for install. Download the pretrained COCO weight as this is what we use. Place it in the root folder of the openpose git submodule.  
-Install the "ResidualMaskingNetwork Facial Expression" neural net library from [4]. Follow the instruction from [3] at sectoin "Live Demo" to setup the inference (download the models and put them where the github specifies).  
-Run ProcessorCode/receiveStream.sh on your processor.  
-Run PiCode/stream.sh on your Pi-based wearable.  
-The wearable is streaming video the bash scripts in PiCode and Processor code. 24FPS 720P over the internet getting saved to a file and piped into python to be processed (thanks to OpenCV).
-create a named pipe in the Processor code section for netcat to stream into and OpenCV to then stream out of:
```
mknod emex p
```
or
```
mkfif emex
```
  
That should be all.  

## Credit and References

This is using the submodule [1] for all of the pose estimation. Thanks to Daniil Osokin and team for a great net. Also thanks to Luan Pham and team for the facial expression neural network.

[1] https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch  
[2] Navarro, Julia, and Marvin Karlins. What every body is saying. HarperCollins Publishers, 2008.  
[3] https://github.com/phamquiluan/ResidualMaskingNetwork#benchmarking  
