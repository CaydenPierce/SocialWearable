# Wearable Social Intelligence Cognitive Extension

Enhance your social intelligence by living in a human-machine feedback loop where the computer co-processor provides you with insights into the non-verbal communication being displayed around you. The system currently does body language and facial expression decoding to provide you with heads up notifications about the social situation that's going on around you. This can also be used to emotionally tag mmemories.


This runs on a setup where the ProcessorCode folder are the server scripts, and the PiCode are the scripts to run on a Pi based wearable. I have my domain name hardcoded here, feel free to move that to a config.py and setup your own server.

## Setup

-Install the "Lightweight OpenPose" git submodule and follow instructions on the github page [1] for install. Download the pretrained COCO weight as this is what we use. Place it in the root folder of the openpose git submodule.  
Install the "ResidualMaskingNetwork Facial Expression" neural net library from [4]. This also is not yet a proper submodule. It is basically stock except I have changed the ssd_infer.py script to run with the rest of this library. Follow the instruction at [3] "Live Demo" to setup the inference (use the latest pretrained net).  
-Run ProcessorCode/bodylanguage.py on your processor.  
-Run PiCode/main.py on your PiBased wearable.  

That should be all.  

## Credit

This is using the submodule [1] for all of the pose estimation. Thanks to Daniil Osokin and team for a great net.

The wearable is streaming video using [2]. Thanks to Rohan Sawant for this solution. Still hacking away at this, will properly implement this as a submodule soon.

## References

[1] https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch  
[2] https://github.com/CT83/SmoothStream  
[3] Navarro, Julia, and Marvin Karlins. What every body is saying. HarperCollins Publishers, 2008.  
[4] https://github.com/phamquiluan/ResidualMaskingNetwork#benchmarking  
