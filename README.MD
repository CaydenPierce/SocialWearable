# Wearable Body Language Cognitive Extension

Live in a human-machine feedback loop where the computer co-processor provides you with insights into the body language being displayed around you.

This runs on a setup where the ProcessorCode folder are the server scripts, and the PiCode are the scripts to run on a Pi based wearable. I have my domain name hardcoded here, feel free to move that to a config.py and setup your own server.

## Setup

Install the "Lightweight OpenPose" git submodule and follow instructions on the github page [1] for install. Download the pretrained COCO weight as this is what we use. Place it in the root folder of the openpose git submodule.

Run ProcessorCode/bodylanguage.py on your processor.

Run PiCode/main.py on your PiBased wearable.

That should be all.

## Credit

This is using the submodule [1] for all of the pose estimation. Thanks to Daniil Osokin and team for a great net.

The wearable is streaming video using [2]. Thanks to Rohan Sawant for this solution. Still hacking away at this, will properly implement this as a submodule soon.

## References

[1] https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
[2] https://github.com/CT83/SmoothStream
