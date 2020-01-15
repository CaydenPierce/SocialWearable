# Wearable Body Language Cognitive Extension

Live in a human-machine feedback loop where the computer co-processor provides you with insights into the body language being displayed around you.

This is iteration II of the project. First iteration was a proof of concept that was hacked together in one day by myself and Isaiah Ballah (@isaiahb on github)

## Setup

Install the "Lightweight OpenPose" git submodule and follow instructions on the github page [1] for install. Download the pretrained COCO weight as this is what we use.

## Credit

This is using the submodule [1] for all of the pose estimation. Thanks to Daniil Osokin and team for a great net.

The wearable is streaming video using [2]. Thanks to Rohan Sawant for this solution.

## References

[1] https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
[2] https://github.com/CT83/SmoothStream
