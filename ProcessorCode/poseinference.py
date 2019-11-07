import sys
sys.path.insert(0, "./PoseEstimationLib")

import time
import torch
import torch.nn.parallel
import torch.optim
from PoseEstimationLib.pose.utils.osutils import mkdir_p, isfile, isdir, join
import PoseEstimationLib.pose.models as models
from scipy.ndimage import gaussian_filter, maximum_filter
import cv2
import numpy as np
import math

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def load_image(img, w, h):
    image = cv2.resize(img, (w, h))
    image = image[:, :, ::-1]  # BGR -> RGB
    image = image / 255.0
    image = image - np.array([[[0.4404, 0.4440, 0.4327]]])  # Extract mean RGB
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = image[np.newaxis, :, :, :]
    return image


def load_model(arch, stacks, blocks, num_classes, mobile, checkpoint_resume):
    # create model
    model = models.__dict__[arch](num_stacks=stacks, num_blocks=blocks, num_classes=num_classes, mobile=mobile)

    # optionally resume from a checkpoint
    if isfile(checkpoint_resume):
        print("=> loading checkpoint '{}'".format(checkpoint_resume))
        checkpoint =  torch.load(checkpoint_resume, map_location=lambda storage, loc: storage)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_resume))

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    model.eval()
    return model


def inference(model, image, device):
    input_tensor = torch.from_numpy(image).float().to(device)
    model = model.to(device)
    output = model(input_tensor)
    output = output[-1]
    output = output.data.cpu()
    kps = post_process_heatmap(output[0,:,:,:])
    return kps


def post_process_heatmap(heatMap, kpConfidenceTh=0.2):
    kplst = list()
    for i in range(heatMap.shape[0]):
        _map = heatMap[i, :, :]
        _map = gaussian_filter(_map, sigma=1)
        _nmsPeaks = non_max_supression(_map, windowSize=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        if len(x) > 0 and len(y) > 0:
            kplst.append((int(x[0]), int(y[0]), _nmsPeaks[y[0], x[0]]))
        else:
            kplst.append((0, 0, 0))

    kp = np.array(kplst)
    return kp


def non_max_supression(plain, windowSize=3, threshold=1e-6):
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))


def neckTouch(hand, neck):
    x1, y1 = hand
    x2,y2 = neck
    threshold = 20
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if distance < threshold:
        return True, distance

    return False, distance


def checkForActions(cvmat, kps, scale_x, scale_y):
    _index = 0
    leftHand = None
    rightHand = None
    neck = None
    topOfHead = None
    notch = None

    for _kp in kps:
        _index = _index + 1
        _x, _y, _conf = _kp
        if _conf > 0.2:
            # cv2.circle(cvmat, center=(int(_x * 4 * scale_x), int(_y * 4 * scale_y)), color=(0, 0, 255), radius=5)
            # cv2.putText(cvmat, str(_index), (_x, _y), cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
            if(_index == 11): rightHand = [_x, _y]
            elif(_index == 16): leftHand = [_x, _y]
            elif(_index == 10): topOfHead = [_x, _y]
            elif(_index == 9): neck = [_x, _y]
            elif(_index == 8): notch = [_x, _y]

    hand = rightHand if rightHand else leftHand
    if hand and neck:
        triggered, distance = neckTouch(hand, neck)
	cv2.putText(cvmat, str(distance), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
   	print("Distance is: {}".format(distance))

    return cvmat


def render_kps(cvmat, kps, scale_x, scale_y):
    _index = 0

    for _kp in kps:
        _index = _index + 1
        _x, _y, _conf = _kp
        if _conf > 0.2:
            cv2.circle(cvmat, center=(int(_x*4*scale_x), int(_y*4*scale_y)), color=(0,0,255), radius=5)
            # cv2.putText(cvmat, str(_index), (int(_x*4*scale_x), int(_y*4*scale_y)), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    return checkForActions(cvmat, kps, scale_x, scale_y)


def loadModel(args):
    # load checkpoint
    model = load_model(args.arch, args.stacks, args.blocks, args.num_classes, args.mobile, args.checkpoint)
    in_res_h , in_res_w = args.in_res, args.in_res
    return model, in_res_h, in_res_w

def main(args, model, in_res_h, in_res_w, image, frame):

    # do inference
    kps = inference(model, image, args.device)

    # render the detected keypoints
    cvmat = frame
    scale_x = cvmat.shape[1]*1.0/in_res_w
    scale_y = cvmat.shape[0]*1.0/in_res_h
    render_kps(cvmat, kps, scale_x, scale_y)

    cv2.imshow('x', cvmat)
    #cv2.imshow('Frame', frame)
    c = cv2.waitKey(10)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=16, type=int, metavar='N',
                        help='Number of keypoints')
    parser.add_argument('--mobile', default=False, type=bool, metavar='N',
                        help='use depthwise convolution in bottneck-block')
    parser.add_argument('--checkpoint', required=True, type=str, metavar='N',
                        help='pre-trained model checkpoint')
    parser.add_argument('--in_res', required=True, type=int, metavar='N',
                        help='input shape 128 or 256')
    parser.add_argument('--image', default='data/sample.jpg', type=str, metavar='N',
                        help='input image')
    parser.add_argument('--device', default='cuda', type=str, metavar='N',
                        help='device')
    model, in_res_h, in_res_w = loadModel(parser.parse_args())

    timeCurr = time.time()
    while True:
        main(parser.parse_args(), model, in_res_h, in_res_w)
        #print("Time taken for one image: {}\r".format(time.time() - timeCurr))
        timeCurr = time.time()
