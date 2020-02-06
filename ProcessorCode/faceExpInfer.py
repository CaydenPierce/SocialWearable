import os
import glob
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from ResidualMaskingNetwork.models import densenet121, resmasking_dropout1


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


FER_2013_EMO_DICT = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}


def load():
    # load configs and set random seed
    net = cv2.dnn.readNetFromCaffe("./ResidualMaskingNetwork/deploy.prototxt.txt", "./ResidualMaskingNetwork/res10_300x300_ssd_iter_140000.caffemodel")
    configs = json.load(open('./ResidualMaskingNetwork/configs/fer2013_config.json'))
    image_size = (configs['image_size'], configs['image_size'])

    # model = densenet121(in_channels=3, num_classes=7)
    model = resmasking_dropout1(in_channels=3, num_classes=7)
    model.cuda()

    state = torch.load('./ResidualMaskingNetwork/saved/checkpoints/Z_resmasking_dropout1_rot30_2019Nov30_13.32')
    model.load_state_dict(state['net'])
    model.eval()
    return model, net, image_size
   
def infer(frame, model, image_size, net): #pass me cv2 frame
    with torch.no_grad():
        try:
            frame = np.fliplr(frame).astype(np.uint8)
            # frame += 50
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = frame

            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            faces = net.forward()

             
            for i  in range(0, faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence < 0.5:
                    continue
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                start_x, start_y, end_x, end_y = box.astype("int")

                #covnert to square images
                center_x, center_y = (start_x + end_x) // 2 , (start_y + end_y) // 2
                square_length = ((end_x - start_x ) + (end_y - start_y)) // 2 // 2

                square_length *= 1.1

                start_x = int(center_x - square_length)
                start_y = int(center_y - square_length)
                end_x = int(center_x + square_length)
                end_y = int(center_y + square_length)
                
                face = gray[start_y:end_y, start_x:end_x]

                face = ensure_color(face)

                face = cv2.resize(face, image_size)
                face = transform(face).cuda()
                face = torch.unsqueeze(face, dim=0)
                
                output = torch.squeeze(model(face), 0)
                proba = torch.softmax(output, 0)
                
                emo_proba, emo_idx = torch.max(proba, dim=0)
                emo_idx = emo_idx.item()
                emo_proba = emo_proba.item()

                emo_label = FER_2013_EMO_DICT[emo_idx]
                print(emo_label)
                return emo_idx
        except Exception as e:
            print(e)
            return None  

if __name__ == "__main__":
    main()
