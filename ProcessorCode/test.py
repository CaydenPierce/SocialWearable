#receive video from standard in and process

import cv2

def main():
    try:
        cap = cv2.VideoCapture('/dev/stdin')
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
    except KeyboardInterrupt as e:
        print(e)
        cap.release()

if __name__ == "__main__":
    main()
