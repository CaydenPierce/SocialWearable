import argparse

import cv2
import zmq

from camera.Camera import Camera
from constants import PORT, SERVER_ADDRESS
from utils import image_to_string

import time


class Streamer:

    def __init__(self, server_address=SERVER_ADDRESS, port=PORT):
        """
        Tries to connect to the StreamViewer with supplied server_address and creates a socket for future use.

        :param server_address: Address of the computer on which the StreamViewer is running, default is `localhost`
        :param port: Port which will be used for sending the stream
        """

        print("Connecting to ", server_address, "at", port)
        self.context = zmq.Context()
        self.footage_socket = self.context.socket(zmq.PUB)
        self.footage_socket.setsockopt(zmq.LINGER, 0)
        self.footage_socket.setsockopt( zmq.RCVTIMEO, 10000 )
        self.footage_socket.connect('tcp://' + server_address + ':' + port)
        self.keep_running = True
        self.frameNum = 0

    def start(self):
        """
        Starts sending the stream to the Viewer.
        Creates a camera, takes a image frame converts the frame to string and sends the string across the network
        :return: None
        """
        print("Streaming Started...")
        camera = Camera(height=640, width=480)
        camera.start_capture(fps=3)
        self.keep_running = True

        old_frame_string = None

        while self.footage_socket and self.keep_running:
            try:
                fpsTime = time.time()
                currTime = time.time()
                frame = camera.current_frame.read()  # grab the current frame
                self.frameNum += 1
                currTime = time.time()
                currTime = time.time()
                image_as_string = image_to_string(frame)
                while image_as_string == old_frame_string:
                    image_as_string = image_to_string(camera.current_frame.read())
                currTime = time.time()
                self.footage_socket.send(image_as_string)
                old_frame_string = image_as_string
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break
        print("Streaming Stopped!")
        cv2.destroyAllWindows()
        return

    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False
        #self.footage_socket.close()
        #self.context.term()


def main():
    port = PORT
    server_address = SERVER_ADDRESS

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server',
                        help='IP Address of the server which you want to connect to, default'
                             ' is ' + SERVER_ADDRESS,
                        required=True)
    parser.add_argument('-p', '--port',
                        help='The port which you want the Streaming Server to use, default'
                             ' is ' + PORT, required=False)

    args = parser.parse_args()

    if args.port:
        port = args.port
    if args.server:
        server_address = args.server

    streamer = Streamer(server_address, port)
    streamer.start()


if __name__ == '__main__':
    main()
