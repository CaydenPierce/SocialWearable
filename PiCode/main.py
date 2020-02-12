#we should make the streamer inherit from thread so we can kill gracefully. As for now... aggresively ctrl-c, with passion 
import socket
import json
import threading
import sys
import os
import pygame
import time
import subprocess

#config
camPort = 1234
infoPort = 5000
serverName = 'caydenpierce.com'

def openConn(port=5000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('caydenpierce.com', port))
    return s

def clean(data): #this seems to me to be a bad thing. I guess the alternative is starting and ending code like <L? or whatever the kids use
    data = data.decode('utf-8')
    return data[:data.index("}")+1]

def updateUser(data):
    print("Stress detected.")
    pygame.mixer.music.play()

def setupAudio():
    pygame.mixer.init()
    pygame.mixer.music.load("./audio/stress.mp3")


def main():
    t1 = threading.Thread(target=subprocess.run("./stream.sh")).start()
    #subprocess.Popen("./stream.sh", shell=False)
    print("stream started")
    setupAudio()
    print("audio setup")
    time.sleep(20)
    sock = openConn(infoPort)
    print("port opened")
    #streamer = subprocess.run("./stream.sh")

    try:
        while True:
            data = sock.recv(1024)
            if len(data) < 1: #skip if empty
                continue
            data = clean(data)
            data = json.loads(data)
            print(data)
            if data['test'] == 1:
                pass
            else:
                updateUser(data)
    except KeyboardInterrupt as e:
        sock.close()
        #streamer.stop()
        print("Sock closed. Goodbye.")

if __name__ == "__main__":
    main()
