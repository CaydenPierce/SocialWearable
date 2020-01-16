#we should make the streamer inherit from thread so we can kill gracefully. As for now... aggresively ctrl-c, with passion 
import socket
import json
import threading
import sys
import os
import pygame

sys.path.insert(0, "./streaming")

#our custom stuffs
from streaming.Streamer import Streamer

#config
camPort = 3000
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
    if data['stress'] > 0.7:
        print("Stress detected.")
        #os.system('mpg123 ./audio/stress.mp3 &')
        pygame.mixer.music.play()

def setupAudio():
    pygame.mixer.init()
    pygame.mixer.music.load("./audio/stress.mp3")
    #pygame.mixer.music.play()


def main():
    setupAudio()
    sock = openConn(infoPort)
    streamer = Streamer(serverName, str(camPort))
    t1 = threading.Thread(target=streamer.start).start()

    try:
        while True:
            data = sock.recv(1024)
            if len(data) < 1: #skip if empty
                continue
            data = clean(data)
            data = json.loads(data)
            print(data)
            updateUser(data)
    except KeyboardInterrupt as e:
        sock.close()
        streamer.stop()
        print("Sock closed. Goodbye.")

if __name__ == "__main__":
    main()
