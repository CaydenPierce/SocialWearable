import socket
import json
import threading
import sys
sys.path.insert(0, "./streaming")

#our custom stuffs
from streaming.Streamer import Streamer

def openConn(port=5000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('caydenpierce.com', port))
    return s

#we should make the streamer inherit from thread so we can kill gracefully. As for now... aggresively ctrl-c, with passion 

def main():
    sock = openConn()
    streamer = Streamer(sys.argv[1], sys.argv[2])
    t1 = threading.Thread(target=streamer.start).start()
    
    while True:
        data = sock.recv(1024)
        data = json.loads(data.decode('utf-8'))
        print(data)

if __name__ == "__main__":
    main()
