import socket
import time
import json

class WearConn:
    def __init__(self, port=5000):
        self.status = False #status represents connection status
        self.port = port
        self.sock, self.conn, self.wearaddr = self.connect()

    def send(self, data: dict):
        data = json.dumps(data).encode('utf-8')
        self.conn.send(data)

    def connect(self):
        # create an INET, STREAMing socket
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # bind the socket to a public host, and a well-known port
        serversocket.bind(('0.0.0.0', self.port))
        # become a server socket, let only one connection at once
        print("Waiting for connection from client...")
        print(serversocket.getsockname())
        serversocket.listen()
        connection, client_address = serversocket.accept()
        print("Connected to client {}".format(client_address))
        return serversocket, connection, client_address

    def end(self):
        self.sock.close()

if __name__ == "__main__":
    pass
