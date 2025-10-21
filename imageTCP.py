import time
import socket
import threading

messageArray = ["hi", "hello", "world"]

HOST = "127.0.0.1" # The server's hostname or IP address
PORT = 65432 # The port used by the server

def send_messages(sock):
    for msg in messageArray:
        sendMessage = msg.strip()
        sock.sendall(sendMessage.encode() + b"\n")

def receive_messages(sock):
    while True:
        data = sock.recv(1024)
        if not data:
            break
        print(f"Received {data!r}")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    recv_thread = threading.Thread(target=receive_messages, args=(s,), daemon=True)
    recv_thread.start()
    send_messages(s)

while recv_thread.is_alive(): # Keep main thread alive as long as receiver is active
    time.sleep(0.1)