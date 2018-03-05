import socket
import struct
import time
from PIL import Image
import numpy as np
import io
import zlib
import cv2
import threading
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '0.0.0.0'
port = 5555
sock.bind((host, port))
sock.listen()

print("Listening:")
(client_sock, client_addr) = sock.accept()
print("Client Info: ", client_sock, client_addr)

img = None
msgs = []

def recv():
    while True:
        try:
            start = time.clock()
            packed_size = client_sock.recv(4)
            size = struct.unpack("i", packed_size)[0] #packet size
            print("total size : %d"%size)
            msg = client_sock.recv(size)
            while len(msg)<size:
                l = len(msg)
                print("partially received : %d"%l)
                msg += client_sock.recv(size-l)
            # print("received all")
            msgs.append(msg)
            print("reception time elapsed : "+str(time.clock() - start))
        except: # if connection is aborted
            break

t = threading.Thread(target=recv, args=(), name='msg_thread')
t.start()

while True:
    if len(msgs)>0:
        start = time.clock()
        decompress = zlib.decompressobj()
        decompressed_data = decompress.decompress(msgs[0])
        decompressed_data += decompress.flush()
        x = len(msgs[0])
        y = len(decompressed_data)
        print("%d %d compress ratio : %.2f"%(x,y,(y-x)/y*100))
        # img_ = Image.open(io.BytesIO(msg))
        # img_ = np.asarray(img_) # slow conversion
        # img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img = cv2.imdecode(np.frombuffer(decompressed_data, np.uint8), 1) # faster reading
        msgs.pop(0)
        end = time.clock()
        print("conversion time elapsed : "+str(end - start))
    if img is not None:
        cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('screen', img.shape[1], img.shape[0])
        cv2.imshow('screen', img)
    k = cv2.waitKey(30) & 0xff #esc
    if k == 27:
        break

cv2.destroyAllWindows()
client_sock.close()

