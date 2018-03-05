import socket
import mss
import struct
from PIL import Image
import numpy as np
from ctypes import windll
import io
import zlib
import threading
import time
user32 = windll.user32
user32.SetProcessDPIAware()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #TCP
host = '127.0.0.1'
port = 5555
sock.connect((host, port))
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

mon = {'top': 0, 'left': 0, 'width': 800, 'height': 2000}
sct = mss.mss()

screenshots = [] # don't need to sync

def screenshot():
    while True:
        try:
            start = time.clock()
            screenshots.append(sct.grab(mon))
            print("capture time elapsed : "+str(time.clock() - start))
        except: # when the connection is aborted
            break
    
threading.Thread(target=screenshot, args=(), name='screenshot_thread').start()

while True:
    try:
        if len(screenshots)>0:
            start = time.clock()
            imgByteArr = io.BytesIO()
            img = Image.frombytes('RGB', screenshots[0].size, screenshots[0].rgb)
            img.save(imgByteArr, format='JPEG', quality=10)
            screenshots.pop(0)
            imgByteArr = imgByteArr.getvalue()
            print("conversion time elapsed : "+str(time.clock() - start))
            start = time.clock()
            compress = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, 15)
            compressed_data = compress.compress(imgByteArr)  
            compressed_data += compress.flush()
            packed_size = struct.pack("i", len(compressed_data))
            sock.sendall(packed_size+compressed_data)
            print("transmission time elapsed : "+str(time.clock() - start))
    except:
        break
        
del screenshots # to stop the thread that captures the screen
        