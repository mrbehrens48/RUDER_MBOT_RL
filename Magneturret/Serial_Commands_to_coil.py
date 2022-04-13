# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:12:03 2021
useful for direct contol of the magneturret, for debugging etc. 
@author: Windows
"""


import serial
import time
from math import pi

try:
    print('trying to connect to arduino')
    #self.ser = serial.Serial('COM8', 9600, timeout=1)
    ser = serial.Serial('COM5', 9600, timeout=1)
    print('successfully connected to arduino')
except:

    print('unable to open com port with arduino')
    input()
    
    
  
def arduino_communication(M_x, M_y, M_z, freq, phi_x, phi_y):

    action_string = f"Mydata={round(M_x,2)},{round(M_y,2)},{round(M_z,2)},{round(freq,2)},{round(phi_x,2)},{round(phi_y,2)}"
    print(action_string)
    action_string += "\n"
    action_string_bytes = action_string.encode('utf-8')
    ser.write(action_string_bytes)
    #time.sleep(.1)
    #print(i)
    Arduino_response = ser.readline()
    Arduino_response = Arduino_response.decode('utf-8')
    print(Arduino_response)
    
    
i = 0
while True:
    for i in range(100):
        arduino_communication(1,1,1,10,0,0)    #order: (Mx, My, Mz, freq, phiX, phiY)
        time.sleep(1)
        print(i)
    
    