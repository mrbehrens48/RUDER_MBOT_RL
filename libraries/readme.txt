Both these libraries need to be installed in your arduino IDE working directory when you compile Magneturret.ino. 

Pololu Dual G2 High-Power Motor Driver 24v14 Shield for Arduino

These are designed so that two motor drivers can be stacked on top of each other and used with the same arduino. The pins on the top 
driver have to be remapped be using jumper wires, so that each motor driver is driven with unique pins. The mapping is:
On the bottom motor driver, standard pinout:
  _M1nSLEEP = 2;
  _M1nFAULT = 6;
  _M1DIR = 7;
  _M1PWM = 9;
  _M1CS = A0;

  _M2nSLEEP = 4;
  _M2nFAULT = 12;
  _M2DIR = 8;
  _M2PWM = 10;
  _M2CS = A1;

on the top motor driver, the new pin mapping is:

  _M3nSLEEP = 2;
  _M3nFAULT = 0;
  _M3DIR = 1;
  _M3PWM = 5;
  _M3CS = A2;

  _M4nSLEEP = 4;
  _M4nFAULT = 13;
  _M4DIR = 3;
  _M4PWM = 11;
  _M4CS = A3;

The drivers have to share some functions, because there aren't enough pins for perfectly orthogonal mapping. for example you can see that M1nSLEEP and M3nSLEEP share the same pin 2.
The remapped pins on the top motor driver are: 
6 -> 0
7 -> 1
9 -> 5
A0 -> A2
12 -> 13
8 -> 3
10 -> 11
A1 -> A3

Pololu planned for this remapping, and the PCB is designed for this. For complete instructions, see https://www.pololu.com/docs/0J72. This should be hard wired.

Other motor driver instructions:
The motor drivers and the arduino should not share a power supply. The ARDVIN pin should be clipped for safety, and there should be no jumper as is described here: https://www.pololu.com/docs/0J72/4.a
Wire the top and bottom motor driver to the same 24v power supply. 
