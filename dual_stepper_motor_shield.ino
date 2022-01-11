//add required libraries
#include <AccelStepper.h>
#include <Timer.h>

//initialize constructs

Timer t1;
AccelStepper stepper1(1, 8, 9);
AccelStepper stepper2(1,10,11);
float sensor1 = 0;
int printTiming = 200;
int tPos = 0;
int d2g = 0;
int cp = 0;

//initialize global variables
int stepperSpeed = 50;

//from calibration offset
//these can be different for every sensor
float sensor1Offset = 0.099;

//initial setup
void setup()
{
  stepper1.setMaxSpeed(20000);
  stepper1.setCurrentPosition(0);
  stepper1.setSpeed(6000);
  stepper2.setMaxSpeed(6000);
  stepper2.setSpeed(6000);
  Serial.begin(9600);
  while (!Serial) {
    ;
  }
  t1.every(printTiming, printReading,0);
}

//main loop
void loop()
{
  t1.update(); 

  stepper2.setSpeed(1000);
  stepper2.runSpeed();
  stepper1.setSpeed(20000);
  stepper1.runSpeedToPosition();
  takeReading();
  moveMagnet();
}

void takeReading()
{
  sensor1 = analogRead(4)*5.0/1024.0;
}

void printReading()
{
  d2g = stepper1.distanceToGo();
  cp = stepper1.currentPosition();
  String Output = String(String(sensor1,3) + "," + String(tPos) + "," + String(d2g) + "," + String(cp));
  Serial.println(Output);
}

void moveMagnet()
{
  if(sensor1 > 1)
  {
    tPos = -20000;
    stepper1.moveTo(tPos);
  }
  else
  {
    tPos = 0;
    stepper1.moveTo(tPos);
  }
  
}




