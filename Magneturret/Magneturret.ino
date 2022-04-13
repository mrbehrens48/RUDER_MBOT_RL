
#include <DualG2HighPowerMotorShield.h>
#include "DualG2HighPowerMotorShieldTop.h"

#include <Timer.h>

DualG2HighPowerMotorShieldTop24v14 md_top; //define the top motor driver. THis on has a single coil connected in the first position. 
DualG2HighPowerMotorShield24v14 md; //define the bottom motor driver. This one has two of the coils connected

Timer t1,t2;

char rawData[100] = "";
char keyword[] = "Mydata=";
char theText[20];
float M_x = 0.0;
float M_y = 0.0;
float M_z = 0.0;
float freq = 0.0;
float phi_x = 0.0;
float phi_y = 0.0;
int actions = 0;

//set which analog inputs the joystick is connected to
int temperaturePin = A5; //horizontal input from joystick

float cmdX = 0; //joystick x axis value
float cmdY = 0; //joystick y axis value

float xEnv = 0; //the envelop function for the x output
float yEnv = 0; //the envelop function for the y output
float zEnv = 0;

float PWMx = 0; //x coil PWM output
float PWMy = 0; //y coil PWM output
float PWMz = 0; //up coil PWM output

float t = 0; //time axis for periodic functions
float pi = 3.141;
int printTiming = 50;
int temperatureTiming = 1000; //10 seconds
int resetTiming = 6000; //60 seconds
int timeInterval = 1;
int squareBit = 0;
int lastSquareT = 0;
int squareT = 0;
int squareRead = 0;
float wave;
int update_speed = 0;

void stopIfFault()
{
  if (md.getM1Fault())
  {
    md.disableDrivers();
    delay(1);
    Serial.println("M1 fault");
    while (1);
  }
  if (md.getM2Fault())
  {
    md.disableDrivers();
    delay(1);
    Serial.println("M2 fault");
    while (1);
  }
  if (md_top.getM3Fault())
  {
    md_top.disableDrivers();
    delay(1);
    Serial.println("M3 fault");
    //while (1);
  }
  if (md_top.getM4Fault())
  {
    md_top.disableDrivers();
    delay(1);
    Serial.println("M4 fault");
    //while (1);
  }
}

void setup() {
  
  Serial.begin(9600);
  while (!Serial) {
    ;
  }
  
  Serial.println("Dual G2 High Power Motor Shield Top and Bottom");
  md_top.init();
  md_top.calibrateCurrentOffsets();
  md.init();
  md.calibrateCurrentOffsets();

  
  md.enableDrivers();
  md_top.enableDrivers();
  delay(10);

  // Uncomment to flip a motor's direction:
  //md.flipM1(true);
  //md.flipM2(true);
  //md_top.flipM3(true);


  //t1.every(temperatureTiming, testTemperature,0);
  //t1.every(timeInterval, incrementTime,0);
  
  
}

void(*ResetFunc)(void) = 0;

//Main loop
void loop() {
  
  
  if (Serial.available() > 0)
  {
    actions ++;
    getCommand(); //recieve a command from python and send it to the motors
  }
  
  //String Output = String(String(M_x) + "," + String(M_y) + "," + String(M_z) + "," + String(freq) + "," + String(phi_x) + "," + String(phi_y));
  //Serial.println(Output);
   
  t1.update(); //update the timer
  t2.update(); //update the timer
  
  t = millis();
  PWMx = M_x*sin(freq*t/1000+phi_x)*400;
  md_top.setM4Speed(PWMx);
  
  t = millis();
  PWMy = M_y*sin(freq*t/1000+phi_y)*400;
  md.setM1Speed(PWMy);

  t = millis();
  PWMz = M_z*sin(freq*t/1000)*400;
  md.setM2Speed(PWMz);
  
  //stopIfFault();
}

void printReading()
{
  //String Output = String(String(cmdX) + "," + String(cmdY));
  String Output = String(String(PWMx) + "," + String(PWMy) + "," + String(PWMz));
  Serial.println(Output);
  //Serial.print("M3 current: ");
  //Serial.println(md_top.getM3CurrentMilliamps());
}

void getCommand()
{
  if (Serial.available() > 0)
  { //new data in
    size_t byteCount = Serial.readBytesUntil('\n', rawData, sizeof(rawData) - 1); //read in data to buffer
    rawData[byteCount] = NULL; //put an end character on the data
    //Serial.print("Raw Data = ");
    //Serial.println(rawData);

    //now find keyword and parse
    char *keywordPointer = strstr(rawData, keyword);
    if (keywordPointer != NULL)
    {
      int dataPosition = (keywordPointer - rawData) + strlen(keyword);
      const char delimiter[] = ",";
      char parsedStrings[6][50];
      int dataCount = 0;
      //Serial.print("data position = ");
      //Serial.println(dataPosition);

      char *token = strtok(&rawData[dataPosition], delimiter); //look for the first piece of data after keyword until comma
      if (token != NULL && strlen(token) < sizeof(parsedStrings[0]))
      {
        strncpy(parsedStrings[0], token, sizeof(parsedStrings[0]));
        dataCount++;
      }
      else
      {
        Serial.println("token too big");
        strcpy(parsedStrings[0], NULL);
      }

      for (int i = 1; i < 6; i++) 
      {
        token = strtok(NULL, delimiter);
        if (token != NULL && strlen(token) < sizeof(parsedStrings[i]))
        {
          strncpy(parsedStrings[i], token, sizeof(parsedStrings[i]));
          dataCount++;
        }
        else
        {
          Serial.println("token 2too big");
          strcpy(parsedStrings[i], NULL);
        }
      }

      //convert to variables that we can actually use
      

      if (dataCount == 6) {
        strncpy(theText, parsedStrings[0], sizeof(theText));
        M_x = atof(parsedStrings[0]);
        M_y = atof(parsedStrings[1]);
        M_z = atof(parsedStrings[2]);
        freq = atof(parsedStrings[3]);
        phi_x = atof(parsedStrings[4]);
        phi_y = atof(parsedStrings[5]);
        testTemperature(); //if we got all the data correctly, send the current temperature reading. This will help monitor for errors
        //Serial.println("Data Received");

        if(M_x == 1 && M_y == 2 && M_z == 3 && freq == 4 && phi_x == 5 && phi_y == 6)
        {
          Serial.println("Arduino Resetting");
          setup();
        }
      }
      else
      {
        Serial.println("Arduino fault: data no good");
      }
    }
    else
    {
      Serial.println("Arduino fault: sorry no keyword");
    }
  }
}



void incrementTime()
{
  t = millis();
}

void testTemperature()
{
  int temp = analogRead(temperaturePin);
  Serial.print("temperature is: ");
  Serial.println(String(temp));
  if( temp > 400)
  {
    md_top.disableDrivers();
    md.disableDrivers();
    while (true)
    {
      Serial.println(String(temp));
      Serial.println("TEMPERATURE TOO HIGH. SHUTTING DOWN");
    }
  }
}
