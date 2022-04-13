#include "DualG2HighPowerMotorShieldTop.h"

boolean DualG2HighPowerMotorShieldTop::_flipM3 = false;
boolean DualG2HighPowerMotorShieldTop::_flipM4 = false;

// Constructors ////////////////////////////////////////////////////////////////

DualG2HighPowerMotorShieldTop::DualG2HighPowerMotorShieldTop()
{
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
}

DualG2HighPowerMotorShieldTop::DualG2HighPowerMotorShieldTop(unsigned char M3nSLEEP,
                                                       unsigned char M3DIR,
                                                       unsigned char M3PWM,
                                                       unsigned char M3nFAULT,
                                                       unsigned char M3CS,
                                                       unsigned char M4nSLEEP,
                                                       unsigned char M4DIR,
                                                       unsigned char M4PWM,
                                                       unsigned char M4nFAULT,
                                                       unsigned char M4CS)
{
  _M3nSLEEP = M3nSLEEP;
  _M3nFAULT = M3nFAULT;
  _M3DIR = M3DIR;
  _M3PWM = M3PWM;
  _M3CS = M3CS;

  _M4nSLEEP = M4nSLEEP;
  _M4nFAULT = M4nFAULT;
  _M4DIR = M4DIR;
  _M4PWM = M4PWM;
  _M4CS = M4CS;
}

// Public Methods //////////////////////////////////////////////////////////////
void DualG2HighPowerMotorShieldTop::init()
{
  pinMode(_M3nSLEEP, OUTPUT);
  pinMode(_M4nSLEEP, OUTPUT);
  pinMode(_M3PWM, OUTPUT);
  pinMode(_M3nFAULT, INPUT_PULLUP);
  pinMode(_M3CS, INPUT);
  pinMode(_M3DIR, OUTPUT);
  pinMode(_M4DIR, OUTPUT);
  pinMode(_M4PWM, OUTPUT);
  pinMode(_M4nFAULT, INPUT_PULLUP);
  pinMode(_M4CS, INPUT);

#ifdef DUALG2HIGHPOWERMOTORSHIELD_TIMER1_AVAILABLE
  if (_M3PWM == _M3PWM_TIMER1_PIN && _M4PWM == _M4PWM_TIMER1_PIN)
  {
    // Timer 1 configuration
    // prescaler: clockI/O / 1
    // outputs enabled
    // phase-correct PWM
    // top of 400
    //
    // PWM frequency calculation
    // 16MHz / 1 (prescaler) / 2 (phase-correct) / 400 (top) = 20kHz
    TCCR1A = 0b10100000;
    TCCR1B = 0b00010001;
    ICR1 = 400;
  }
#endif
}
// Set speed for motor 1, speed is a number betwenn -400 and 400
void DualG2HighPowerMotorShieldTop::setM3Speed(int speed)
{
  unsigned char reverse = 0;

  if (speed < 0)
  {
    speed = -speed;  // Make speed a positive quantity
    reverse = 1;  // Preserve the direction
  }
  if (speed > 400)  // Max PWM dutycycle
    speed = 400;

#ifdef DUALG2HIGHPOWERMOTORSHIELD_TIMER1_AVAILABLE
  if (_M3PWM == _M3PWM_TIMER1_PIN && _M4PWM == _M4PWM_TIMER1_PIN)
  {
    //OCR1A = speed;
  }
  else
  {
    analogWrite(_M3PWM, speed * 51 / 80); // map 400 to 255
  }
#else
  analogWrite(_M3PWM, speed * 51 / 80); // map 400 to 255
#endif
  analogWrite(_M3PWM, speed * 51 / 80); // map 400 to 255

  if (reverse ^ _flipM3) // flip if speed was negative or _flipM3 setting is active, but not both
  {
    digitalWrite(_M3DIR, HIGH);
  }
  else
  {
    digitalWrite(_M3DIR, LOW);
  }
}

// Set speed for motor 2, speed is a number betwenn -400 and 400
void DualG2HighPowerMotorShieldTop::setM4Speed(int speed)
{
  unsigned char reverse = 0;

  if (speed < 0)
  {
    speed = -speed;  // make speed a positive quantity
    reverse = 1;  // preserve the direction
  }
  if (speed > 400)  // Max
    speed = 400;

#ifdef DUALG2HIGHPOWERMOTORSHIELD_TIMER1_AVAILABLE
  if (_M3PWM == _M3PWM_TIMER1_PIN && _M4PWM == _M4PWM_TIMER1_PIN)
  {
    //OCR1B = speed;
  }
  else
  {
    analogWrite(_M4PWM, speed * 51 / 80); // map 400 to 255
  }
#else
  analogWrite(_M4PWM, speed * 51 / 80); // map 400 to 255
#endif
  analogWrite(_M4PWM, speed * 51 / 80); // map 400 to 255

  if (reverse ^ _flipM4) // flip if speed was negative or _flipM3 setting is active, but not both
  {
    digitalWrite(_M4DIR, HIGH);
  }
  else
  {
    digitalWrite(_M4DIR, LOW);
  }
}

// Set speed for motor 1 and 2
void DualG2HighPowerMotorShieldTop::setSpeeds(int m1Speed, int m2Speed)
{
  setM3Speed(m1Speed);
  setM4Speed(m2Speed);
}

// Return error status for motor 1
unsigned char DualG2HighPowerMotorShieldTop::getM3Fault()
{
  return !digitalRead(_M3nFAULT);
}

// Return error status for motor 2
unsigned char DualG2HighPowerMotorShieldTop::getM4Fault()
{
  return !digitalRead(_M4nFAULT);
}

void DualG2HighPowerMotorShieldTop::flipM3(boolean flip)
{
  DualG2HighPowerMotorShieldTop::_flipM3 = flip;
}

void DualG2HighPowerMotorShieldTop::flipM4(boolean flip)
{
  DualG2HighPowerMotorShieldTop::_flipM4 = flip;
}

// Enables the MOSFET driver for M3.
void DualG2HighPowerMotorShieldTop::enableM3Driver()
{
  digitalWrite(_M3nSLEEP, HIGH);
}

// Enables the MOSFET driver for M4.
void DualG2HighPowerMotorShieldTop::enableM4Driver()
{
  digitalWrite(_M4nSLEEP, HIGH);
}

// Enables the MOSFET drivers for both M3 and M4.
void DualG2HighPowerMotorShieldTop::enableDrivers()
{
  enableM3Driver();
  enableM4Driver();
}

// Puts the MOSFET driver for M3 into sleep mode.
void DualG2HighPowerMotorShieldTop::disableM3Driver()
{
  digitalWrite(_M3nSLEEP, LOW);
}

// Puts the MOSFET driver for M4 into sleep mode.
void DualG2HighPowerMotorShieldTop::disableM4Driver()
{
  digitalWrite(_M4nSLEEP, LOW);
}

// Puts the MOSFET drivers for both M3 and M4 into sleep mode.
void DualG2HighPowerMotorShieldTop::disableDrivers()
{
  disableM3Driver();
  disableM4Driver();
}

unsigned int DualG2HighPowerMotorShieldTop::getM3CurrentReading()
{
  return analogRead(_M3CS);
}

unsigned int DualG2HighPowerMotorShieldTop::getM4CurrentReading()
{
  return analogRead(_M4CS);
}

// Set voltage offset of M3 current reading at 0 speed.
void DualG2HighPowerMotorShieldTop::calibrateM3CurrentOffset()
{
  setM3Speed(0);
  enableM3Driver();
  delay(1);
  DualG2HighPowerMotorShieldTop::_offsetM3 = getM3CurrentReading();
}

// Set voltage offset of M4 current reading at 0 speed.
void DualG2HighPowerMotorShieldTop::calibrateM4CurrentOffset()
{
  setM4Speed(0);
  enableM4Driver();
  delay(1);
  DualG2HighPowerMotorShieldTop::_offsetM4 = getM4CurrentReading();
}

// Get voltage offset of M3 and M4 current readings.
void DualG2HighPowerMotorShieldTop::calibrateCurrentOffsets()
{
  setSpeeds( 0, 0);
  enableDrivers();
  delay(1);
  DualG2HighPowerMotorShieldTop::_offsetM3 = getM3CurrentReading();
  DualG2HighPowerMotorShieldTop::_offsetM4 = getM4CurrentReading();
}


// Return M3 current value in milliamps using the gain value for the specific version.
unsigned int DualG2HighPowerMotorShieldTop::getM3CurrentMilliamps(int gain)
{
  // 5V / 1024 ADC counts / gain mV per A
  // The 24v14, 18v18 and 24v18 results in 244 mA per count.
  // The 18v22 results in 488 mA per count.
  unsigned int mAPerCount = 5000000/1024/gain;
  int reading = (getM3CurrentReading() - DualG2HighPowerMotorShieldTop::_offsetM3);
  if (reading > 0)
  {
    return reading * mAPerCount;
  }
  return 0;
}

// Return M4 current value in milliamps using the gain value for the specific version.
unsigned int DualG2HighPowerMotorShieldTop::getM4CurrentMilliamps(int gain)
{
  // 5V / 1024 ADC counts / gain mV per A
  // The 24v14, 18v18 and 24v18 results in 244 mA per count.
  // The 18v22 results in 488 mA per count.
  unsigned int mAPerCount = 5000000/1024/gain;
  int reading = (getM4CurrentReading() - DualG2HighPowerMotorShieldTop::_offsetM4);
  if (reading > 0)
  {
    return reading * mAPerCount;
  }
  return 0;
}

// Return M3 current value in milliamps for 24v14 version.
unsigned int DualG2HighPowerMotorShieldTop24v14::getM3CurrentMilliamps()
{
  return DualG2HighPowerMotorShieldTop::getM3CurrentMilliamps(20);
}

// Return M4 current value in milliamps for 24v14 version.
unsigned int DualG2HighPowerMotorShieldTop24v14::getM4CurrentMilliamps()
{
  return DualG2HighPowerMotorShieldTop::getM4CurrentMilliamps(20);
}

// Return M3 current value in milliamps for 18v18 version.
unsigned int DualG2HighPowerMotorShieldTop18v18::getM3CurrentMilliamps()
{
  return DualG2HighPowerMotorShieldTop::getM3CurrentMilliamps(20);
}

// Return M4 current value in milliamps for 18v18 version.
unsigned int DualG2HighPowerMotorShieldTop18v18::getM4CurrentMilliamps()
{
  return DualG2HighPowerMotorShieldTop::getM4CurrentMilliamps(20);
}

// Return M3 current value in milliamps for 24v18 version.
unsigned int DualG2HighPowerMotorShieldTop24v18::getM3CurrentMilliamps()
{
  return DualG2HighPowerMotorShieldTop::getM3CurrentMilliamps(20);
}

// Return M4 current value in milliamps for 24v18 version.
unsigned int DualG2HighPowerMotorShieldTop24v18::getM4CurrentMilliamps()
{
  return DualG2HighPowerMotorShieldTop::getM4CurrentMilliamps(20);
}
// Return M3 current value in milliamps for 18v22 version.
unsigned int DualG2HighPowerMotorShieldTop18v22::getM3CurrentMilliamps()
{
  return DualG2HighPowerMotorShieldTop::getM3CurrentMilliamps(10);
}

// Return M4 current value in milliamps for 18v22 version.
unsigned int DualG2HighPowerMotorShieldTop18v22::getM4CurrentMilliamps()
{
  return DualG2HighPowerMotorShieldTop::getM4CurrentMilliamps(10);
}
