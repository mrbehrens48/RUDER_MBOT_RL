#pragma once

#if defined(__AVR_ATmega168__) || defined(__AVR_ATmega328P__) || \
    defined(__AVR_ATmega328PB__) || defined (__AVR_ATmega32U4__)
  #define DUALG2HIGHPOWERMOTORSHIELD_TIMER1_AVAILABLE
#endif

#include <Arduino.h>

class DualG2HighPowerMotorShieldTop
{
  public:
    // CONSTRUCTORS
    DualG2HighPowerMotorShieldTop();
    DualG2HighPowerMotorShieldTop(unsigned char M3nSLEEP,
                               unsigned char M3DIR,
                               unsigned char M3PWM,
                               unsigned char M3nFAULT,
                               unsigned char M3CS,
                               unsigned char M4nSLEEP,
                               unsigned char M4DIR,
                               unsigned char M4PWM,
                               unsigned char M4nFAULT,
                               unsigned char M4CS);

    // PUBLIC METHODS
    void init();
    void setM3Speed(int speed); // Set speed for M3.
    void setM4Speed(int speed); // Set speed for M4.
    void setSpeeds(int m1Speed, int m2Speed); // Set speed for both M3 and M4.
    unsigned char getM3Fault(); // Get fault reading from M3.
    unsigned char getM4Fault(); // Get fault reading from M4.
    void flipM3(boolean flip); // Flip the direction of the speed for M3.
    void flipM4(boolean flip); // Flip the direction of the speed for M4.
    void enableM3Driver(); // Enables the MOSFET driver for M3.
    void enableM4Driver(); // Enables the MOSFET driver for M4.
    void enableDrivers(); // Enables the MOSFET drivers for both M3 and M4.
    void disableM3Driver(); // Puts the MOSFET driver for M3 into sleep mode.
    void disableM4Driver(); // Puts the MOSFET driver for M4 into sleep mode.
    void disableDrivers(); // Puts the MOSFET drivers for both M3 and M4 into sleep mode.
    unsigned int getM3CurrentReading();
    unsigned int getM4CurrentReading();
    void calibrateM3CurrentOffset();
    void calibrateM4CurrentOffset();
    void calibrateCurrentOffsets();
    unsigned int getM3CurrentMilliamps(int gain);
    unsigned int getM4CurrentMilliamps(int gain);

  protected:
    unsigned int _offsetM3;
    unsigned int _offsetM4;

  private:
    unsigned char _M3PWM;
    static const unsigned char _M3PWM_TIMER1_PIN = 5;
    unsigned char _M4PWM;
    static const unsigned char _M4PWM_TIMER1_PIN = 11;
    unsigned char _M3nSLEEP;
    unsigned char _M4nSLEEP;
    unsigned char _M3DIR;
    unsigned char _M4DIR;
    unsigned char _M3nFAULT;
    unsigned char _M4nFAULT;
    unsigned char _M3CS;
    unsigned char _M4CS;
    static boolean _flipM3;
    static boolean _flipM4;

};

class DualG2HighPowerMotorShieldTop24v14 : public DualG2HighPowerMotorShieldTop
{
  public:
    using DualG2HighPowerMotorShieldTop::DualG2HighPowerMotorShieldTop;
    unsigned int getM3CurrentMilliamps(); // Get current reading for M3.
    unsigned int getM4CurrentMilliamps(); // Get current reading for M4.
};

class DualG2HighPowerMotorShieldTop18v18 : public DualG2HighPowerMotorShieldTop
{
  public:
    using DualG2HighPowerMotorShieldTop::DualG2HighPowerMotorShieldTop;
    unsigned int getM3CurrentMilliamps(); // Get current reading for M3.
    unsigned int getM4CurrentMilliamps(); // Get current reading for M4.
};

class DualG2HighPowerMotorShieldTop24v18 : public DualG2HighPowerMotorShieldTop
{
  public:
    using DualG2HighPowerMotorShieldTop::DualG2HighPowerMotorShieldTop;
    unsigned int getM3CurrentMilliamps(); // Get current reading for M3.
    unsigned int getM4CurrentMilliamps(); // Get current reading for M4.
};

class DualG2HighPowerMotorShieldTop18v22 : public DualG2HighPowerMotorShieldTop
{
  public:
    using DualG2HighPowerMotorShieldTop::DualG2HighPowerMotorShieldTop;
    unsigned int getM3CurrentMilliamps(); // Get current reading for M3.
    unsigned int getM4CurrentMilliamps(); // Get current reading for M4.
};
