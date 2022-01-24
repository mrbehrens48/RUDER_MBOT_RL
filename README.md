# Soft Actor Critic for Magnetic Microrobot Control


Welcome to the official repository for _Smart Magnetic Microrobots Learn to Swim with Deep Reinforcement Learning._ This repository contains all the code used to implement a feedback control system for a helical agar magnetic microrobot and custom three-axis electromagnet.  The full paper describing the implementation details is available on arxiv [here](https://arxiv.org/abs/2201.05599). This code was developed in Windows 10 on a Lambda Labs workstation with dual NVIDIA RTX1080 GPUs. This code is provided as is with the hope that it might be useful or illuminating, but without warranty or expectation of ongoing support.

<img align="left" src="readme/Figure 1.jpg" height=480px>


Included Files:
1) microrobot_SAC_control.py (the main program file)
2) microrobot_environment.py (interface between main program and hardware)
3) Magneturret.ino (arduino C++ code, run on an Arduino STEMtera)
4) Arduino Libraries for stacking two Pololu motor controller shields together.

Resources: <b> [PDF](https://arxiv.org/abs/2201.05599) 

Authors: [Michael Behrens](https://www.linkedin.com/in/michael-behrens-phd/), [Warren Ruder](https://www.warrenruder.com/)

### Contact:

If you have any questions please contact me:  
Michael Behrens: mrb157[at]pitt[dot]edu
