# This repository contains RL learning examples

using Unity and https://github.com/Unity-Technologies/ml-agents.

In the following sections, I will describe the environments, their state and action spaces and the reward and done settings.

# balance_broom
The agent controls a plane, on which an inverted broom stands. The agent tries to move the plane so that the broom doesn't fall (more precisely, so that its angle of inclination is less than 30 degrees.

* state size : 8
  1.  The position of the broom with respect to the center of the plane (2 float numbers)
  2.  The velocity of the broom (2 float numbers)
  3.  The angle between the broom and the normal vector of the plane, the y axis (2 float numbers)
  4.  The angular velocity of the broom (2 float numbers)
* action size : 2, continuous
  1.  up or down
  2.  left or right
* rewards and *done* :
  * Every step : 1 + 1/(|angle_z|+1) + 1/(|angle_x|+1), to minimize the angle of inclination
  * When |angle_z|>30 or |angle_x|>30, set the reward to -20 and *done*.
  
# robot arm (3d)
The agent controls a robot arm consisting of two joints, fixed at origin. Yellow cubes appears at a random position and disappears after a while. The agent tries to move the arm and grab the cube.

* state size : 10
  1.  A integer (0 or 1) indicating whether the arm has reached the cube (1 int number)
  2.  The position of the cube and the upper arm (6 float numbers)
  3.  The relative position between the hand and the cube (3 float numbers)
* action size : 4, continuous
  1.  Turn the joints (4 float numbers, 2 for each joint)
* rewards and *done* :
  * Every step : - distance between the hand and the cube, to encourage approach to the cube.
  * When distance < 2, set reward to 50.
  * *done* when the time exceeds 5000.
  
# autocar
The agent controls a car on a (partial or full) track. The agent tries to move the car to the goal, or to complete the whole track without falling. The agent is equipped with a camera in front of the car to tell the information of the track.

* state size : 16x40 + 1
  1.  The camera has resolution 16(height)x40(width). It sees part of the sky and the track.
  2.  The forward velocity of the car (defined by the velocity dot product the normalized forward vector of the car)
* action size : 2, continuous
  1.  up or down
  2.  left or right
* rewards and *done* :
  * Every step : The norm of the forward velocity/5, to encourage forward movement. If it moves backwards, multiply the (negative) reward by 4.
  * When the car falls off the track, set reward to -50 and *done*.
  * When the car reaches the goal (if there is any), set reward to 100 and *done*.
  
 Link to youtube video showing fully learnt car movements :
 [autocar](https://youtu.be/pHsxddQF0Tc)
 
 # animals
 For the moment, I only modelled a bipedal robot who has four joints, two on each foot. The agent tries to move forward without falling.
 
 * state size : 12
   1.  The angles of the joints and the body (5 float numbers)
   2.  The y position of the body (1 float number)
   3.  The forward velocities of the joints (4 float numbers)
   4.  The upward velocities of the lower joints of each foot (2 float numbers)
* action size : 6, continuous
   1.  Turn left or right for the four joints (4 float numbers), the maximum angle is +-80 degrees
   2.  If the foot is on the ground and the angle is less than 60 degrees, it can jump (apply a vertical force to that foot) (2 float numbers).
* rewards and *done* :
   * Every step : The amount of forward movement * 10 - the angle of the body / 100, to encourage forward movement and to avoid unstable body.
   * When the body is too low, consider it gets stuck, set reward to -10 and *done*. This also avoids the rolling behaviour observed in training.
   
Link to youtube video showing a possible strategy for the robot :
[bipedal robot](https://youtu.be/iETQGdEFVxI)