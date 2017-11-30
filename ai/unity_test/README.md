# This repository contains RL learning examples

using Unity and https://github.com/Unity-Technologies/ml-agents.

In the following sections, I will describe the environments, their state and action spaces and the reward settings.

# balance_broom
The agent controls a plane, on which an inverted broom stands. The agent moves the plane so that the broom doesn't fall (more precisely, so that its inclining angle is less than 30 degrees.

* state size : 8
  1. The position of the broom with respect to the center of the plane (2 float numbers)
  2. The velocity of the broom (2 float numbers)
  3. The angle between the broom and the normal vector of the plane, the y axis (2 float numbers)
  4. The angular velocity of the broom (2 float numbers)
