A robot with two arms (left_arm and right_arm) is tasked with transferring a needle through a set of rings in a specified sequence to a goal position. To pass a needle through a ring, the arms need to hand over the needle through the ring. Only then is a ring considered to be passed.

The actions available to the robot are:
- move
    - Moves the arm to a specified position (needle or ring)
- pick
    - Closes the arms gripper at the current position
- place
    - Opens the arms gripper at the current position
