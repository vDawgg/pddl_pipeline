A robot with two arms (left_arm and right_arm) is tasked with transferring a needle from one arm to another through a set of rings in a specified sequence before finally placing the needle on a goal position. Before handing over, picking up or placing the needle, the arm always has to first move to the required position (needle/ring/goal). Note, that when transferring the needle from one arm to the other, it should always be grasped by at least one arm so that it does not get dropped.

The actions available to the robot are:
- move
    - Moves the arm to a specified position (needle or ring)
- pick
    - Closes the arms gripper at the current position
- place
    - Opens the arms gripper at the current position
