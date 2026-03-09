## Task description

The actions available to the robot are:
- move(to)
    - Moves the robot to a specified position. Note, that the robot **always** has to move to a specific position before being able to execute another action.
- pick()
    - Closes the robot arms gripper at the current position
- place()
    - Opens the robot arms gripper at the current position

The goal is for the robot to place all rings on the pegs of the same color. Note the naming of the rings and pegs: <color>_ring and <color>_peg. The only possible locations for the rings are the pegs.
