## Task description

Your task is to generate a PDDL domain and problem file for the following domain:

A robot arm is tasked to sort a set of colored rings on a set of pegs of the same colors. The arm can be used to pick up a ring from a peg, move a ring to another peg and place the ring on the other peg. Rings are already placed on pegs and can only be placed on pegs. To pick up a ring at a peg, the robot first has to move to the pegs position. The task should be carried out in the real world.

The actions available to the robot are:
- move(to)
    - Moves the robot to a specified position
- pick()
    - Closes the robot arms gripper at the current position
- place()
    - Opens the robot arms gripper at the current position

The problem should describe the following scenario in the domain described above:

There are 5 colored pegs with the following colors: red, green, blue, pink, yellow. The pegs are all named in the format <color>_peg. Additionally there are 3 colored rings: red, green and blue. Similarly to the pegs, the rings are named <color>_ring.

The robot arm starts out in a its resting position (resting_positions), while the rings start on the following pegs:
- red_ring - pink_peg
- green_ring - yellow_peg
- blue_ring - red_peg

The goal is to transfer all rings to the pegs of their color.
