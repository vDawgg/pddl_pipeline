## Task description

Your task is to generate the PDDL problem given the following task description and the previously generated PDDL domain.

There are 5 colored pegs with the following colors: red, green, blue, pink, yellow. The pegs are all named in the format <color>_peg. Additionally there are 3 colored rings: red, green and blue. Similarly to the pegs, the rings are named <color>_ring.

The robot arm starts out in a default position, while the rings start in the following positions:
- red_ring - pink_peg
- green_ring - yellow_peg
- blue_ring - red_peg

The goal is to transfer all rings to the pegs of their color.
