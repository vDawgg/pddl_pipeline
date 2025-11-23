## Task Context

You are an expert at generating PDDL domain and problem files. You will be asked to generated PDDL files for a problem given in natural language. You will first be asked to generate the domain file for the given task.

Make sure to adhere to the naming of objects and actions given in the prompt.

Answer only with PDDL as output.

## Task description

Your task is to generate the PDDL domain for the following scenario:

A robot arm is tasked to re-assemble stackable blocks on a table with unlimited space. The robot arm can be used for stacking a block onto a block, unstacking a block from a block, putting down a block, or picking up a block.

The actions available to the robot are:
- pick-up(x)
- put-down(x)
- stack(x, y)
- unstack(x, y)
