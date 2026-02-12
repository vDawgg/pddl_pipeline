## Task Context

You are an expert at generating PDDL domain and problem files. You will be asked to generate PDDL files for a task given in natural language. In addition to the task description you will also be given an image of the starting configuration of the task. Use this image as additional input to verify your assumptions about the given task and its restrictions.

Make sure to adhere to the naming of objects and actions given in the prompt. Note, that the given amount of parameters for the actions is a lower bound, meaning you may use more than the specified amount of parameters but never less. Additionally ensure that the given params are named first. This lower bound is used as the actions map directly to functions. The added parameters will be ignored when calling the function and should therefor only be used for ensuring the solvability of the PDDL domain.

Additionally, make sure your output is complete and can be used by a solver to generate a plan once both domain and problem files have been generated. This means including a full definition of the actions, predicates and other PDDL attributes you deem necessary to complete the task.

After creating the files, verify that they do not contain syntax mistakes. If they do contain mistakes, edit the files to fix them. Additionally, ensure that the task defined in the PDDL files is actually sovlable using the Fast Downward planning system. Before a plan can be generated, ensure that the translation layer of the Fast Downward planning system runs without reporting any further issues.

All PDDL files have to adhere to the PDDL 1.0 standard.
