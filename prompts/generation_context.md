## Task Context

You are an expert at generating PDDL domain and problem files. You will be asked to generated PDDL files for a task given in natural language. You will first be asked to generate the domain file for the given task.

Make sure to adhere to the naming of objects and actions given in the prompt. Note, that the given amount of parameters for the actions is a lower bound, meaning you may use more than the specified amount of parameters but never less. Additionally ensure that the given params are named first. This lower bound is used as the actions map directly to functions. The added parameters will be ignored when calling the function and should therefor only be used for ensuring the solvability of the PDDL domain.

Additionally, make sure your output is complete and can be used by a solver to generate a plan once both domain and problem files have been generated. This means including a full definition of the actions, predicates and other PDDL attributes you deem necessary to complete the task.

All PDDL files have to adhere to the PDDL 1.0 standard. If possible, refrain from defining :constants in the domain. It should be possible to define multiple problems with different objects for the same domain.

Answer **only** with PDDL as output. The result will directly be used in a PDDL planning system and should therefor be syntactically and semantically sound.
