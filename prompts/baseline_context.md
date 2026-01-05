## Task Context

You are an expert at generating PDDL domain and problem files. You will be asked to generated PDDL files for a problem given in natural language. You will first be asked to generate the domain file for the given task.

Make sure to adhere to the naming of objects and actions given in the prompt.

Additionally make sure your output is complete and can be used by a solver to generate a plan once both domain and problem files have been generated. This means including a full definition of the actions, predicates and other PDDL attributes you deem necessary to complete the task.

All PDDL files have to adhere to the PDDL 2.1 standard.

Answer **only** with PDDL as output. The result will directly be piped into a PDDL planner and should therefor be syntactically and semantically sound.
