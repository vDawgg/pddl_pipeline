# pddl_pipeline

Code repository for master thesis 'Automated PDDL Generation Using Large Language Models for Task Planning in Surgical Robotics'

## Setup

While the dependencies noted below are part of the docker image, the following outlines the local development setup if needed.

For the syntax checks courtesy of [VAL](https://github.com/KCL-Planning/VAL) make sure to install the ```Parser``` binary to your system together with the corresponding ```libval.so``` library.

Additionally, for the FastDownward planning system build the ... version, which you can find here and place its source directoy one level outside of this, so it can be referenced in our code by ```../../fast-downward-24.06.1/fast-downward.py```.
