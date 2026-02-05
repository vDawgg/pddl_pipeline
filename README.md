# pddl_pipeline

Code repository for master thesis 'Automated PDDL Generation Using Large Language Models for Task Planning in Surgical Robotics'

## Setup

While the dependencies noted below are part of the docker image, the following outlines the local development setup if needed.

For the syntax checks courtesy of [VAL](https://github.com/KCL-Planning/VAL) make sure to install the ```Parser``` binary to your system together with the corresponding ```libval.so``` library.

Additionally, for the FastDownward planning system build version ```24.06.1```, which you can find [here](https://www.fast-downward.org/latest/releases/24.06/), place its source directoy one level outside of this directory and build the release by running ```build.py release```, so it can be referenced in our code by ```../../fast-downward-24.06.1/fast-downward.py```.

## Running the project

Before running the pipelines proposed in this thesis, the inference server hosting the LLM has to be started. For this, we provide docker compose services, which are set up for the models evaluated in this project. All current implementations use quantized versions of the models for faster inference. To start one of the models, run ```docker compose up <model-name>```. For running a model and a pipeline, run ```docker compose --profile <model-name> up```, the config for the evaluation pipeline can be changed by setting the values inside .env.

**Note:** We assume that the [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is set up on your system, to make GPU inference work.

The two main entrypoints to run the project are [main.py](main.py) and [eval.py](eval.py). While main is intended for testing and debugging purposes, eval shows the eval output after a specific number of iterations and does not log debug information to stdout. For both, one of the pipelines defined in [pipeline](./src/pipeline/) has to be specified before starting.
