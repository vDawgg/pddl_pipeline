# pddl_pipeline

Code repository for master thesis 'Automated PDDL Generation Using Large Language Models for Task Planning in Surgical Robotics'

## Local Setup

While the dependencies noted below are part of the docker image, the following outlines the local development setup if needed.

For the syntax checks courtesy of [VAL](https://github.com/vDawgg/VAL) make sure to install the ```Parser``` binary to your system together with the corresponding ```libval.so``` library.

Additionally, for the FastDownward planning system build version ```24.06.1```, which you can find [here](https://www.fast-downward.org/latest/releases/24.06/), place its source directoy one level outside of this directory and build the release by running ```build.py release```, so it can be referenced in our code by ```../../fast-downward-24.06.1/fast-downward.py```.

## Running the project

### Local

Before running the project locally, you either have to add a api-key, which has to be copied to ```.openrouter-key``` or ```.openai-key``` depending on the model you want to use, or first start one of the models for local inference as outlined below.

The two main entrypoints to run the project are [main.py](main.py) and [eval.py](eval.py). While main is intended for testing and debugging purposes, eval shows the eval output after a specific number of iterations and does not log debug information to stdout. For both, one of the pipelines defined in [pipeline](./src/pipeline/) has to be specified before starting. The remaining options can be shown via ```uv run main.py --help```

#### Example Run

The following command runs the tool_call pipeline with gpt-oss-120b on the ring_and_peg_1 problem over 30 iterations.

```bash
uv run eval.py --model gpt_oss_120b --pipeline tool_call --domain ring_and_peg --problem ring_and_peg_1 --iterations 30
```

The results for each run are then written as a csv-file to the [results](./results/) directory. Likewise the generated plans (if any), logs and pddl files are written to the directories of their matching names.

### Docker

For inference on local hardware, we provide a set of [llama-cpp-server]() configurations which can be run using docker-compose. To start up the server running e.g. gemma-4-4b, run ```docker compose --profile gemma-4 up -d``` and wait until the service is healthy.

The pipeline can then be started by runninng the command below. Note that the container always runs the eval script for the pipeline and accepts the same arguments as the script when its run locally.

```bash
docker compose run --build --rm pipeline --model gemma_4 --domain ring_and_peg --problem ring_and_peg_1 --pipeline tool_call --iterations 30
```

Running the pipeline also automatically mounts the results, plans, logs and pddl directories so the artifacts from each run can be referenced later.

**Note:** We assume that the [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is set up on your system, to make GPU inference work.
