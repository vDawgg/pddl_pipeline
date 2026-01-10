FROM ghcr.io/astral-sh/uv:bookworm-slim AS val

# VAL installation
RUN apt update && apt upgrade -y
RUN apt install cmake make g++ mingw-w64 flex bison git -y
RUN git clone https://github.com/KCL-Planning/VAL
WORKDIR /VAL
RUN chmod +x ./scripts/linux/build_linux64.sh
RUN ./scripts/linux/build_linux64.sh Parser Release
RUN mv build/linux64/Release/bin/Parser /usr/local/bin/
RUN mv build/linux64/Release/bin/libVAL.so /usr/local/lib/
RUN ldconfig
WORKDIR /

FROM val AS fast_downward

# FastDownward installation
RUN apt install curl
RUN curl https://www.fast-downward.org/latest/files/release24.06/fast-downward-24.06.1.tar.gz -o fast_downward.tar.gz
RUN tar -xvf fast_downward.tar.gz
WORKDIR /fast-downward-24.06.1
RUN uv run python ./build.py

FROM fast_downward AS app_base

WORKDIR /app
COPY ./pyproject.toml ./uv.lock ./
RUN uv sync --locked

FROM app_base

COPY ./src ./src
COPY ./prompts ./prompts
COPY ./main.py ./main.py
COPY ./eval.py ./eval.py
COPY ./.default-key ./.default-key

CMD uv run eval.py \
    --iterations ${EVAL_ITERATIONS} \
    --pipeline ${EVAL_PIPELINE} \
    --model ${EVAL_MODEL} \
    --domain ${EVAL_DOMAIN}
