FROM vllm/vllm-openai:v0.17.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Configure pip mirrors (Tsinghua + USTC + Huawei)
RUN pip config set global.index-url http://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip config set global.extra-index-url "http://pypi.mirrors.ustc.edu.cn/simple/ http://repo.huaweicloud.com/repository/pypi/simple" && \
    pip config set global.timeout 300 && \
    pip config set install.trusted-host "pypi.tuna.tsinghua.edu.cn pypi.mirrors.ustc.edu.cn repo.huaweicloud.com"

# Install speculators dependencies (code mounted at runtime via hostPath)
COPY . /workspace/speculators
WORKDIR /workspace/speculators
RUN pip install -e ".[datagen]"

# Install uv and psutil (used by gen_and_train.py)
RUN pip install uv psutil

WORKDIR /workspace
