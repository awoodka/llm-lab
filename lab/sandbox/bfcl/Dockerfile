# BFCL with its own handler and checkers. The package is installed without its dependencies: the full set
# pulls torch and every vendor SDK, and the five categories run here need only these.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
RUN pip install --no-cache-dir --no-deps \
      "bfcl-eval @ https://files.pythonhosted.org/packages/ba/41/ed458527c770c50225b60bae3b0c3444b26804ee455fa2d8f187018d2cb2/bfcl_eval-2026.3.23-py3-none-any.whl#sha256=3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a" \
 && pip install --no-cache-dir openai==3.14.1 overrides==7.7.0 tenacity==9.1.4 filelock==3.32.7 mpmath==1.3.0
COPY common/boxproto.py /box/boxproto.py
COPY bfcl/entry.py /box/entry.py
ENV BFCL_PROJECT_ROOT=/tmp/bfcl OPENAI_API_KEY=unused PYTHONDONTWRITEBYTECODE=1
WORKDIR /box
ENTRYPOINT ["python", "-u", "/box/entry.py"]
