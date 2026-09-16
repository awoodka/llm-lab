# LiveCodeBench code generation, with LCB's own prompt, code extraction and grader (MIT, pinned commit).
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
RUN pip install --no-cache-dir numpy==2.5.3 tqdm==4.70.1
COPY livecodebench/fetch.py livecodebench/lcb_files.sha256 /build/
RUN python /build/fetch.py /opt/lcb /build/lcb_files.sha256
COPY common/boxproto.py /box/boxproto.py
COPY livecodebench/entry.py /box/entry.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/lcb:/box TMPDIR=/tmp
# LCB reads its few-shot files relative to the repository root.
WORKDIR /opt/lcb
ENTRYPOINT ["python", "-u", "/box/entry.py"]
