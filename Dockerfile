# ShieldBench guard server (v0.42). Build:  docker build -t shieldbench .
# Run with a key:   docker run -p 8765:8765 -e SHIELDBENCH_KEY=<a long random string> shieldbench
# Public demo:      docker run -p 8765:8765 -e SHIELDBENCH_PUBLIC_DEMO=1 shieldbench   (no key; scan, agent and Claude guards are off; visitors are rate limited)
# Cloud hosts (Render, Fly, ...) set PORT; the server listens on it, or on 8765 when PORT is not set.
# The image has the regex and trained-detector guards and no bot attached (guard-only mode).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml* ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data/models ./data/models
COPY docs ./docs
ENV PYTHONPATH=/app/src
EXPOSE 8765
CMD ["sh", "-c", "python -m shieldbench.live --serve ${PORT:-8765} --host 0.0.0.0 --guard g2 --site docs"]
