# ShieldBench guard server (v0.39). Build:  docker build -t shieldbench .
# Run:    docker run -p 8765:8765 -e SHIELDBENCH_KEY=<a long random string> shieldbench
# Then point any OpenAI client at http://localhost:8765/v1 with that key as the API key.
# The image has the regex and trained-detector guards and no bot attached (guard-only mode); add --target openai and OPENAI_COMPAT_* settings to attach a model.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml* ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data/models ./data/models
COPY docs ./docs
ENV PYTHONPATH=/app/src
EXPOSE 8765
CMD ["python", "-m", "shieldbench.live", "--serve", "8765", "--host", "0.0.0.0", "--guard", "g2", "--site", "docs"]
