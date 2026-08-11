# Dockerfile

FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffers stdout —
# both make container logs behave more predictably.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv inside the container (same tool we use locally,
# for consistency between dev and production environments).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only dependency files first — this is a deliberate layer-caching
# optimization: Docker only re-installs dependencies when THESE files
# change, not on every single code change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Now copy the actual application code
COPY src/ ./src/
COPY data/sample_sales.csv ./data/sample_sales.csv

EXPOSE 8501

# Basic container health check — lets orchestration platforms know
# if the app is actually alive, not just "the process started."
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "src/ai_data_agent/ui/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]