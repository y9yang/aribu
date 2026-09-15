FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore \
    HOME=/home/user PYTHONPATH=/home/user/app/src OMP_NUM_THREADS=2 \
    STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user && mkdir /home/user/app && chown user /home/user/app
WORKDIR /home/user/app

RUN pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install -r deploy/requirements.txt && python -c "import rasterio, torch; assert torch.version.cuda is None"

USER user
COPY --chown=user pyproject.toml ./
COPY --chown=user src/ src/
COPY --chown=user models/fusion.pt models/radar-only.pt models/
COPY --chown=user data/demo/ data/demo/
COPY --chown=user data/live/ data/live/

COPY --chown=user .streamlit/ .streamlit/
COPY --chown=user app/ app/

EXPOSE 8080
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
