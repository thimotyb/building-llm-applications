FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system agent && adduser --system --ingroup agent agent

COPY ch13/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY --chown=agent:agent ch13 /app/ch13

USER agent

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/.well-known/agent-card.json', timeout=2)"]

CMD ["uvicorn", "ch13.a2a_server.agent:a2a_app", "--host", "0.0.0.0", "--port", "8001"]
