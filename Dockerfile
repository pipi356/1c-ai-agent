FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    httpx==0.27.2 \
    pydantic==2.9.2 \
    qdrant-client==1.11.3 \
    python-multipart==0.0.12 \
    pypdf==5.0.0 \
    python-docx==1.1.2

COPY ./app /app

EXPOSE 8090

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
