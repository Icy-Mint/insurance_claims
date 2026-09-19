# Build from the insurance_claims/ directory (one level up from this file)
# so both backend/ and fixtures/ are in the build context:
#   docker build -f backend/Dockerfile -t insurance-claims-backend .
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... insurance-claims-backend
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./backend/app
COPY fixtures ./fixtures

WORKDIR /app/backend

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
