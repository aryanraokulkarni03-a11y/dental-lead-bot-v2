# Medical Clinic Lead Generation SaaS

## Overview
FastAPI application for dental and dermatology clinic lead management with AI integration.

## Project Structure
- `main.py`: Core API logic and endpoints.
- `dashboard_html.html`: Frontend dashboard served at `/dashboard`.
- `static/`: Directory for static assets (images, CSS, JS).
- `requirements.txt`: Project dependencies.

## Recent Changes
- Added `/dashboard` endpoint to serve the HTML dashboard.
- Mounted `/static` directory for static file serving.
- Updated `requirements.txt` with `openai` and `pydantic`.
- Configured Git global identity.

## Endpoints
- `GET /health`: Health check.
- `GET /dashboard`: Leads dashboard.
- `GET /docs`: API documentation (Swagger).
- `POST /lead/dental`: Create dental lead.
- `POST /lead/dermatology`: Create dermatology lead.
