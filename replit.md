# Medical Clinic Lead Generation SaaS

## Overview
FastAPI application for dental and dermatology clinic lead management with AI integration.

## Project Structure
- `src/server.py`: Core API logic and endpoints.
- `src/static/`: Directory for static assets and dashboard.
- `requirements.txt`: Project dependencies.

## Recent Changes
- Refactored project structure to `src/`.
- Fixed YCloud webhook signature verification.
- Switched to `gpt-4o` for faster and better AI responses.
- Updated dashboard path logic.

## Endpoints
- `GET /health`: Health check.
- `GET /dashboard`: Leads dashboard.
- `GET /docs`: API documentation (Swagger).
- `POST /webhook/ycloud`: WhatsApp integration.
- `POST /lead/dental`: Create dental lead.
- `POST /lead/dermatology`: Create dermatology lead.
