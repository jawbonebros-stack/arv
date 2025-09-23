#!/bin/bash
# Onsenses ARV Application Startup Script
# This script ensures proper deployment initialization

# Install dependencies if needed
echo "Starting Onsenses ARV..."

# Run database initialization (optional - it's also done in startup event)
echo "Initializing database..."

# Start the FastAPI application with Uvicorn
echo "Starting FastAPI server on port 5000..."
exec uvicorn main:app --host 0.0.0.0 --port 5000 --reload=false