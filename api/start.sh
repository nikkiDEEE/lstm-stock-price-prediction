#!/usr/bin/env bash

cd "$(dirname "$0")"

echo "Starting Uvicorn server in production mode..."

uvicorn main:app --workers 1 --log-level info --port 8080
