#!/bin/bash
# Trading Bot — Deployment
# Ausführen mit: bash setup.sh

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Trading Bot Deployment ==="

# Kaputten Ordner entfernen falls vorhanden (alter zip-Bug)
if [ -d '{data,signals,charts,utils}' ]; then
  echo "→ Entferne kaputten Ordner..."
  rm -rf '{data,signals,charts,utils}'
fi

echo "→ Starte Docker Build..."
docker compose -f docker-compose.simple.yml down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose.simple.yml up --build -d

echo ""
echo "✅ Fertig! Dashboard: http://$(hostname -I | awk '{print $1}'):5000"
