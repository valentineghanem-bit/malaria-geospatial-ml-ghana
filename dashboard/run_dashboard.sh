#!/bin/bash
# macOS / Linux launcher -- Ghana Malaria Dashboard
cd "$(dirname "$0")"
echo Starting Ghana Malaria Dashboard...
python3 app.py &
sleep 1 && open "http://127.0.0.1:8050" 2>/dev/null || xdg-open "http://127.0.0.1:8050"
wait
