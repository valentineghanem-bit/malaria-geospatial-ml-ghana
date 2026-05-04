#!/usr/bin/env python3
"""Serve Ghana Malaria 260 Districts dashboard (stdlib only -- no extra dependencies)."""
import os, http.server, threading, webbrowser

PORT = 8050
DIR  = os.path.dirname(os.path.abspath(__file__))
FILE = "Ghana_Malaria_260District_Dashboard.html"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)
    def log_message(self, *a):
        pass

def _open():
    import time; time.sleep(0.8)
    webbrowser.open(f"http://127.0.0.1:{PORT}/{FILE}")

if __name__ == "__main__":
    threading.Thread(target=_open, daemon=True).start()
    print(f"Dashboard at http://127.0.0.1:{PORT}/{FILE}")
    print("Press Ctrl+C to stop.")
    http.server.HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
