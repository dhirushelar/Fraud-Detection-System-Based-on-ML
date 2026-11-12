"""
Fraud Detection System (FDS) - Production-Grade Local & Development Server
Author: Alpha
"""

import os
import sys
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app


class ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
    """Threaded WSGI server capable of handling multiple concurrent requests."""
    daemon_threads = True
    allow_reuse_address = True


class CleanLoggingHandler(WSGIRequestHandler):
    """Clean HTTP request logging for development and demonstration."""
    def log_message(self, format, *args):
        # Format: [HTTP] 127.0.0.1 - - [timestamp] "GET /api/v1/health HTTP/1.1" 200 -
        sys.stdout.write(f"[HTTP] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()


def run():
    env = os.getenv("FLASK_ENV", "development")
    app = create_app(env)

    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "127.0.0.1")

    print("=" * 72)
    print("  FRAUD DETECTION SYSTEM (FDS) - REAL-TIME ANOMALY DETECTOR")
    print("=" * 72)
    print(f"[*] Environment:          {env.capitalize()}")
    print(f"[*] Base URL:             http://{host}:{port}")
    print(f"[*] Executive Dashboard:  http://{host}:{port}/dashboard")
    print(f"[*] Live Alert Queue:     http://{host}:{port}/dashboard/alerts")
    print(f"[*] Scoring Simulator:    http://{host}:{port}/dashboard/simulate")
    print(f"[*] REST API Health:      http://{host}:{port}/api/v1/health")
    print(f"[*] REST API Metrics:     http://{host}:{port}/api/v1/metrics")
    print("=" * 72)
    print(f"[*] Server listening on {host}:{port} (Threaded)... Press Ctrl+C to stop.")
    sys.stdout.flush()

    httpd = make_server(host, port, app, server_class=ThreadedWSGIServer, handler_class=CleanLoggingHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Gracefully shutting down Fraud Detection System server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()
