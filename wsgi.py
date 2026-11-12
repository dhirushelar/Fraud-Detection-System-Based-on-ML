"""
Fraud Detection System (FDS) - Production WSGI Entrypoint
Author: Alpha
"""

import os
from app import create_app

# Default to development configuration for local runs, override with FLASK_ENV
env = os.getenv("FLASK_ENV", "development")
app = create_app(env)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
