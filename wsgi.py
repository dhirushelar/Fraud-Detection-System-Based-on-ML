"""
Fraud Detection System (FDS) - Production WSGI Entrypoint
Author: Alpha
"""

import os
from app import create_app

# Default to production configuration in containerized environments
env = os.getenv("FLASK_ENV", "production")
app = create_app(env)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
