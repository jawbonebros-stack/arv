#!/usr/bin/env python3
"""
Startup script for Onsenses ARV Platform
This script ensures proper initialization and startup for deployment
"""

import os
import sys
import uvicorn
from main import app

def main():
    # Get port from environment (for deployment) or default to 5000
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting Onsenses ARV Platform on {host}:{port}")
    
    # Run the application
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        access_log=True
    )

if __name__ == "__main__":
    main()