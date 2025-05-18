#!/usr/bin/env python
"""
Run script for the RAG System Visualization Interface

This script starts the visualization web server and opens the dashboard in a browser.
"""

import os
import sys
import webbrowser
import time
import logging
from threading import Timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def open_browser():
    """Open the browser to the visualization dashboard."""
    webbrowser.open('http://localhost:5001')

def main():
    """Main function to run the visualization interface."""
    try:
        # Create necessary directories if they don't exist
        os.makedirs(os.path.join('visualization', 'data'), exist_ok=True)
        os.makedirs(os.path.join('visualization', 'static', 'js'), exist_ok=True)
        os.makedirs(os.path.join('visualization', 'static', 'css'), exist_ok=True)
        os.makedirs(os.path.join('visualization', 'templates'), exist_ok=True)
        
        # Change to the visualization directory
        os.chdir('visualization')
        
        # Import the Flask app
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from visualization.app import app
        
        # Open browser after a delay to ensure server is running
        Timer(1.5, open_browser).start()
        
        # Run the Flask app
        logger.info("Starting visualization server at http://localhost:5001")
        app.run(host='0.0.0.0', port=5001, debug=False)
        
    except Exception as e:
        logger.error(f"Error running visualization interface: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
