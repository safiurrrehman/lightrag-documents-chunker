"""
Mock LightRAG Server

This is a simple Flask server that mimics a LightRAG deployment for testing purposes.
It provides endpoints for health checks and upserting chunks.
"""

from flask import Flask, request, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

# Create a directory to store upserted chunks
os.makedirs('mock_lightrag_data', exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/upsert', methods=['POST'])
def upsert_chunks():
    """Endpoint to upsert chunks."""
    try:
        # Get request data
        data = request.json
        
        if not data or 'chunks' not in data:
            return jsonify({"error": "Invalid request format"}), 400
        
        chunks = data['chunks']
        batch_number = data.get('batch_number', 0)
        
        # Save chunks to file for inspection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mock_lightrag_data/chunks_{timestamp}_batch_{batch_number}.json"
        
        with open(filename, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        # Return success response
        return jsonify({
            "status": "success",
            "message": f"Upserted {len(chunks)} chunks",
            "saved_to": filename
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Mock LightRAG Server on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000)
