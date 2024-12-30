# backend/app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from query_processing.query_handler import process_query
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

@app.route('/api/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided."}), 400
    
    query = data['query']
    top_k = data.get('top_k', 5)
    similarity_threshold = data.get('similarity_threshold', 0.6)
    response = process_query(query, top_k=top_k, similarity_threshold=similarity_threshold)
    
    return jsonify(response)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
