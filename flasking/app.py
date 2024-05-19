from flask import Flask, request, jsonify
from model import generate_text

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    instruction = data.get('instruction', '')
    response = generate_text(instruction)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)