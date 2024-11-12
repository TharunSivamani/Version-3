from flask import Flask, render_template, jsonify, request
import threading
import queue

app = Flask(__name__)

# Queue to store training data
training_data = queue.Queue()
loss_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    data = request.json
    loss_history.append(data['loss'])
    training_data.put(data)
    return jsonify({"status": "success"})

@app.route('/get_data')
def get_data():
    if not training_data.empty():
        data = training_data.get()
        return jsonify(data)
    return jsonify({"status": "no_data"})

@app.route('/get_loss_history')
def get_loss_history():
    return jsonify({"loss_history": loss_history})

if __name__ == '__main__':
    app.run(debug=True) 