from flask import Flask, render_template, jsonify, request
import threading
import queue

app = Flask(__name__)

# Store training data
training_data = queue.Queue()
loss_history = []
accuracy_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    try:
        data = request.json
        # Store both loss and accuracy
        loss_history.append(data['loss'])
        accuracy_history.append(data['accuracy'])
        training_data.put(data)
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error in update: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_data')
def get_data():
    try:
        if not training_data.empty():
            data = training_data.get()
            return jsonify(data)
        return jsonify({"status": "no_data"})
    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_loss_history')
def get_loss_history():
    try:
        return jsonify({
            "loss_history": loss_history,
            "accuracy_history": accuracy_history
        })
    except Exception as e:
        print(f"Error in get_loss_history: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Add a route to clear history (useful for multiple training runs)
@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        loss_history.clear()
        accuracy_history.clear()
        while not training_data.empty():
            training_data.get()
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error in clear_history: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 