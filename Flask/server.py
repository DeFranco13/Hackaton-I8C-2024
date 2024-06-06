from flask import Flask, render_template, Response, jsonify


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient', defaults={'name': None})
@app.route('/patient/<name>')
def show_product(name):
    if name:
        return name
    else:
        return jsonify({'error': 'Bad Request', 'details': 'No patient name provided.'}), 400, {"Content-Type": "application/json"}

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
