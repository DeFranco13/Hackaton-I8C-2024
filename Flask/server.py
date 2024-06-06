from flask import Flask, jsonify, redirect


app = Flask(__name__)

@app.route('/patient', defaults={'name': None})
@app.route('/patient/<name>')
def show_product(name):
    if name:
        return
    else:
        return


app.run(host="0.0.0.0")
