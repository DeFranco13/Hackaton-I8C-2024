from flask import Flask, render_template


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient', defaults={'name': None})
@app.route('/patient/<name>')
def show_product(name):
    if name:
        return
    else:
        return

if __name__ == '__main__':
    app.run(host="0.0.0.0")
