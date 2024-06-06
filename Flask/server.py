from flask import Flask, render_template, jsonify, request
import requests
import json
import googlemaps
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)


ALLOWED_EXTENSIONS = {'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if  file and allowed_file(file.filename):
        content = file.read()
    return content

@app.route('/route/<destLocation>')
def get_route_info(destLocation):

    originLocation = "Veldkant 33A, 2550 Kontich"
    gmaps = googlemaps.Client(key=os.getenv('GMAPS_KEY'))
    directions_result = gmaps.directions(originLocation,
                                     destLocation,
                                     mode="driving")
    distanceSplits = (directions_result[0]['legs'][0]['distance']['text'].split(' '))
    driveTimeSplits = directions_result[0]['legs'][0]['duration']['text'].split(' ')
    
    if distanceSplits[1] != "km":
        distance = float(distanceSplits[0]) / 1000
    else:
        distance = float(distanceSplits[0])
        print(distance)

    if driveTimeSplits[1] == "hours":
        driveTime = float(driveTimeSplits[0]) * 60 + float(driveTimeSplits[2])
    else:
        driveTime = float(driveTimeSplits[0])

    return {
        "patientLocation": {
            "latitude": directions_result[0]['legs'][0]['end_location']['lat'],
            "longitude": directions_result[0]['legs'][0]['end_location']['lng']
        },
        "nurseLocation": {
            "latitude": directions_result[0]['legs'][0]['start_location']['lat'],
            "longitude": directions_result[0]['legs'][0]['start_location']['lng']
        },
        "distance": distance,
        "driveTime": driveTime
    }

    
    








if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
