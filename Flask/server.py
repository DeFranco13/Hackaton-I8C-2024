from flask import Flask, render_template, Response, jsonify
import requests
import json
import googlemaps

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

@app.route('/route/<destLocation>')
def get_route_info(destLocation):
    
    originLocation = "Veldkant 33A, 2550 Kontich"
    gmaps = googlemaps.Client(key='AIzaSyAboDqid2eYqKzbDys4ACzEh479k5lAM3k')
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
