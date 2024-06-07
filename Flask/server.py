from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import json
import googlemaps
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import asyncio

load_dotenv()

app = Flask(__name__)
CORS(app)


ALLOWED_EXTENSIONS = {'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def geo_anomalies():
    # Data laden
    data = pd.read_json('./Flask/dataset.json')

    # Voorbereiding van de features
    X = data[['driveTime', 'distance']].values

    # Standaardisatie van de features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Definieer de parameter grid voor K-means
    param_grid = {
        'n_clusters': range(2, 6),
        'n_init': [10, 20, 30, 40],
        'max_iter': [300, 400, 500]
    }

    # K-means model
    kmeans = KMeans(random_state=42)

    # Gebruik GridSearchCV voor hyperparameter optimalisatie
    grid_search = GridSearchCV(kmeans, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_scaled)

    # Beste model kiezen
    best_kmeans = grid_search.best_estimator_
    # clusters = best_kmeans.predict(X_scaled)

    # Afstanden van elk punt tot het clustercentrum berekenen
    distances = best_kmeans.transform(X_scaled).min(axis=1)
    distance_threshold_high = np.percentile(distances, 85)
    distance_threshold_medium = np.percentile(distances, 60)

    # Risiconiveaus toewijzen op basis van afstand en reistijd
    data['Fraud Risk'] = np.where(distances >= distance_threshold_high, 'High Risk',
                                np.where(distances >= distance_threshold_medium, 'Medium Risk', 'Low Risk'))

    # Simulatie van 'true labels' voor demonstratiedoeleinden
    thresholds = data[['driveTime', 'distance']].quantile(0.85)
    true_labels = ((data['driveTime'] > thresholds['driveTime']) | 
                (data['distance'] > thresholds['distance'])).astype(int)
    predicted_labels = data['Fraud Risk'].apply(lambda x: 1 if x != 'Low Risk' else 0)

    # Prestatiemetingen
    accuracy = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticklabels(['', 'Low Risk', 'Medium/High Risk'])
    ax.set_yticklabels(['', 'Low Risk', 'Medium/High Risk'])
    # plt.show()

    # Filter en toon records die geclassificeerd zijn als 'High Risk' en 'Medium Risk'
    high_risk_records = data[data['Fraud Risk'] == 'High Risk']
    medium_risk_records = data[data['Fraud Risk'] == 'Medium Risk']
    low_risk_records = data[data['Fraud Risk'] == 'Low Risk']

    print("Records die hoogstwaarschijnlijk fraude zijn (High Risk):")
    print(high_risk_records)
    print("Records die mogelijk fraude zijn (Medium Risk):")
    print(medium_risk_records)

    # Tel het aantal records in elke risicocategorie
    high_risk_count = high_risk_records.shape[0]
    medium_risk_count = medium_risk_records.shape[0]
    low_risk_count = low_risk_records.shape[0]

    print(f"Aantal 'High Risk' records: {high_risk_count}")
    print(f"Aantal 'Medium Risk' records: {medium_risk_count}")
    print(f"Aantal 'Low Risk' records: {low_risk_count}")

    # Exporteer de 'High Risk' en 'Medium Risk' records naar een JSON-object
    high_risk_records_json = high_risk_records.to_json(orient='records', lines=True)
    medium_risk_records_json = medium_risk_records.to_json(orient='records', lines=True)
    print("JSON Output van hoogstwaarschijnlijke fraude records (High Risk):")
    print(high_risk_records_json)
    print("JSON Output van mogelijk frauduleuze records (Medium Risk):")
    print(medium_risk_records_json)

    # Visualisatie van de clusters
    plt.scatter(data['driveTime'], data['distance'], c=data['Fraud Risk'].map({'Low Risk': 'green', 'Medium Risk': 'yellow', 'High Risk': 'red'}))
    plt.xlabel('Drive Time')
    plt.ylabel('Distance')
    plt.colorbar()
    # plt.show()

    return {"medium": medium_risk_records_json, "high": high_risk_records_json}

async def visit_anomalies():
    # Gegevens inladen
    data = pd.read_json('./Flask/dataset.json')

    # Tijdsstempels converteren naar datetime
    data['visit_timestamp'] = pd.to_datetime_ext(data['visit_timestamp'])

    # Functie om verdachte scanuren te controleren
    def check_strange_hours(data):
        return data[data['visit_timestamp'].dt.hour > 22]

    # Functie om frequente zorg op dezelfde dag te detecteren
    def check_frequent_visits(data):
        daily_visits = data.groupby(['rijksregisterPatient', data['visit_timestamp'].dt.date]).size()
        return daily_visits[daily_visits > 3]

    # Functie om te controleren of dezelfde zorg door verschillende zorgverleners op dezelfde dag wordt verleend
    def check_multiple_nurses(data):
        service_per_day = data.groupby(['rijksregisterPatient', 'service', data['visit_timestamp'].dt.date])['rijksregisterNurse'].nunique()
        return service_per_day[service_per_day > 1]

    # Afstandsberekening tussen twee locaties
    from geopy.distance import geodesic

    # Functie om de afstand te controleren
    def check_location_mismatch(data):
        suspicious_distance = []
        for index, row in data.iterrows():
            nurse_loc = (row['visit']['nurseLocation']['latitude'], row['visit']['nurseLocation']['longtitude'])
            patient_loc = (row['visit']['patientLocation']['latitude'], row['visit']['patientLocation']['longtitude'])
            if geodesic(nurse_loc, patient_loc).kilometers > 50:  # Meer dan 50 km afstand is verdacht
                suspicious_distance.append(row['visit']['id'])
        return suspicious_distance

    # Pas deze functies toe om de respectievelijke checks uit te voeren
    strange_hours = check_strange_hours(data)
    frequent_visits = check_frequent_visits(data)
    multiple_nurses = check_multiple_nurses(data)
    location_mismatches = check_location_mismatch(data)

    # Resultaten printen
    print("Strange Hours Visits:", strange_hours)
    print("Frequent Visits:", frequent_visits)
    print("Multiple Nurses Same Day Service:", multiple_nurses)
    print("Location Mismatches:", location_mismatches)

    return 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/anomalies')
async def anomalies():
    geoAnomalies = await geo_anomalies(); 

    return geoAnomalies

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if  file and allowed_file(file.filename):
        content = file.read()
    
    with open('./Flask/dataset.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)

    json_object.append(json.loads(content))

    with open("./Flask/dataset.json", "w") as outfile:
        json.dump(json_object, outfile, indent=4)
    
    return {"status": "OK", "code": 200}

@app.route('/route/<destLocation>')
def get_route_info(destLocation):

    originLocation = "Veldkant 33A, 2550 Kontich" # Hardcoded 'nurse' adress for DEMO purposes
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
