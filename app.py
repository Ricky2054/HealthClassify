import streamlit as st
import csv
import geocoder
import overpy
from geopy.distance import geodesic
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_health_data(csv_file):
    health_data = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if all(row.values()):  # Check if all fields are non-empty
                health_data.append({
                    'symptoms': row['symptoms'],
                    'disease': row['disease'],
                    'icd10_code': row['icd10_code'],
                    'severity': int(row['severity']),
                    'medication': row['medication']
                })
    return health_data

def preprocess_data(health_data):
    if not health_data:
        raise ValueError("No valid data found in the CSV file.")
    
    symptoms = [item['symptoms'] for item in health_data]
    diseases = [item['disease'] for item in health_data]
    medications = [item['medication'] for item in health_data]
    
    symptom_encoder = LabelEncoder()
    disease_encoder = LabelEncoder()
    medication_encoder = LabelEncoder()
    
    symptom_encoder.fit(sum([s.split(';') for s in symptoms], []))
    disease_encoder.fit(diseases)
    medication_encoder.fit(medications)
    
    X = [np.mean([symptom_encoder.transform([s])[0] for s in symptom.split(';')]) for symptom in symptoms]
    y_disease = disease_encoder.transform(diseases)
    y_medication = medication_encoder.transform(medications)
    
    return np.array(X).reshape(-1, 1), y_disease, y_medication, symptom_encoder, disease_encoder, medication_encoder

def create_model(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models(X, y_disease, y_medication):
    X_train, X_test, y_disease_train, y_disease_test, y_medication_train, y_medication_test = train_test_split(
        X, y_disease, y_medication, test_size=0.2, random_state=42
    )
    
    disease_model = create_model(X.shape[1], len(np.unique(y_disease)))
    medication_model = create_model(X.shape[1], len(np.unique(y_medication)))
    
    disease_model.fit(X_train, y_disease_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    medication_model.fit(X_train, y_medication_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    return disease_model, medication_model

def predict_health_info(symptoms, symptom_encoder, disease_encoder, medication_encoder, disease_model, medication_model):
    symptom_list = symptoms.split(';')
    try:
        encoded_symptoms = np.mean([symptom_encoder.transform([s])[0] for s in symptom_list]).reshape(1, -1)
    except ValueError as e:
        raise ValueError(f"One or more symptoms '{symptoms}' not found in training data. Please provide known symptoms.")
    
    predicted_disease = disease_encoder.inverse_transform(np.argmax(disease_model.predict(encoded_symptoms), axis=1))[0]
    predicted_medication = medication_encoder.inverse_transform(np.argmax(medication_model.predict(encoded_symptoms), axis=1))[0]
    return predicted_disease, predicted_medication

def get_real_time_location():
    g = geocoder.ip('me')
    return g.latlng if g.latlng else (0, 0)

def get_nearby_medical_centers(latitude, longitude):
    api = overpy.Overpass()
    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:5000,{latitude},{longitude});
      node["amenity"="clinic"](around:5000,{latitude},{longitude});
    );
    out body;
    """
    result = api.query(query)
    medical_centers = []
    for node in result.nodes:
        name = node.tags.get("name", "Unknown")
        lat = node.lat
        lon = node.lon
        distance = round(geodesic((latitude, longitude), (lat, lon)).kilometers, 2)
        phone = node.tags.get("phone", "N/A")
        medical_centers.append({
            'name': name,
            'distance': distance,
            'phone': phone
        })
    medical_centers.sort(key=lambda x: x['distance'])
    return medical_centers[:5]

def get_medical_advice(predicted_disease, predicted_medication, latitude, longitude):
    severity = np.random.choice(['mild', 'moderate', 'severe'], p=[0.6, 0.3, 0.1])
    
    if severity == 'severe':
        advice = "Seek immediate medical attention."
        medical_centers = get_nearby_medical_centers(latitude, longitude)
        return severity, advice, medical_centers
    elif severity == 'moderate':
        advice = "Consult with a healthcare provider soon."
        return severity, advice, [predicted_medication]
    else:
        advice = "Monitor your symptoms and rest."
        home_remedies = ["stay hydrated", "rest", "take over-the-counter medications if necessary"]
        return severity, advice, home_remedies

def evaluate_health(name, symptoms, symptom_encoder, disease_encoder, medication_encoder, disease_model, medication_model):
    predicted_disease, predicted_medication = predict_health_info(symptoms, symptom_encoder, disease_encoder, medication_encoder, disease_model, medication_model)
    latitude, longitude = get_real_time_location()
    severity, advice, recommendations = get_medical_advice(predicted_disease, predicted_medication, latitude, longitude)
    results = {
        'name': name,
        'symptoms': symptoms,
        'predicted_disease': predicted_disease,
        'predicted_medication': predicted_medication,
        'severity': severity,
        'advice': advice,
        'recommendations': recommendations,
        'latitude': latitude,
        'longitude': longitude
    }
    return results

def main():
    st.title("Advanced Health Evaluation App")
    
    csv_file = 'health_data.csv'
    try:
        health_data = load_health_data(csv_file)
        X, y_disease, y_medication, symptom_encoder, disease_encoder, medication_encoder = preprocess_data(health_data)
        disease_model, medication_model = train_models(X, y_disease, y_medication)
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        return

    name = st.text_input("Enter your name:")
    symptoms = st.text_area("Enter your symptoms (separate multiple symptoms with a semicolon ';'):")
    
    if st.button("Evaluate Health"):
        if name and symptoms:
            try:
                results = evaluate_health(name, symptoms, symptom_encoder, disease_encoder, medication_encoder, disease_model, medication_model)
                
                st.subheader("Evaluation Results")
                st.write(f"Name: {results['name']}")
                st.write(f"Symptoms: {results['symptoms']}")
                
                st.subheader("Predicted Information")
                st.write(f"Predicted Disease: {results['predicted_disease']}")
                st.write(f"Predicted Medication: {results['predicted_medication']}")
                
                st.subheader("Assessment")
                st.write(f"Severity: {results['severity']}")
                st.write(f"Advice: {results['advice']}")
                
                st.subheader("Recommendations")
                if isinstance(results['recommendations'], list):
                    for rec in results['recommendations']:
                        if isinstance(rec, dict):
                            st.write(f"- Name: {rec['name']}")
                            st.write(f"  Distance: {rec['distance']} km")
                            st.write(f"  Phone: {rec['phone']}")
                        else:
                            st.write(f"- {rec}")
                else:
                    st.write("No specific recommendations available.")
                
                st.subheader("Your Location")
                st.map(data={"lat": [results['latitude']], "lon": [results['longitude']]})
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"An error occurred during evaluation: {str(e)}")
        else:
            st.warning("Please enter both name and symptoms.")

if __name__ == '__main__':
    main()
