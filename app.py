from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from db.db import SessionLocal, RichiestaFinanziamento
import requests

app = Flask(__name__)
CORS(app)

# Carica modello e scaler
MODEL_PATH = os.path.join("model_scaler", "best_model.pkl")
SCALER_PATH = os.path.join("model_scaler", "best_scaler.pkl")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Colonne attese dal modello (ordine importante!)
MODEL_COLUMNS = [
    'Eta', 'RedditoLordoUltimoAnno', 'AnniEsperienzaLavorativa', 'ImportoRichiesto',
    'TassoInteresseFinanziamento', 'ImportoRichiestoDivisoReddito', 'DurataDellaStoriaCreditiziaInAnni',
    'AffidabilitàCreditizia',
    'Sesso_M',
    'TitoloStudio_Dottorato di ricerca', 'TitoloStudio_Laurea',
    'InformazioniImmobile_ProprietàMutuoDaEstinguere', 'InformazioniImmobile_ProprietàMutuoEstinto',
    'ScopoFinanziamento_InizioAttivitaImprenditoriale', 'ScopoFinanziamento_Medico',
    'ScopoFinanziamento_Personale', 'ScopoFinanziamento_RistrutturazioneAltriDebiti',
    'ScopoFinanziamento_RistrutturazioneCasa',
    'InadempienzeFinanziamentiPrecedenti_SI'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Crea DataFrame da input
    input_df = pd.DataFrame([data])

    # Codifica le categoriche come nel training
    input_df = pd.get_dummies(input_df)
    # Aggiungi colonne mancanti (set a 0)
    for col in MODEL_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0
    # Ordina le colonne come nel training
    input_df = input_df[MODEL_COLUMNS]

    # Applica scaler
    X_scaled = scaler.transform(input_df)

    # Predici probabilità e classe
    proba = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]
    classe = "SI" if pred == 1 else "NO"

    return jsonify({
        "probabilita_approvazione": float(proba),
        "classe_prevista": classe
    })

# parte database
@app.route('/importa', methods=['POST'])
def importa():
    url = "https://testbobphp2.altervista.org/000AiForemaProjectWork/richieste_finanziamenti.php"
    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({"error": "Errore nel recupero dati"}), 500
    richieste = response.json()
    session = SessionLocal()
    nuovi = 0
    for r in richieste:
        # Controlla se già presente
        if session.query(RichiestaFinanziamento).filter_by(RichiestaFinanziamentoID=r["RichiestaFinanziamentoID"]).first():
            continue
        # Calcola probabilità
        input_df = pd.DataFrame([r])
        input_df = pd.get_dummies(input_df)
        for col in MODEL_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[MODEL_COLUMNS]
        X_scaled = scaler.transform(input_df)
        proba = float(model.predict_proba(X_scaled)[0, 1])
        # Crea oggetto e salva
        richiesta = RichiestaFinanziamento(
            ProbabilitaFinanziamentoApprovato=proba,
            **r
        )
        session.add(richiesta)
        nuovi += 1
    session.commit()
    session.close()
    return jsonify({"importati": nuovi})

@app.route('/importa_html')
def importa_html():
    return render_template('importa.html')

if __name__ == '__main__':
    app.run(debug=True)