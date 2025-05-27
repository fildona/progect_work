from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from sqlalchemy import func
import joblib
import pandas as pd
import numpy as np
import os
from db.db import SessionLocal, RichiestaFinanziamento
import requests
import io

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

# parte richieste + export

@app.route('/richieste')
def richieste_html():
    return render_template('richieste.html')

@app.route('/api/richieste')
def api_richieste():
    session = SessionLocal()
    query = session.query(RichiestaFinanziamento)

    # Filtri dinamici
    def get_range(query, field):
        min_val = request.args.get(f"{field}_min", type=float)
        max_val = request.args.get(f"{field}_max", type=float)
        if min_val is not None:
            query = query.filter(getattr(RichiestaFinanziamento, field) >= min_val)
        if max_val is not None:
            query = query.filter(getattr(RichiestaFinanziamento, field) <= max_val)
        return query

    # Range numerici
    for field in [
        "Eta", "RedditoLordoUltimoAnno", "AnniEsperienzaLavorativa", "ImportoRichiesto",
        "TassoInteresseFinanziamento", "ImportoRichiestoDivisoReddito", "DurataDellaStoriaCreditiziaInAnni",
        "AffidabilitàCreditizia", "ProbabilitaFinanziamentoApprovato"
    ]:
        query = get_range(query, field)

    # Filtri select
    for field in [
        "Sesso", "TitoloStudio", "InformazioniImmobile", "ScopoFinanziamento", "InadempienzeFinanziamentiPrecedenti"
    ]:
        value = request.args.get(field)
        if value:
            query = query.filter(getattr(RichiestaFinanziamento, field) == value)

    results = query.all()
    session.close()

    # Serializza risultati
    def serialize(obj):
        return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

    return jsonify([serialize(r) for r in results])

@app.route('/api/richieste/export')
def api_richieste_export():
    format = request.args.get("format", "csv")
    session = SessionLocal()
    query = session.query(RichiestaFinanziamento)

    def get_range(query, field):
        min_val = request.args.get(f"{field}_min", type=float)
        max_val = request.args.get(f"{field}_max", type=float)
        if min_val is not None:
            query = query.filter(getattr(RichiestaFinanziamento, field) >= min_val)
        if max_val is not None:
            query = query.filter(getattr(RichiestaFinanziamento, field) <= max_val)
        return query

    for field in [
        "Eta", "RedditoLordoUltimoAnno", "AnniEsperienzaLavorativa", "ImportoRichiesto",
        "TassoInteresseFinanziamento", "ImportoRichiestoDivisoReddito", "DurataDellaStoriaCreditiziaInAnni",
        "AffidabilitàCreditizia", "ProbabilitaFinanziamentoApprovato"
    ]:
        query = get_range(query, field)

    for field in [
        "Sesso", "TitoloStudio", "InformazioniImmobile", "ScopoFinanziamento", "InadempienzeFinanziamentiPrecedenti"
    ]:
        value = request.args.get(field)
        if value:
            query = query.filter(getattr(RichiestaFinanziamento, field) == value)

    results = query.all()
    session.close()

    def serialize(obj):
        return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

    data = [serialize(r) for r in results]
    df = pd.DataFrame(data)

    if format == "json":
        return jsonify(data)
    elif format == "excel":
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        return send_file(output, download_name="richieste.xlsx", as_attachment=True)
    else:  # csv
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode()), download_name="richieste.csv", as_attachment=True)

# parte statistiche

@app.route('/statistiche')
def statistiche_html():
    return render_template('statistiche.html')

@app.route('/api/statistiche')
def api_statistiche():
    session = SessionLocal()
    query = session.query(RichiestaFinanziamento)

    # Applica filtri se presenti
    for field in ["Sesso", "TitoloStudio", "InformazioniImmobile", "ScopoFinanziamento"]:
        value = request.args.get(field)
        if value:
            query = query.filter(getattr(RichiestaFinanziamento, field) == value)

    # 1. Grafico a torta: conteggio richieste per Sesso
    sesso_counts = dict(query.with_entities(
        RichiestaFinanziamento.Sesso, 
        func.count(RichiestaFinanziamento.RichiestaFinanziamentoID)
    ).group_by(RichiestaFinanziamento.Sesso).all())

    # 2. Grafico a barre verticali: somma ImportiRichiesti per InformazioniImmobile
    immobile_importi = dict(query.with_entities(
        RichiestaFinanziamento.InformazioniImmobile,
        func.sum(RichiestaFinanziamento.ImportoRichiesto)
    ).group_by(RichiestaFinanziamento.InformazioniImmobile).all())

    # 3. Grafico a barre orizzontali: somma ImportiRichiesti per TitoloStudio
    titolo_importi = dict(query.with_entities(
        RichiestaFinanziamento.TitoloStudio,
        func.sum(RichiestaFinanziamento.ImportoRichiesto)
    ).group_by(RichiestaFinanziamento.TitoloStudio).all())

    # 4. Grafico a scelta: conteggio per ScopoFinanziamento
    scopo_counts = dict(query.with_entities(
        RichiestaFinanziamento.ScopoFinanziamento,
        func.count(RichiestaFinanziamento.RichiestaFinanziamentoID)
    ).group_by(RichiestaFinanziamento.ScopoFinanziamento).all())

    session.close()
    return jsonify({
        "sesso_counts": sesso_counts,
        "immobile_importi": immobile_importi,
        "titolo_importi": titolo_importi,
        "scopo_counts": scopo_counts
    })

if __name__ == '__main__':
    app.run(debug=True)