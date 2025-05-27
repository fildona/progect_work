from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
from flask_cors import CORS
import requests
from pymongo import MongoClient

print("--- Script Iniziato ---")

# --- Configurazione Nomi File Modello ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(MODEL_DIR, 'GradientBoosting_model.pkl')
SCALER_FILENAME = os.path.join(MODEL_DIR, 'loan_scaler.joblib')
MODEL_COLUMNS_FILENAME = os.path.join(MODEL_DIR, 'model_columns.joblib')
COLS_TO_SCALE_FILENAME = os.path.join(MODEL_DIR, 'cols_to_scale.joblib')
TRAINING_BOUNDS_FILENAME = os.path.join(MODEL_DIR, 'training_bounds.joblib')

# --- Caricamento Modello, Scaler, Colonne ---
print("Caricamento componenti modello...")
model, scaler, model_columns, cols_to_scale, training_bounds = None, None, [], [], None
try:
    model = joblib.load(MODEL_FILENAME)
    print(f"OK: Modello caricato da {MODEL_FILENAME}")
    scaler = joblib.load(SCALER_FILENAME)
    print(f"OK: Scaler caricato da {SCALER_FILENAME}")
    model_columns = joblib.load(MODEL_COLUMNS_FILENAME)
    print(f"OK: Colonne modello caricate da {MODEL_COLUMNS_FILENAME} (Tot: {len(model_columns)})")
    # print(f"   Colonne modello attese: {model_columns}") # Decommenta per debug
    cols_to_scale = joblib.load(COLS_TO_SCALE_FILENAME)
    print(f"OK: Colonne da scalare caricate da {COLS_TO_SCALE_FILENAME}")
    # print(f"   Colonne da scalare attese: {cols_to_scale}") # Decommenta per debug
except Exception as e_load:
    print(f"!!! ERRORE CRITICO durante il caricamento dei file modello/scaler: {e_load}")
    print("!!! L'applicazione potrebbe non funzionare correttamente.")

try:
    training_bounds = joblib.load(TRAINING_BOUNDS_FILENAME)
    print(f"OK: Limiti training caricati da {TRAINING_BOUNDS_FILENAME}")
except FileNotFoundError:
    training_bounds = None
    print(f"INFO: File dei limiti training '{TRAINING_BOUNDS_FILENAME}' non trovato. Clipping (se usato) sarà disabilitato.")
except Exception as e_bounds:
    training_bounds = None
    print(f"ATTENZIONE: Errore caricamento limiti training '{TRAINING_BOUNDS_FILENAME}': {e_bounds}. Clipping disabilitato.")
print("--- Caricamento componenti modello completato ---")


app = Flask(__name__)
CORS(app)

# --- Configurazione MongoDB ---
# !!! MODIFICA CON LA TUA PASSWORD REALE QUI !!!

MONGO_URI = "mongodb+srv://esame:esame@esame.ynuzlxk.mongodb.net/?retryWrites=true&w=majority&appName=Esame"
DB_NAME = "finanziamenti_db_final_test" # Scegli un nome per il tuo DB
COLLECTION_NAME = "richieste_final_test" # Scegli un nome per la tua collezione

print(f"\n--- Configurazione MongoDB ---")
client = None
richieste_collection = None
try:
    print(f"Tentativo di connessione a MongoDB Atlas...") # Non stampare l'URI completo con password in produzione
    client = MongoClient(MONGO_URI)
    client.admin.command('ping')
    print("OK: Connessione a MongoDB Atlas riuscita!")
    db = client[DB_NAME]
    richieste_collection = db[COLLECTION_NAME]
    # Crea indice se non esiste. background=True è utile per non bloccare l'avvio.
    richieste_collection.create_index("RichiestaFinanziamentoID", unique=True, background=True)
    print(f"OK: Database '{DB_NAME}', Collezione '{COLLECTION_NAME}' pronta.")
    print("OK: Indice univoco su 'RichiestaFinanziamentoID' assicurato/creato.")
except Exception as e_mongo_conn:
    print(f"!!! ERRORE CRITICO DI CONNESSIONE A MONGODB: {e_mongo_conn}")
    print("!!! L'importazione dati fallirà. Controlla URI, password, whitelist IP e `dnspython`.")
print("--- Configurazione MongoDB completata ---\n")


# --- Funzione Helper per Preprocessing e Predizione ---
def get_prediction_for_request_simplified(id_req_debug, input_data_api_format):
    print(f"\n--- Predizione per ID_API_FORMAT: {id_req_debug} ---")
    if not all([model, scaler, model_columns, cols_to_scale]):
        print(f"!!! Predizione ID {id_req_debug}: Modello o componenti essenziali non caricati. Ritorno 0.0")
        return 0.0

    # 1. Mappatura nomi API -> nomi interni e conversioni tipo base
    data_internal_format = {
        "Eta": float(input_data_api_format.get("eta_cliente", 0)),
        "Sesso": str(input_data_api_format.get("sesso_cliente", "N/D")),
        "TitoloStudio": str(input_data_api_format.get("titolo_studio_cliente", "N/D")),
        "RedditoLordoUltimoAnno": float(input_data_api_format.get("reddito_annuo_lordo", 0)),
        "AnniEsperienzaLavorativa": float(input_data_api_format.get("anni_esperienza_lavorativa", 0)),
        "InformazioniImmobile": str(input_data_api_format.get("info_immobile", "N/D")),
        "ImportoRichiesto": float(input_data_api_format.get("importo_richiesto_finanziamento", 0)),
        "ScopoFinanziamento": str(input_data_api_format.get("scopo_finanziamento", "N/D")),
        "TassoInteresseFinanziamento": float(input_data_api_format.get("tasso_interesse_proposto", 0)),
        "ImportoRichiestoDivisoReddito": float(input_data_api_format.get("rapporto_importo_reddito", 999)),
        "DurataDellaStoriaCreditiziaInAnni": float(input_data_api_format.get("durata_storia_creditizia_anni", 0)),
        "AffidabilitàCreditizia": float(input_data_api_format.get("punteggio_affidabilita_creditizia", 300)),
        "InadempienzeFinanziamentiPrecedenti": "SI" if input_data_api_format.get("inadempienze_pregresse") == 1 else "NO"
    }
    # print(f"ID {id_req_debug} - Dati MAPPATI (formato interno):\n{pd.Series(data_internal_format)}")

    # 2. Clipping (se training_bounds è disponibile)
    if training_bounds:
        for col, limits in training_bounds.items():
            if col in data_internal_format and pd.notna(data_internal_format[col]):
                val = float(data_internal_format[col])
                min_b, max_b = limits.get('min'), limits.get('max')
                if pd.notna(min_b) and pd.notna(max_b):
                     data_internal_format[col] = np.clip(val, min_b, max_b)

    current_input_df = pd.DataFrame([data_internal_format])
    # print(f"ID {id_req_debug} - DF Iniziale (dopo mappatura API e clipping):\n{current_input_df.T}")

    # 3. Mappature (Sesso, TitoloStudio, Inadempienze)
    map_titolo = {"Diploma": 0, "Laurea": 1, "Dottorato di ricerca": 2, "N/D": 0}
    map_sesso = {"F": 0, "M": 1, "N/D": 0}
    map_inad = {"NO": 0, "SI": 1}

    current_input_df["TitoloStudio"] = current_input_df["TitoloStudio"].map(map_titolo).fillna(0)
    current_input_df["Sesso"] = current_input_df["Sesso"].map(map_sesso).fillna(0)
    current_input_df["InadempienzeFinanziamentiPrecedenti"] = current_input_df["InadempienzeFinanziamentiPrecedenti"].map(map_inad).fillna(0)
    # print(f"ID {id_req_debug} - DF dopo mappature S,T,I:\n{current_input_df.T}")

    # 4. One-Hot Encoding
    cols_for_ohe = []
    if "InformazioniImmobile" in current_input_df.columns: cols_for_ohe.append("InformazioniImmobile")
    if "ScopoFinanziamento" in current_input_df.columns: cols_for_ohe.append("ScopoFinanziamento")
    if cols_for_ohe:
        current_input_df = pd.get_dummies(current_input_df, columns=cols_for_ohe, dummy_na=False)
    # print(f"ID {id_req_debug} - DF dopo OHE (colonne: {current_input_df.columns.tolist()}):\n{current_input_df.T}")

    # 5. Allinea colonne con quelle del modello e fillna(0) -> CRUCIALE
    df_aligned_for_model = pd.DataFrame(columns=model_columns) # Inizia con DF con le colonne del modello
    for col in model_columns:
        if col in current_input_df.columns:
            df_aligned_for_model[col] = current_input_df[col]
    df_aligned_for_model = df_aligned_for_model.fillna(0)
    # print(f"ID {id_req_debug} - DF Allineato a model_columns (colonne: {df_aligned_for_model.columns.tolist()}):\n{df_aligned_for_model.T}")

    # 6. Assicura che tutte le colonne siano numeriche
    for col in df_aligned_for_model.columns:
        df_aligned_for_model[col] = pd.to_numeric(df_aligned_for_model[col], errors='coerce').fillna(0)
    # print(f"ID {id_req_debug} - DF dopo to_numeric finale:\n{df_aligned_for_model.T}")


    # 7. Normalizzazione
    actual_cols_to_scale_in_df = [col for col in cols_to_scale if col in df_aligned_for_model.columns]
    if actual_cols_to_scale_in_df:
        # print(f"ID {id_req_debug} - Colonne DA SCALARE: {actual_cols_to_scale_in_df}")
        # print(f"    ID {id_req_debug} - Valori PRIMA scaling:\n{df_aligned_for_model[actual_cols_to_scale_in_df].T}")
        try:
            # Lo scaler si aspetta float
            df_aligned_for_model[actual_cols_to_scale_in_df] = df_aligned_for_model[actual_cols_to_scale_in_df].astype(float)
            scaled_values = scaler.transform(df_aligned_for_model[actual_cols_to_scale_in_df])
            df_aligned_for_model[actual_cols_to_scale_in_df] = scaled_values
            # print(f"    ID {id_req_debug} - Valori DOPO scaling:\n{df_aligned_for_model[actual_cols_to_scale_in_df].T}")
        except Exception as e_scale:
            print(f"!!! ID {id_req_debug} - Errore SCALING: {e_scale}. Ritorno 0.0")
            print(f"    Dati per lo scaler: {df_aligned_for_model[actual_cols_to_scale_in_df]}")
            return 0.0
    else:
        print(f"ID {id_req_debug} - Nessuna colonna da scalare trovata in df_aligned_for_model.")


    # 8. Predizione
    try:
        # Assicurati che l'input per il modello sia ESATTAMENTE df_aligned_for_model
        # e che le sue colonne siano quelle in model_columns
        final_input_payload = df_aligned_for_model[model_columns]
        # print(f"ID {id_req_debug} - Input FINALE per il modello (shape: {final_input_payload.shape}, Dtypes:\n{final_input_payload.dtypes}):\n{final_input_payload.T}")

        if final_input_payload.isnull().values.any():
            print(f"!!! ID {id_req_debug} - NaN TROVATI nell'input finale del modello! Colonne con NaN:")
            print(final_input_payload.columns[final_input_payload.isnull().any()].tolist())
            return 0.0
        if np.isinf(final_input_payload.to_numpy()).any(): # Converti a numpy per isinf
            print(f"!!! ID {id_req_debug} - Valori Infiniti TROVATI nell'input finale del modello!")
            return 0.0

        probability = model.predict_proba(final_input_payload)[0, 1]
        print(f"ID {id_req_debug} - Probabilità calcolata: {probability:.4f}")
        return float(probability)
    except Exception as e_predict:
        print(f"!!! ID {id_req_debug} - Errore PREDIZIONE: {e_predict}. Ritorno 0.0")
        print(f"    Forma dati per predizione: {final_input_payload.shape if 'final_input_payload' in locals() else 'N/A'}")
        # print(f"    Colonne dati per predizione: {final_input_payload.columns.tolist() if 'final_input_payload' in locals() else 'N/A'}")
        return 0.0


# --- Endpoint per l'importazione dei dati (Parte 3) ---
EXTERNAL_API_URL = "https://testbobphp2.altervista.org/000AiForemaProjectWork/richieste_finanziamenti.php"

@app.route('/import_data', methods=['POST'])
def import_data_route():
    print("\n--- Endpoint /import_data chiamato ---")
    if richieste_collection is None:
        print("!!! /import_data: Connessione a MongoDB non disponibile!")
        return jsonify({"message": "Errore DB: Connessione non disponibile", "imported": 0, "skipped": 0}), 503

    try:
        response = requests.get(EXTERNAL_API_URL, timeout=25)
        response.raise_for_status() # Controlla errori HTTP
        api_data_list = response.json()
        print(f"Ricevuti {len(api_data_list)} record dall'API esterna.")
    except requests.exceptions.Timeout:
        print(f"!!! Timeout chiamata API esterna: {EXTERNAL_API_URL}")
        return jsonify({"message": "Timeout API esterna", "imported": 0, "skipped": 0}), 504
    except requests.exceptions.RequestException as e_api:
        print(f"!!! Errore chiamata API esterna: {e_api}")
        return jsonify({"message": f"Errore API esterna: {str(e_api)}", "imported": 0, "skipped": 0}), 502
    except ValueError as e_json: # Errore nel decodificare JSON
        print(f"!!! Errore decodifica JSON da API esterna: {e_json}")
        return jsonify({"message": "Formato risposta API esterna non valido (JSON)", "imported": 0, "skipped": 0}), 502

    imported_count = 0
    skipped_count = 0
    error_processing_count = 0

    for item_original_api in api_data_list:
        # !!! DEVI VERIFICARE IL NOME ESATTO DEL CAMPO ID DALL'API ESTERNA !!!
        # Se l'API usa "RichiestaFinanziamentoID", va bene. Se usa "RichiestaFinanziamentoID", cambia qui.
        id_api = item_original_api.get("RichiestaFinanziamentoID") # <--- VERIFICA QUESTO NOME CAMPO!

        if not id_api:
            print(f"Record API skippato, ID ('RichiestaFinanziamentoID') mancante: {item_original_api}")
            error_processing_count += 1
            continue

        if richieste_collection.count_documents({"RichiestaFinanziamentoID": id_api}) > 0:
            skipped_count += 1
            continue

        # La funzione get_prediction_for_request_simplified prende i dati originali dell'API
        probabilita = get_prediction_for_request_simplified(str(id_api), item_original_api.copy())

        document_to_save = item_original_api.copy()
        document_to_save["RichiestaFinanziamentoID"] = id_api # Standardizza il nome dell'ID nel DB
        document_to_save["ProbabilitaFinanziamentoApprovato"] = probabilita

        try:
            richieste_collection.insert_one(document_to_save)
            imported_count += 1
        except Exception as e_db:
            print(f"!!! Errore INSERIMENTO DB per ID {id_api}: {e_db}")
            error_processing_count +=1

    summary_msg = (f"Importazione completata. "
                   f"Importati: {imported_count}. "
                   f"Skippati (già esistenti): {skipped_count}. "
                   f"Errori processamento/predizione/salvataggio: {error_processing_count}.")
    print(summary_msg)
    return jsonify({
        "message": summary_msg,
        "imported": imported_count,
        "skipped": skipped_count,
        "errors_processing": error_processing_count
    }), 200


# --- Endpoint per la predizione singola (per banca.html) ---
@app.route('/predict', methods=['POST'])
def predict_single_loan_route():
    print("\n--- Endpoint /predict chiamato ---")
    json_data_from_form = request.get_json()
    if not json_data_from_form:
        print("!!! /predict: Richiesta JSON vuota.")
        return jsonify({"error": "Richiesta JSON vuota"}), 400

    # Mappa i nomi del form HTML ai nomi "stile API" attesi da get_prediction_for_request_simplified
    data_api_format_from_form = {
        "eta_cliente": json_data_from_form.get("Eta"),
        "sesso_cliente": json_data_from_form.get("Sesso"),
        "titolo_studio_cliente": json_data_from_form.get("TitoloStudio"),
        "reddito_annuo_lordo": json_data_from_form.get("RedditoLordoUltimoAnno"),
        "anni_esperienza_lavorativa": json_data_from_form.get("AnniEsperienzaLavorativa"),
        "info_immobile": json_data_from_form.get("InformazioniImmobile"),
        "importo_richiesto_finanziamento": json_data_from_form.get("ImportoRichiesto"),
        "scopo_finanziamento": json_data_from_form.get("ScopoFinanziamento"),
        "tasso_interesse_proposto": json_data_from_form.get("TassoInteresseFinanziamento"),
        # "rapporto_importo_reddito" sarà calcolato sotto se non presente
        "durata_storia_creditizia_anni": json_data_from_form.get("DurataDellaStoriaCreditiziaInAnni"),
        "punteggio_affidabilita_creditizia": json_data_from_form.get("AffidabilitàCreditizia"),
        "inadempienze_pregresse": 1 if str(json_data_from_form.get("InadempienzeFinanziamentiPrecedenti","NO")).upper() == "SI" else 0
    }

    # Calcola 'rapporto_importo_reddito' se non è stato inviato (o se vuoi ricalcolarlo)
    try:
        reddito = float(data_api_format_from_form.get("reddito_annuo_lordo", 0))
        importo = float(data_api_format_from_form.get("importo_richiesto_finanziamento", 0))
        if reddito > 0:
            data_api_format_from_form["rapporto_importo_reddito"] = importo / reddito
        else:
            data_api_format_from_form["rapporto_importo_reddito"] = 999 # Placeholder
    except (TypeError, ValueError):
         data_api_format_from_form["rapporto_importo_reddito"] = 999 # Placeholder

    # print(f"/predict - Dati form mappati per predizione: {data_api_format_from_form}")
    probabilita = get_prediction_for_request_simplified("FROM_FORM", data_api_format_from_form.copy())

    print(f"/predict - Probabilità calcolata per dati form: {probabilita}")
    return jsonify({"probabilita_approvazione": probabilita})


if __name__ == '__main__':
    print("\n--- Avvio server Flask ---")
    print(f"URL per il form di test (se presente): http://127.0.0.1:5000/ (o il file .html diretto)")
    print(f"URL per importazione dati (da chiamare con POST da import_page.html): http://127.0.0.1:5000/import_data")
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False per output più pulito