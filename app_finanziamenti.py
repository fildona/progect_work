from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
from flask_cors import CORS
import logging

# --- Configurazione Nomi File Modello ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(MODEL_DIR, 'GradientBoosting_model.pkl')
SCALER_FILENAME = os.path.join(MODEL_DIR, 'loan_scaler.joblib')
MODEL_COLUMNS_FILENAME = os.path.join(MODEL_DIR, 'model_columns.joblib')
COLS_TO_SCALE_FILENAME = os.path.join(MODEL_DIR, 'cols_to_scale.joblib')
TRAINING_BOUNDS_FILENAME = os.path.join(MODEL_DIR, 'training_bounds.joblib') # Nuovo file

# --- Caricamento Modello, Scaler, Colonne e Limiti di Training ---
model = None
scaler = None
model_columns = None
cols_to_scale = None
training_bounds = None # Nuovo

try:
    model = joblib.load(MODEL_FILENAME)
    print(f"Modello caricato da {MODEL_FILENAME}")
except FileNotFoundError:
    print(f"ERRORE CRITICO: File del modello '{MODEL_FILENAME}' non trovato.")
except Exception as e:
    print(f"ERRORE CRITICO durante il caricamento del modello: {e}")

try:
    scaler = joblib.load(SCALER_FILENAME)
    print(f"Scaler caricato da {SCALER_FILENAME}")
except FileNotFoundError:
    print(f"ERRORE CRITICO: File dello scaler '{SCALER_FILENAME}' non trovato.")
except Exception as e:
    print(f"ERRORE CRITICO durante il caricamento dello scaler: {e}")

try:
    model_columns = joblib.load(MODEL_COLUMNS_FILENAME)
    print(f"Colonne del modello caricate da {MODEL_COLUMNS_FILENAME}")
except FileNotFoundError:
    print(f"ERRORE CRITICO: File delle colonne del modello '{MODEL_COLUMNS_FILENAME}' non trovato.")
except Exception as e:
    print(f"ERRORE CRITICO durante il caricamento delle colonne del modello: {e}")

try:
    cols_to_scale = joblib.load(COLS_TO_SCALE_FILENAME)
    print(f"Colonne da scalare caricate da {COLS_TO_SCALE_FILENAME}")
except FileNotFoundError:
    print(f"ERRORE CRITICO: File delle colonne da scalare '{COLS_TO_SCALE_FILENAME}' non trovato.")
except Exception as e:
    print(f"ERRORE CRITICO durante il caricamento delle colonne da scalare: {e}")

try:
    training_bounds = joblib.load(TRAINING_BOUNDS_FILENAME)
    print(f"Limiti del training set caricati da {TRAINING_BOUNDS_FILENAME}")
except FileNotFoundError:
    print(f"ATTENZIONE: File dei limiti del training set '{TRAINING_BOUNDS_FILENAME}' non trovato. Il clipping non sarà basato sui limiti esatti del training.")
except Exception as e:
    print(f"ERRORE durante il caricamento dei limiti del training set: {e}")


app = Flask(__name__)
CORS(app)

# Configura il logger di Flask per essere più verboso
app.logger.setLevel(logging.DEBUG)
if not app.logger.handlers: # Evita di aggiungere handler multipli se Flask si riavvia
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    stream_handler.setFormatter(formatter)
    app.logger.addHandler(stream_handler)


@app.route('/predict', methods=['POST'])
def predict_loan_approval():
    if not all([model, scaler, model_columns, cols_to_scale]): # training_bounds è opzionale per il funzionamento base
        missing_components = []
        if not model: missing_components.append("modello")
        if not scaler: missing_components.append("scaler")
        if not model_columns: missing_components.append("colonne del modello")
        if not cols_to_scale: missing_components.append("colonne da scalare")
        error_msg = f"Errore server: Componenti necessari non caricati ({', '.join(missing_components)})."
        app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

    try:
        json_data = request.get_json()
        if not json_data:
            app.logger.warning("Richiesta JSON vuota o malformata ricevuta.")
            return jsonify({"error": "Richiesta JSON vuota o malformata"}), 400

        app.logger.debug(f"Dati JSON ricevuti: {json_data}")

        # --- Calcolo di ImportoRichiestoDivisoReddito ---
        try:
            importo_richiesto = float(json_data.get("ImportoRichiesto", 0))
            reddito_lordo = float(json_data.get("RedditoLordoUltimoAnno", 0))
            if reddito_lordo > 0:
                rapporto = importo_richiesto / reddito_lordo
                json_data["ImportoRichiestoDivisoReddito"] = rapporto
                app.logger.debug(f"ImportoRichiestoDivisoReddito calcolato: {rapporto}")
            else:
                # Se il reddito è zero, il rapporto è problematico.
                # Potrebbe essere 0, un valore molto alto, o NaN da gestire dopo.
                # Il modello si aspetta un numero. Mettiamo un valore alto ma finito.
                # Il clipping successivo dovrebbe gestirlo se training_bounds è disponibile.
                json_data["ImportoRichiestoDivisoReddito"] = 999 # Placeholder per reddito zero, sarà clippato
                app.logger.warning(f"Reddito Lordo è 0 o non fornito. ImportoRichiestoDivisoReddito impostato a placeholder: 999")
        except (TypeError, ValueError) as e:
            app.logger.error(f"Errore nel calcolo di ImportoRichiestoDivisoReddito: {e}. Impostato a placeholder 999.", exc_info=True)
            json_data["ImportoRichiestoDivisoReddito"] = 999 # Placeholder in caso di errore


        # --- Clipping dei valori basato sui limiti del training set (se disponibili) ---
        if training_bounds:
            app.logger.debug("Applicazione clipping basato sui limiti del training set...")
            for col, limits in training_bounds.items():
                if col in json_data and pd.notna(json_data[col]):
                    try:
                        original_value = float(json_data[col])
                        col_min = limits.get('min')
                        col_max = limits.get('max')

                        if pd.notna(col_min) and pd.notna(col_max):
                            clipped_value = np.clip(original_value, col_min, col_max)
                            if original_value != clipped_value:
                                app.logger.info(f"Clipping colonna '{col}': Originale {original_value}, Clippato a {clipped_value} (Limiti: min={col_min}, max={col_max})")
                            json_data[col] = clipped_value
                        # Potresti aggiungere logica per clippare solo min o solo max se l'altro non è definito
                    except (ValueError, TypeError) as e:
                        app.logger.warning(f"Impossibile clippare la colonna '{col}' (valore: {json_data[col]}), errore: {e}. Lasciato invariato.")
        else:
            app.logger.info("Nessun clipping basato sui limiti del training set (training_bounds.joblib non caricato o vuoto).")
        
        app.logger.debug(f"JSON data dopo potenziale clipping: {json_data}")

        # Creazione DataFrame
        input_df = pd.DataFrame([json_data])
        app.logger.debug(f"Input DataFrame prima del preprocessing dettagliato: \n{input_df.to_string()}")

        # 1. Mappatura variabili categoriche ordinali e binarie
        mapping_titolo_studio = {"Diploma": 0, "Laurea": 1, "Dottorato di ricerca": 2}
        if "TitoloStudio" in input_df.columns:
            input_df["TitoloStudio"] = input_df["TitoloStudio"].map(mapping_titolo_studio)

        mapping_sesso = {"F": 0, "M": 1}
        if "Sesso" in input_df.columns:
            input_df["Sesso"] = input_df["Sesso"].map(mapping_sesso)

        mapping_inadempienze = {"NO": 0, "SI": 1}
        if "InadempienzeFinanziamentiPrecedenti" in input_df.columns:
            # Converti a stringa e poi a maiuscolo per gestire input come "no", "si"
            val_inad = str(json_data.get("InadempienzeFinanziamentiPrecedenti", "")).upper()
            input_df["InadempienzeFinanziamentiPrecedenti"] = mapping_inadempienze.get(val_inad) # Usa .get per evitare KeyError

        app.logger.debug(f"Input DataFrame dopo mappature: \n{input_df.to_string()}")

        # 2. One-Hot Encoding
        categorical_cols_ohe = []
        if "InformazioniImmobile" in input_df.columns: categorical_cols_ohe.append("InformazioniImmobile")
        if "ScopoFinanziamento" in input_df.columns: categorical_cols_ohe.append("ScopoFinanziamento")

        if categorical_cols_ohe:
            input_df = pd.get_dummies(input_df, columns=categorical_cols_ohe, dummy_na=False)

        app.logger.debug(f"Input DataFrame dopo OHE (colonne: {input_df.columns.tolist()}): \n{input_df.to_string()}")

        # 3. Allinea le colonne
        processed_df = pd.DataFrame(columns=model_columns)
        for col in model_columns:
            if col in input_df.columns:
                
                processed_df[col] = input_df[col]

        processed_df = processed_df.fillna(0) # Gestisce NaN da mappature fallite e colonne OHE mancanti
        app.logger.debug(f"Processed DataFrame dopo allineamento e fillna(0): \n{processed_df.to_string()}")

        # Converti tutte le colonne in numerico per sicurezza
        for col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
        app.logger.debug(f"Processed DataFrame dopo conversione finale a numerico: \n{processed_df.to_string()}")

        # 4. Normalizzazione
        actual_cols_to_scale_in_df = [col for col in cols_to_scale if col in processed_df.columns]
        if actual_cols_to_scale_in_df:
            app.logger.debug(f"Colonne da scalare: {actual_cols_to_scale_in_df}")
            app.logger.debug(f"Dati da scalare (prima) per colonne {actual_cols_to_scale_in_df}: \n{processed_df[actual_cols_to_scale_in_df].to_string()}")
            try:
                # Assicurati che le colonne siano float prima dello scaling
                processed_df[actual_cols_to_scale_in_df] = processed_df[actual_cols_to_scale_in_df].astype(float)
                scaled_values = scaler.transform(processed_df[actual_cols_to_scale_in_df])
                processed_df[actual_cols_to_scale_in_df] = scaled_values
                app.logger.debug(f"Dati scalati (dopo): \n{processed_df[actual_cols_to_scale_in_df].to_string()}")
            except ValueError as ve:
                app.logger.error(f"Errore durante lo scaling: {ve}. Shape dati: {processed_df[actual_cols_to_scale_in_df].shape}. Scaler n_features_in_: {getattr(scaler, 'n_features_in_', 'N/A')}", exc_info=True)
                return jsonify({"error": f"Errore di coerenza dati durante lo scaling: {ve}"}), 400
        else:
            app.logger.warning("Nessuna colonna da scalare trovata o specificata.")

        # --- PREDIZIONE ---
        final_input_for_model = processed_df[model_columns]
        app.logger.debug(f"Input finale per il modello (shape {final_input_for_model.shape}): \n{final_input_for_model.to_string()}")
        app.logger.debug(f"Colonne finali per il modello: {final_input_for_model.columns.tolist()}")

        try:
            probability_approved = model.predict_proba(final_input_for_model)[0, 1]
            app.logger.info(f"Probabilità calcolata: {probability_approved}")
        except ValueError as ve: # Es. per "Input X contains NaN" or "infinity"
            app.logger.error(f"Errore durante la predizione: {ve}. Shape dati: {final_input_for_model.shape}. Modello n_features_in_: {getattr(model, 'n_features_in_', 'N/A')}", exc_info=True)
            # Controlla se ci sono NaN o inf nel final_input_for_model
            if final_input_for_model.isnull().values.any():
                app.logger.error("NaN TROVATI nell'input finale per il modello!")
                app.logger.error(final_input_for_model[final_input_for_model.isnull().any(axis=1)])
            if np.isinf(final_input_for_model.to_numpy()).any():
                app.logger.error("VALORI INFINITI TROVATI nell'input finale per il modello!")
                app.logger.error(final_input_for_model[np.isinf(final_input_for_model.to_numpy()).any(axis=1)])
            return jsonify({"error": f"Errore di coerenza dati per la predizione (es. NaN/inf): {ve}"}), 400

        return jsonify({"probabilita_approvazione": float(probability_approved)})

    except KeyError as e:
        app.logger.warning(f"Campo mancante o errato nella richiesta JSON: {str(e)}", exc_info=True)
        return jsonify({"error": f"Campo mancante o errato nella richiesta JSON: {str(e)}"}), 400
    except ValueError as e:
        app.logger.error(f"Errore nel valore di un campo o durante la conversione: {str(e)}", exc_info=True)
        return jsonify({"error": f"Errore nel valore di un campo o durante la conversione: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Errore non gestito: {str(e)}", exc_info=True)
        return jsonify({"error": f"Errore interno del server."}), 500

if __name__ == '__main__':
    # Il logger di Flask è già configurato sopra
    app.run(host='0.0.0.0', port=5000, debug=True)