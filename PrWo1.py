# PrWo1.py

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler # Note: This was imported but MinMaxScaler used later
import joblib

# 1. Caricamento Dati
# 2. Controllo e gestione dati mancanti, outlier, inconsistenze (normalmente i dati sono già tutti ok
# perché provengono da database)
# 3. Divisione in set di training e set di valutazione
# 4. Analisi esplorativa (indici statistici vari)
# 5. Trovare correlazioni fra le variabili
# 6. Normalizzazione variabili
# 7. Scelta algoritmo e ricerca dei migliori parametri e salvataggio del modello su file
# 8. Valutazione modello

# 1 caricamento dati
# df= pd.read_excel("C:\\ex\\ProjectWork\\DataSetTestTraining44000.xlsx")
# Assuming the CSV file is in the same directory as this script or you provide the correct path.
# For this conversion, I'll assume the CSV 'DataSetTestTraining44000.csv' will be used.
# Since the original notebook converted it to CSV and then likely used it,
# let's assume the CSV is available in the working directory.
# If the original .xlsx is preferred, ensure it's accessible and uncomment the line above,
# and comment out the .read_csv line.
try:
    df = pd.read_csv("DataSetTestTraining44000.csv")
except FileNotFoundError:
    print("DataSetTestTraining44000.csv not found. Please ensure the file is in the correct path.")
    print("Attempting to read DataSetTestTraining44000.xlsx instead...")
    try:
        df_excel = pd.read_excel("DataSetTestTraining44000.xlsx")
        df_excel.to_csv("DataSetTestTraining44000.csv", index=False)
        df = pd.read_csv("DataSetTestTraining44000.csv")
        print("Successfully read .xlsx and converted to .csv")
    except FileNotFoundError:
        print("DataSetTestTraining44000.xlsx also not found. Please provide the data file.")
        exit()


# Displaying the dataframe (optional, for script execution you might want to print head or shape)
# In a script, simply calling df won't output it.
print("DataFrame head:")
print(df.head())
# print(df) # This would print the entire dataframe, might be too much for a script output

# 2. Controllo e gestione dati mancanti, outlier, inconsistenze (normalmente i dati sono già tutti ok perché provengono da database)
print("\nDataFrame info:")
df.info()

print("\nDataFrame shape:")
print(df.shape)

print("\nUnique values in 'ScopoFinanziamento':")
print(df["ScopoFinanziamento"].unique())

# outliers
numeric_df = df.select_dtypes(include=np.number) # Changed 'number' to np.number for broader compatibility
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

outliers = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
df_outliers = df[outliers.any(axis=1)]
print("\nNumber of rows with outliers:", len(df_outliers))
# print("Outliers detected:") # This original print was vague, showing count is more informative

# non ci sono valori mancanti crazy follia
print(f"\nValori mancanti per colonna:")
print(df.isnull().sum())

df = pd.get_dummies(df, columns=['InformazioniImmobile', 'ScopoFinanziamento'])
print("\nDataFrame head after get_dummies:")
print(df.head())
# print(df)

print("\nUnique values in 'TitoloStudio':")
print(df["TitoloStudio"].unique())
print("\nUnique values in 'Sesso':")
print(df["Sesso"].unique())
print("\nUnique values in 'InadempienzeFinanziamentiPrecedenti':")
print(df["InadempienzeFinanziamentiPrecedenti"].unique())
print("\nUnique values in 'FinanziamentoApprovato':")
print(df["FinanziamentoApprovato"].unique())

# mappature variabili cateporiche
mapping = {
    "Diploma": 0,
    "Laurea": 1,
    "Dottorato di ricerca": 2
}

# mappatura titolo_studio_cod
df["TitoloStudio"] = df["TitoloStudio"].map(mapping)

mapping_sesso = {
    "F": 0,
    "M": 1,
}

# mappatura Sesso
df["Sesso"] = df["Sesso"].map(mapping_sesso)

mapping_Nevio = {
    "NO": 0,
    "SI": 1,
}

# mappatura inadempienza
df["InadempienzeFinanziamentiPrecedenti"] = df["InadempienzeFinanziamentiPrecedenti"].map(mapping_Nevio)

mapping_approved = {
    "NO": 0,
    "SI": 1,
}

# mappatura label target
df["FinanziamentoApprovato"] = df["FinanziamentoApprovato"].map(mapping_approved)

print("\nDataFrame info after mapping categorical variables:")
df.info()

# 3. Divisione in set di training e set di valutazione
# dopo

# 4. Analisi esplorativa (indici statistici vari)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot(df['Eta'])
plt.title('Boxplot Età')

plt.subplot(1, 2, 2)
plt.boxplot(df['TitoloStudio'])
plt.title('Boxplot TitoloStudio')

plt.tight_layout()
# plt.show() # In a script, plt.show() is needed to display plots.
            # For automated runs, you might save figures instead.

# 5. Trovare correlazioni fra le variabili
col = "FinanziamentoApprovato"

# Rimuovi la colonna e aggiungila in fondo
df_corr = df[[c for c in df.columns if c != col] + [col]] # Use a copy for correlation
correlation_matrix = df_corr.corr()
# print(correlation_matrix) # Printing matrix can be large
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix") # Added title for clarity
# plt.show() # Show or save figure

print("\nDataFrame columns before splitting:")
print(df.columns)

# punto 3 divisione in train e test
X = df.drop(columns=['FinanziamentoApprovato'])
y = df['FinanziamentoApprovato']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDimensioni X_train: {X_train.shape}")
print(f"Dimensioni X_test: {X_test.shape}")
print(f"Dimensioni y_train: {y_train.shape}")
print(f"Dimensioni y_test: {y_test.shape}")


# 6. Normalizzazione variabili e Analisi Range Training Set
from sklearn.preprocessing import MinMaxScaler
# import numpy as np # Already imported
# import joblib # Already imported

cols_to_scale = [
    'Eta', 'RedditoLordoUltimoAnno', 'AnniEsperienzaLavorativa',
    'ImportoRichiesto', 'TassoInteresseFinanziamento',
    'ImportoRichiestoDivisoReddito', 'DurataDellaStoriaCreditiziaInAnni',
    'AffidabilitàCreditizia'
]

cols_to_scale = [col for col in cols_to_scale if col in X_train.columns]
if len(cols_to_scale) != 8:
     print(f"ATTENZIONE: Trovate {len(cols_to_scale)} colonne da scalare. Attese 8. Colonne: {cols_to_scale}")

print("\n--- Statistiche Descrittive delle Colonne da Scalare (X_train PRIMA dello scaling) ---")
try:
    training_set_description = X_train[cols_to_scale].astype(float).describe()
    print(training_set_description)

    training_bounds = {}
    for col_name in cols_to_scale: # Changed variable name to avoid conflict
        if col_name in training_set_description.columns:
            training_bounds[col_name] = {
                'min': training_set_description.loc['min', col_name],
                'max': training_set_description.loc['max', col_name]
            }
        else:
            print(f"ATTENZIONE: Colonna '{col_name}' non trovata in training_set_description.columns.")

    print("\n--- Limiti Min/Max del Training Set (per cols_to_scale) ---")
    for col_name, bounds in training_bounds.items(): # Changed variable name
        min_val_str = f"{bounds.get('min', 'N/A'):.4f}" if pd.notna(bounds.get('min')) else "N/A"
        max_val_str = f"{bounds.get('max', 'N/A'):.4f}" if pd.notna(bounds.get('max')) else "N/A"
        print(f"Colonna '{col_name}': Min = {min_val_str}, Max = {max_val_str}")

    training_bounds_filename = 'training_bounds.joblib'
    joblib.dump(training_bounds, training_bounds_filename)
    print(f"\nLimiti del training set salvati in: {training_bounds_filename}")

except Exception as e:
    print(f"Errore durante il describe() o il salvataggio dei training_bounds: {e}")

scaler = MinMaxScaler()

X_train_scaled_cols_np = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled_cols_np = scaler.transform(X_test[cols_to_scale]) # Renamed to _np as it's an array

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[cols_to_scale] = X_train_scaled_cols_np # Used the _np array
X_test_scaled[cols_to_scale] = X_test_scaled_cols_np   # Used the _np array

print("\nPrime 5 righe di X_train normalizzato (solo colonne scalate):")
print(X_train_scaled[cols_to_scale].head())

print("\nPrime 5 righe di X_test normalizzato (solo colonne scalate):")
print(X_test_scaled[cols_to_scale].head())

print("\n--- Statistiche Descrittive delle Colonne Scalate (X_train_scaled) ---")
print(X_train_scaled[cols_to_scale].describe())

print("\n--- Statistiche Descrittive delle Colonne Scalate (X_test_scaled) ---")
print(X_test_scaled[cols_to_scale].describe())

model_columns = X_train_scaled.columns.tolist() # This was defined later, moved here for consistency.
# print("\nColonne del modello:", model_columns) # Already printed above effectively

# Creare copie per evitare SettingWithCopyWarning
# This block was redundant with lines 200-201. X_train_scaled_cols was already an array.
# X_train_scaled = X_train.copy()
# X_test_scaled = X_test.copy()
#
# X_train_scaled[cols_to_scale] = X_train_scaled_cols_np
# X_test_scaled[cols_to_scale] = X_test_scaled_cols_np

print("\nPrime 5 righe di X_train normalizzato (dopo assegnazione array, solo colonne scalate):") # Clarified print
print(X_train_scaled[cols_to_scale].head())

# model_columns = X_train_scaled.columns.tolist() # Already defined
print("\nColonne del modello (conferma):", model_columns)


# ricerca modello
models_params = {
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    },
}
best_models = {}
best_scores = {}

print("\nInizio ricerca dei migliori parametri per ogni modello...\n")

for name, mp in models_params.items():
    print(f"➡️ Modello: {name}")
    grid = GridSearchCV(estimator=mp['model'],
                        param_grid=mp['params'],
                        scoring='roc_auc',
                        cv=3,
                        n_jobs=-1,
                        verbose=1)
    grid.fit(X_train_scaled, y_train)
    best_models[name] = grid.best_estimator_
    best_scores[name] = grid.best_score_

    print(f" Migliori parametri per {name}: {grid.best_params_}")
    print(f" Miglior ROC AUC (CV): {grid.best_score_:.4f}\\n")

best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]

print("Miglior modello complessivo:", best_model_name)
print("ROC AUC CV:", best_scores[best_model_name])

joblib.dump(best_model, f'{best_model_name}_model.pkl')
print(f"Modello salvato in: {best_model_name}_model.pkl")


# Salva lo scaler
scaler_filename = 'loan_scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"\nScaler salvato in: {scaler_filename}")

# Salva la lista delle colonne del modello
columns_filename = 'model_columns.joblib'
joblib.dump(model_columns, columns_filename)
print(f"Lista colonne del modello salvata in: {columns_filename}")

# Salva la lista delle colonne che sono state scalate
cols_to_scale_filename = 'cols_to_scale.joblib'
joblib.dump(cols_to_scale, cols_to_scale_filename)
print(f"Lista colonne scalate salvata in: {cols_to_scale_filename}")

# To display plots at the end if any were generated and not shown inline
plt.show()