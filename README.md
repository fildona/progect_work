# SaaS Finance® - Sistema di Previsione per Richieste di Finanziamento

## 📋 Descrizione del Progetto

SaaS Finance® è un'applicazione web che utilizza machine learning per prevedere l'approvazione di richieste di finanziamento. Il sistema permette di inserire i dati di una nuova richiesta, visualizzare la probabilità di approvazione, importare richieste da fonti esterne, e analizzare statistiche dettagliate.

## ✨ Funzionalità Principali

- **Previsione di Approvazione**: Inserisci i dettagli di una richiesta e ricevi immediatamente una previsione sulla probabilità di approvazione.
- **Gestione Richieste**: Visualizza, filtra e ordina tutte le richieste di finanziamento.
- **Dashboard Statistiche**: Analizza i dati attraverso grafici interattivi e indicatori di performance chiave (KPI).
- **Importazione Dati**: Importa richieste di finanziamento da sistemi esterni.
- **Esportazione Dati**: Esporta i risultati in formato CSV, Excel o JSON.

## 💻 Tecnologie Utilizzate

- **Backend**: Python con Flask
- **Database**: SQLite con SQLAlchemy
- **Frontend**: HTML, CSS, Bootstrap 5
- **Visualizzazioni**: Chart.js
- **Machine Learning**: Modello pre-addestrato con scikit-learn

## 🔧 Setup dell'ambiente

1. Assicurati di avere [Miniconda](https://www.anaconda.com/download/success) installato.
2. Apri un nuovo terminale dalla cartella del progetto.
3. Crea l'ambiente con:

```bash
conda env create -f environment.yml
```
Attiva l'ambiente:
```bash
conda activate its_test1
```
Avvia il progetto:
```bash
python app.py
```
### Per chi preferisce pip:
```bash
pip install -r requirements.txt
```

## 🚀 Utilizzo dell'Applicazione

1. **Pagina Principale**: Compila il formulario con i dettagli della richiesta di finanziamento per ottenere una previsione istantanea.
2. **Richieste**: Visualizza tutte le richieste esistenti con possibilità di filtrarle e ordinarle per diversi parametri.
3. **Statistiche**: Esplora la dashboard interattiva con grafici e KPI per analizzare i trend delle richieste.
4. **Importa**: Importa nuove richieste dal sistema esterno con un semplice click.

## 🏗️ Struttura del Progetto

- static: File CSS, JavaScript e altri asset statici
- templates: Template HTML per le diverse pagine dell'applicazione
- model_scaler: Modello di machine learning e scaler preaddestrati
- db: Configurazione del database e definizione dei modelli
- app.py: File principale dell'applicazione Flask

## 📊 Modello di Previsione

L'applicazione utilizza un modello di machine learning preaddestrato per prevedere la probabilità di approvazione di una richiesta di finanziamento basandosi su:
- Dati demografici (età, sesso)
- Situazione economica (reddito, esperienza lavorativa)
- Dettagli del finanziamento (importo, scopo, tasso d'interesse)
- Storia creditizia (affidabilità, inadempienze precedenti)