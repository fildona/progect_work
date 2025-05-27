# **Project Work Analisi delle richieste di Finanziamento**

Il file DataSetTestTraining44000.xlsx contiene i dati relativi al rischio finanziario per l'approvazione dei
finanziamenti di un istituto bancario (44.000 righe)
**Colonne del dataset**:
1. Eta
2. Sesso
3. TitoloStudio
4. RedditoLordoUltimoAnno
5. AnniEsperienzaLavorativa
6. InformazioniImmobile
7. ImportoRichiesto
8. ScopoFinanziamento
9. TassoInteresseFinanziamento
10. ImportoRichiestoDivisoReddito
11. DurataDellaStoriaCreditiziaInAnni
12. AffidabilitàCreditizia
13. InadempienzeFinanziamentiPrecedenti
14. FinanziamentoApprovato (target)

l'Affidabilità creditizia è una rappresentazione numerica dell'affidabilità creditizia di un individuo, che in
genere varia da **300 a 850**. Riflette la probabilità che una persona rimborsi puntualmente un finanziamento.
Punteggi più alti indicano generalmente una maggiore probabilità di rimborso e possono portare a
condizioni di finanziamento migliori.

## Parte 1
1. Caricamento Dati
2. Controllo e gestione dati mancanti, outlier, inconsistenze (normalmente i dati sono già tutti ok
perché provengono da database)
3. Divisione in set di training e set di valutazione
4. Analisi esplorativa (indici statistici vari)
5. Trovare correlazioni fra le variabili
6. Normalizzazione variabili
7. Scelta algoritmo e ricerca dei migliori parametri e salvataggio del modello su file
8. Valutazione modello

## Parte 2
1. Implementare una WebAPI con Flask che accetta una richiesta POST con i dati di una richiesta di
finanziamento in formato JSON e restituisce in formato JSON la probabilità che il finanziamento venga
approvato
2. Creare una Pagina web (HTML + Javascript) per inserire i dati della richiesta di finanziamento. Inserire
un button per inviare i dati all’endpoint Flask. Visualizzare la previsione (probabilità di finanziamento
approvato) nella pagina.

## Parte 3
Creare un database in locale o su server remoto con tecnologia a scelta (mongodb, elasticsearch, sql server)
Nel database deve essere possibile memorizzare dei documenti (o record) con le seguenti colonne:
1. RichiestaFinanziamentoID
2. Eta
3. Sesso
4. TitoloStudio
5. RedditoLordoUltimoAnno
6. AnniEsperienzaLavorativa
7. InformazioniImmobile
8. ImportoRichiesto
9. ScopoFinanziamento
10. TassoInteresseFinanziamento
11. ImportoRichiestoDivisoReddito
12. DurataDellaStoriaCreditiziaInAnni
13. AffidabilitàCreditizia
14. InadempienzeFinanziamentiPrecedenti
15. ProbabilitaFinanziamentoApprovato

Attenzione che da questo punto in poi è possibile implementare le funzionalità richieste in vario modo
incrociando script lato client e script lato server in vari modi.
Creare una pagina web con un button “importa” che premendolo permette di ottenere questo risultato:
1. I dati di 100 nuove richieste di finanziamento ottenuti da questa chiamata
https://testbobphp2.altervista.org/000AiForemaProjectWork/richieste_finanziamenti.php
vengono caricati sulla tabella del database creato: notare che la webapi non restituisce la
colonna “ProbabilitaFinanziamentoApprovato”
2. Per ogni richiesta di finanziamento importato nel database -> calcolare la
ProbabilitaFinanziamentoApprovato utilizzando il modello implementato nella Parte 1.
3. Attenzione che in caso di rilancio della procedura bisogna evitare di reimportare Richieste di
Finanziamento già importate (identificabili univocamente tramite RichiestaFinanziamentoID).
Inoltre nei 3 giorni del project work la chiamata potrebbe restituire le 100 richieste +altre
nuove.

## Parte 4
Creare una pagina web che fa visualizza la lista delle richieste di finanziamento importate con la probabilità
che venga approvato (colonna ProbabilitaFinanziamentoApprovato)
Nella pagina deve essere possibile fare le ricerche fra le varie richieste di finanziamento:
1. Eta (compresa fra min e max)
2. Sesso (select)
3. TitoloStudio (select)
4. RedditoLordoUltimoAnno (compreso fra min e max)
5. AnniEsperienzaLavorativa (compresa fra min e max)
6. InformazioniImmobile (select)
7. ImportoRichiesto (compreso fra min e max)
8. ScopoFinanziamento (select)
9. TassoInteresseFinanziamento (compreso fra min e max)
10. ImportoRichiestoDivisoReddito (compreso fra min e max)
11. DurataDellaStoriaCreditiziaInAnni (compreso fra min e max)
12. AffidabilitàCreditizia (compreso fra min e max)
13. InadempienzeFinanziamentiPrecedenti (select)
14. ProbabilitaFinanziamentoApprovato (compreso fra min e max)

Deve essere possibile esportare in formato **csv/Excel/json** il risultato della ricerca

## Parte 5
Creare una pagina html che permette di effettuare delle statistiche grafiche sui dati presenti sul database
(indipendentemente dalla ProbabilitaFinanziamentoApprovato)
Creare 4 grafici:
1. Grafico a torta con il conteggio delle richieste di finanziamento per Sesso
2. Grafico a barre verticali che visualizza la somma degli ImportiRichiesti per InformazioniImmobile
3. Grafico a barre orizzontali che visualizza la somma degli ImportiRichiesti dei finanziamenti per
TitoloStudio
4. Grafico a scelta che visualizza con il conteggio dei finanziamenti per ScopoFinanziamento