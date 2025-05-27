// grafici cliccabili e che filtrano i dati
let activeFilters = {};

function buildQueryString(filters) {
    return Object.entries(filters)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
        .join('&');
}

async function loadCharts() {
    const qs = buildQueryString(activeFilters);
    const res = await fetch('/api/statistiche' + (qs ? '?' + qs : ''));
    const data = await res.json();

    // KPI CARDS
    const kpiDiv = document.getElementById('kpiCards');
    if (kpiDiv) {
        kpiDiv.innerHTML = `
            <div class="col">
                <div class="card border-primary text-center">
                    <div class="card-body">
                        <div class="h2">${data.totale_richieste}</div>
                        <div class="small text-muted">Totale Richieste</div>
                    </div>
                </div>
            </div>
            <div class="col">
                <div class="card border-success text-center">
                    <div class="card-body">
                        <div class="h2">€ ${data.importo_totale.toLocaleString('it-IT', {maximumFractionDigits:0})}</div>
                        <div class="small text-muted">Importo Totale Richiesto</div>
                    </div>
                </div>
            </div>
            <div class="col">
                <div class="card border-info text-center">
                    <div class="card-body">
                        <div class="h2">${data.percentuale_approvate.toFixed(1)}%</div>
                        <div class="small text-muted">Percentuale Approvazione</div>
                    </div>
                </div>
            </div>
            <div class="col">
                <div class="card border-warning text-center">
                    <div class="card-body">
                        <div class="h2">€ ${data.importo_medio.toLocaleString('it-IT', {maximumFractionDigits:0})}</div>
                        <div class="small text-muted">Importo Medio Richiesto</div>
                    </div>
                </div>
            </div>
        `;
    }

    // Distruggi i grafici precedenti se esistono
    if (window.charts) window.charts.forEach(c => c.destroy());
    window.charts = [];

    // 1. Sesso
    const chartSesso = new Chart(document.getElementById('chartSesso'), {
        type: 'pie',
        data: {
            labels: Object.keys(data.sesso_counts),
            datasets: [{
                data: Object.values(data.sesso_counts),
                backgroundColor: ['#36A2EB', '#FF6384', '#FFCE56', '#8BC34A']
            }]
        },
        options: {
            onClick: (evt, elements) => {
                if (elements.length > 0) {
                    const label = chartSesso.data.labels[elements[0].index];
                    activeFilters = { ...activeFilters, Sesso: label };
                    loadCharts();
                    showFilters();
                }
            }
        }
    });
    window.charts.push(chartSesso);

    // 2. Immobile
    const chartImmobile = new Chart(document.getElementById('chartImmobile'), {
        type: 'bar',
        data: {
            labels: Object.keys(data.immobile_importi),
            datasets: [{
                label: 'Somma Importi Richiesti',
                data: Object.values(data.immobile_importi),
                backgroundColor: '#36A2EB'
            }]
        },
        options: {
            indexAxis: 'x',
            plugins: { legend: { display: false } },
            onClick: (evt, elements) => {
                if (elements.length > 0) {
                    const label = chartImmobile.data.labels[elements[0].index];
                    activeFilters = { ...activeFilters, InformazioniImmobile: label };
                    loadCharts();
                    showFilters();
                }
            }
        }
    });
    window.charts.push(chartImmobile);

    // 3. TitoloStudio
    const chartTitolo = new Chart(document.getElementById('chartTitolo'), {
        type: 'bar',
        data: {
            labels: Object.keys(data.titolo_importi),
            datasets: [{
                label: 'Somma Importi Richiesti',
                data: Object.values(data.titolo_importi),
                backgroundColor: '#FF6384'
            }]
        },
        options: {
            indexAxis: 'y',
            plugins: { legend: { display: false } },
            onClick: (evt, elements) => {
                if (elements.length > 0) {
                    const label = chartTitolo.data.labels[elements[0].index];
                    activeFilters = { ...activeFilters, TitoloStudio: label };
                    loadCharts();
                    showFilters();
                }
            }
        }
    });
    window.charts.push(chartTitolo);

    // 4. ScopoFinanziamento
    const chartScopo = new Chart(document.getElementById('chartScopo'), {
        type: 'bar',
        data: {
            labels: Object.keys(data.scopo_counts),
            datasets: [{
                label: 'Conteggio',
                data: Object.values(data.scopo_counts),
                backgroundColor: '#FFCE56'
            }]
        },
        options: {
            indexAxis: 'x',
            plugins: { legend: { display: false } },
            onClick: (evt, elements) => {
                if (elements.length > 0) {
                    const label = chartScopo.data.labels[elements[0].index];
                    activeFilters = { ...activeFilters, ScopoFinanziamento: label };
                    loadCharts();
                    showFilters();
                }
            }
        }
    });
    window.charts.push(chartScopo);

    // Importo medio richiesto/approvato per sesso
    const sessoLabels = Object.keys(data.importo_medio_sesso);
    const richiestoSesso = sessoLabels.map(k => data.importo_medio_sesso[k]);
    const approvatoSesso = sessoLabels.map(k => data.importo_medio_approvato_sesso[k] || 0);
    const chartMedioSesso = new Chart(document.getElementById('chartMedioSesso'), {
        type: 'bar',
        data: {
            labels: sessoLabels,
            datasets: [
                { label: 'Richiesto', data: richiestoSesso, backgroundColor: '#36A2EB' },
                { label: 'Approvato', data: approvatoSesso, backgroundColor: '#8BC34A' }
            ]
        },
        options: { plugins: { legend: { display: true } } }
    });
    window.charts.push(chartMedioSesso);

    // Importo medio richiesto/approvato per titolo studio
    const titoloLabels = Object.keys(data.importo_medio_titolo);
    const richiestoTitolo = titoloLabels.map(k => data.importo_medio_titolo[k]);
    const approvatoTitolo = titoloLabels.map(k => data.importo_medio_approvato_titolo[k] || 0);
    const chartMedioTitolo = new Chart(document.getElementById('chartMedioTitolo'), {
        type: 'bar',
        data: {
            labels: titoloLabels,
            datasets: [
                { label: 'Richiesto', data: richiestoTitolo, backgroundColor: '#FF6384' },
                { label: 'Approvato', data: approvatoTitolo, backgroundColor: '#FFCE56' }
            ]
        },
        options: { plugins: { legend: { display: true } }, indexAxis: 'y' }
    });
    window.charts.push(chartMedioTitolo);

    // Top 10 tabella
    const tbody = document.querySelector('#top10Table tbody');
    if (tbody) {
        tbody.innerHTML = '';
        data.top10_importi.forEach(r => {
            tbody.innerHTML += `
                <tr>
                    <td>${r.RichiestaFinanziamentoID}</td>
                    <td>€ ${r.ImportoRichiesto.toLocaleString('it-IT', {maximumFractionDigits:0})}</td>
                    <td>${r.Sesso}</td>
                    <td>${r.TitoloStudio}</td>
                    <td>${(r.ProbabilitaFinanziamentoApprovato*100).toFixed(1)}%</td>
                </tr>
            `;
        });
    }
}

function showFilters() {
    const div = document.getElementById('activeFilters');
    div.innerHTML = '';
    const keys = Object.keys(activeFilters);
    if (keys.length === 0) return;
    div.innerHTML = '<b>Filtri attivi:</b> ' + keys.map(k => 
        `<span class="badge bg-info text-dark me-2">${k}: ${activeFilters[k]}</span>`
    ).join('') + 
    ' <button class="btn btn-sm btn-outline-secondary ms-2" id="resetFilters">Reset filtri</button>';
    document.getElementById('resetFilters').onclick = () => {
        activeFilters = {};
        loadCharts();
        showFilters();
    };
}

window.addEventListener('DOMContentLoaded', () => {
    // Aggiungi un div per i filtri attivi sopra i grafici
    const container = document.querySelector('.card-body');
    const filterDiv = document.createElement('div');
    filterDiv.id = 'activeFilters';
    filterDiv.className = 'mb-3';
    container.prepend(filterDiv);

    loadCharts();
});