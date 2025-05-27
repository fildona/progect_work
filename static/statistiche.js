// grafici cliccabili e che filtrano i dati
let activeFilters = {};
let charts = []; // Store chart instances to destroy them later

// Chart.js Global Defaults for a more modern look
Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
Chart.defaults.plugins.legend.position = 'bottom';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.7)';
Chart.defaults.plugins.tooltip.titleFont = { weight: 'bold' };
Chart.defaults.plugins.tooltip.bodyFont = { size: 14 };
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 4;

const CHART_COLORS = {
    blue: 'rgb(54, 162, 235)',
    red: 'rgb(255, 99, 132)',
    orange: 'rgb(255, 159, 64)',
    yellow: 'rgb(255, 205, 86)',
    green: 'rgb(75, 192, 192)',
    purple: 'rgb(153, 102, 255)',
    grey: 'rgb(201, 203, 207)'
};

const CHART_COLORS_ALPHA = {
    blue: 'rgba(54, 162, 235, 0.7)',
    red: 'rgba(255, 99, 132, 0.7)',
    orange: 'rgba(255, 159, 64, 0.7)',
    yellow: 'rgba(255, 205, 86, 0.7)',
    green: 'rgba(75, 192, 192, 0.7)',
    purple: 'rgba(153, 102, 255, 0.7)',
    grey: 'rgba(201, 203, 207, 0.7)'
};

function buildQueryString(filters) {
    return Object.entries(filters)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
        .join('&');
}

function showLoadingOverlay(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = show ? 'flex' : 'none';
}

async function loadChartsAndData() {
    showLoadingOverlay(true);
    const qs = buildQueryString(activeFilters);
    try {
        const res = await fetch('/api/statistiche' + (qs ? '?' + qs : ''));
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({ message: 'Errore sconosciuto' }));
            throw new Error(errorData.message || `Errore HTTP: ${res.status}`);
        }
        const data = await res.json();

        renderKpiCards(data);
        renderCharts(data);
        renderTop10Table(data.top10_importi);
        displayActiveFilters();

    } catch (error) {
        console.error("Errore caricamento statistiche:", error);
        const kpiDiv = document.getElementById('kpiCardsRow');
        if (kpiDiv) kpiDiv.innerHTML = `<div class="col-12"><div class="alert alert-danger">Errore nel caricamento delle statistiche: ${error.message}</div></div>`;
        // Clear charts and table on error too
        if (charts) charts.forEach(c => c.destroy());
        charts = [];
        const tbody = document.querySelector('#top10Table tbody');
        if (tbody) tbody.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Dati non disponibili.</td></tr>';
    } finally {
        showLoadingOverlay(false);
    }
}

function renderKpiCards(data) {
    const kpiDiv = document.getElementById('kpiCardsRow');
    if (!kpiDiv) return;

    const formatCurrency = (value) => `€ ${value.toLocaleString('it-IT', {maximumFractionDigits:0})}`;

    const kpis = [
        { title: 'Totale Richieste', value: data.totale_richieste.toLocaleString('it-IT'), icon: 'fas fa-file-invoice', color: 'primary' },
        { title: 'Importo Totale Richiesto', value: formatCurrency(data.importo_totale), icon: 'fas fa-coins', color: 'success' },
        { title: 'Percentuale Approvazione', value: `${data.percentuale_approvate.toFixed(1)}%`, icon: 'fas fa-percentage', color: 'info' },
        { title: 'Importo Medio Richiesto', value: formatCurrency(data.importo_medio), icon: 'fas fa-calculator', color: 'warning' }
    ];

    kpiDiv.innerHTML = kpis.map(kpi => `
        <div class="col">
            <div class="card kpi-card shadow-sm border-start border-5 border-${kpi.color} h-100">
                <div class="card-body text-center">
                    <div class="display-4 mb-2"><i class="${kpi.icon} text-${kpi.color}"></i></div>
                    <h3 class="h2 fw-bold mb-1">${kpi.value}</h3>
                    <p class="text-muted mb-0">${kpi.title}</p>
                </div>
            </div>
        </div>
    `).join('');
}

function handleChartClick(chartInstance, elements, filterKey) {
    if (elements.length > 0) {
        const label = chartInstance.data.labels[elements[0].index];
        if (activeFilters[filterKey] === label) {
            delete activeFilters[filterKey]; // Toggle off if already selected
        } else {
            activeFilters[filterKey] = label;
        }
        loadChartsAndData();
    }
}

function renderCharts(data) {
    if (charts) charts.forEach(c => c.destroy());
    charts = [];

    const chartOptionsTemplate = (filterKey) => ({
        responsive: true,
        maintainAspectRatio: false,
        onClick: (evt, elements) => handleChartClick(evt.chart, elements, filterKey),
        plugins: {
            legend: {
                labels: { font: { size: 12 } }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        let label = context.dataset.label || '';
                        if (label) label += ': ';
                        if (context.parsed.y !== null) {
                            if (context.chart.config.type === 'pie' || context.chart.config.type === 'doughnut') {
                                label += context.parsed.toLocaleString('it-IT');
                            } else {
                                label += parseFloat(context.parsed.y).toLocaleString('it-IT', { style: 'currency', currency: 'EUR', minimumFractionDigits: 0});
                            }
                        }
                        return label;
                    }
                }
            }
        }
    });

    // Sesso (Pie Chart)
    if(document.getElementById('chartSesso')) {
        const chartSesso = new Chart(document.getElementById('chartSesso'), {
            type: 'pie',
            data: {
                labels: Object.keys(data.sesso_counts),
                datasets: [{
                    data: Object.values(data.sesso_counts),
                    backgroundColor: [CHART_COLORS_ALPHA.blue, CHART_COLORS_ALPHA.red, CHART_COLORS_ALPHA.yellow, CHART_COLORS_ALPHA.green],
                    borderColor: [CHART_COLORS.blue, CHART_COLORS.red, CHART_COLORS.yellow, CHART_COLORS.green],
                    borderWidth: 1
                }]
            },
            options: chartOptionsTemplate('Sesso')
        });
        charts.push(chartSesso);
    }

    // Immobile (Bar Chart - Vertical)
    if(document.getElementById('chartImmobile')){
        const chartImmobile = new Chart(document.getElementById('chartImmobile'), {
            type: 'bar',
            data: {
                labels: Object.keys(data.immobile_importi),
                datasets: [{
                    label: 'Somma Importi (€)',
                    data: Object.values(data.immobile_importi),
                    backgroundColor: CHART_COLORS_ALPHA.green,
                    borderColor: CHART_COLORS.green,
                    borderWidth: 1
                }]
            },
            options: {
                ...chartOptionsTemplate('InformazioniImmobile'),
                indexAxis: 'x',
                scales: { y: { beginAtZero: true, ticks: { callback: value => '€ ' + value.toLocaleString('it-IT') } } }
            }
        });
        charts.push(chartImmobile);
    }

    // TitoloStudio (Bar Chart - Horizontal)
    if(document.getElementById('chartTitolo')){
        const chartTitolo = new Chart(document.getElementById('chartTitolo'), {
            type: 'bar',
            data: {
                labels: Object.keys(data.titolo_importi),
                datasets: [{
                    label: 'Somma Importi (€)',
                    data: Object.values(data.titolo_importi),
                    backgroundColor: CHART_COLORS_ALPHA.orange,
                    borderColor: CHART_COLORS.orange,
                    borderWidth: 1
                }]
            },
            options: {
                ...chartOptionsTemplate('TitoloStudio'),
                indexAxis: 'y',
                scales: { x: { beginAtZero: true, ticks: { callback: value => '€ ' + value.toLocaleString('it-IT') } } }
            }
        });
        charts.push(chartTitolo);
    }

    // ScopoFinanziamento (Bar Chart - Vertical)
    if(document.getElementById('chartScopo')){
        const chartScopo = new Chart(document.getElementById('chartScopo'), {
            type: 'bar',
            data: {
                labels: Object.keys(data.scopo_counts),
                datasets: [{
                    label: 'Numero Richieste',
                    data: Object.values(data.scopo_counts),
                    backgroundColor: CHART_COLORS_ALPHA.purple,
                    borderColor: CHART_COLORS.purple,
                    borderWidth: 1
                }]
            },
            options: {
                ...chartOptionsTemplate('ScopoFinanziamento'),
                 scales: { y: { beginAtZero: true, ticks: { callback: value => Number.isInteger(value) ? value : null } } }
            }
        });
        charts.push(chartScopo);
    }

    // Importo medio richiesto/approvato per sesso (Grouped Bar)
    if(document.getElementById('chartMedioSesso')){
        const sessoLabels = Object.keys(data.importo_medio_sesso);
        const chartMedioSesso = new Chart(document.getElementById('chartMedioSesso'), {
            type: 'bar',
            data: {
                labels: sessoLabels,
                datasets: [
                    { label: 'Importo Medio Richiesto (€)', data: sessoLabels.map(k => data.importo_medio_sesso[k]), backgroundColor: CHART_COLORS_ALPHA.blue, borderColor: CHART_COLORS.blue, borderWidth: 1 },
                    { label: 'Importo Medio Approvato (€)', data: sessoLabels.map(k => data.importo_medio_approvato_sesso[k] || 0), backgroundColor: CHART_COLORS_ALPHA.green, borderColor: CHART_COLORS.green, borderWidth: 1 }
                ]
            },
            options: { ...chartOptionsTemplate(), scales: { y: { beginAtZero: true, ticks: { callback: value => '€ ' + value.toLocaleString('it-IT') } } } }
        });
        charts.push(chartMedioSesso);
    }

    // Importo medio richiesto/approvato per titolo studio (Grouped Bar - Horizontal)
    if(document.getElementById('chartMedioTitolo')){
        const titoloLabels = Object.keys(data.importo_medio_titolo);
        const chartMedioTitolo = new Chart(document.getElementById('chartMedioTitolo'), {
            type: 'bar',
            data: {
                labels: titoloLabels,
                datasets: [
                    { label: 'Importo Medio Richiesto (€)', data: titoloLabels.map(k => data.importo_medio_titolo[k]), backgroundColor: CHART_COLORS_ALPHA.red, borderColor: CHART_COLORS.red, borderWidth: 1 },
                    { label: 'Importo Medio Approvato (€)', data: titoloLabels.map(k => data.importo_medio_approvato_titolo[k] || 0), backgroundColor: CHART_COLORS_ALPHA.yellow, borderColor: CHART_COLORS.yellow, borderWidth: 1 }
                ]
            },
            options: { ...chartOptionsTemplate(), indexAxis: 'y', scales: { x: { beginAtZero: true, ticks: { callback: value => '€ ' + value.toLocaleString('it-IT') } } } }
        });
        charts.push(chartMedioTitolo);
    }
}

function renderTop10Table(top10Data) {
    const tbody = document.querySelector('#top10Table tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    if (!top10Data || top10Data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center">Nessun dato disponibile.</td></tr>';
        return;
    }
    top10Data.forEach(r => {
        const probApprovazione = (r.ProbabilitaFinanziamentoApprovato * 100).toFixed(1);
        const esito = r.ProbabilitaFinanziamentoApprovato > 0.5 
            ? '<span class="badge bg-success-subtle text-success-emphasis rounded-pill">Approvato</span>' 
            : '<span class="badge bg-danger-subtle text-danger-emphasis rounded-pill">Rifiutato</span>';
        tbody.innerHTML += `
            <tr>
                <td>${r.RichiestaFinanziamentoID}</td>
                <td class="text-end">${r.ImportoRichiesto.toLocaleString('it-IT', { style: 'currency', currency: 'EUR', minimumFractionDigits: 0 })}</td>
                <td class="text-center">${r.Sesso}</td>
                <td>${r.TitoloStudio}</td>
                <td class="text-center">${probApprovazione}%</td>
                <td class="text-center">${esito}</td>
            </tr>
        `;
    });
}

function displayActiveFilters() {
    const container = document.getElementById('activeFiltersContainer');
    const listDiv = document.getElementById('activeFiltersList');
    const resetButton = document.getElementById('resetFiltersBtn');

    if (!container || !listDiv || !resetButton) return;

    const filterEntries = Object.entries(activeFilters);
    if (filterEntries.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    resetButton.style.display = 'inline-block';
    listDiv.innerHTML = filterEntries.map(([key, value]) => 
        `<span class="badge bg-primary text-white me-2 mb-2"> 
            ${key.replace(/([A-Z])/g, ' $1').trim()}: ${value} 
            <button type="button" class="btn-close btn-close-white ms-2" aria-label="Close" data-filter-key="${key}"></button>
         </span>`
    ).join('');

    listDiv.querySelectorAll('.btn-close').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const keyToRemove = e.target.dataset.filterKey;
            if (keyToRemove) {
                delete activeFilters[keyToRemove];
                loadChartsAndData();
            }
        });
    });

    resetButton.onclick = () => {
        activeFilters = {};
        loadChartsAndData();
    };
}

document.addEventListener('DOMContentLoaded', () => {
    const container = document.querySelector('.card-body'); // Main card body
    if (!document.getElementById('activeFiltersContainer') && container) {
        // This was in the HTML, but good to ensure it exists or to create it if needed dynamically.
    }
    loadChartsAndData();
});