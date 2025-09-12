async function fetchOHLC() {
    const res = await fetch("/ohlc_data");
    return await res.json();
}

async function fetchFeatures(index) {
    const res = await fetch(`/candle_features/${index}`);
    return await res.json();
}

async function initChart() {
    const data = await fetchOHLC();

    // Map time -> index for fast lookup
    const timeToIndex = {};
    data.forEach((d, i) => {
        timeToIndex[d.time] = i;
    });

    const chartContainer = document.getElementById('chart');

    // Create chart with initial size
    const chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: 600, // initial height
        layout: {
            backgroundColor: '#ffffff',
            textColor: '#000',
        },
        grid: {
            vertLines: { color: '#eee' },
            horzLines: { color: '#eee' },
        },
        rightPriceScale: {
            borderColor: '#ccc',
        },
        timeScale: {
            borderColor: '#ccc',
        },
    });

    const candleSeries = chart.addCandlestickSeries();
    candleSeries.setData(data);

    // Highlight clicked candle
    let marker = null;

    chart.subscribeClick(async (param) => {
        if (!param || !param.time) return;

        const index = timeToIndex[param.time];
        if (index === undefined) return;

        // Fetch features
        const features = await fetchFeatures(index);

        // Update table
        const table = document.getElementById('features');
        const thead = table.querySelector('thead tr');
        const tbody = table.querySelector('tbody');
        thead.innerHTML = '';
        tbody.innerHTML = '';

        Object.keys(features).forEach(k => {
            const th = document.createElement('th');
            th.textContent = k;
            thead.appendChild(th);
        });

        const tr = document.createElement('tr');
        Object.values(features).forEach(v => {
            const td = document.createElement('td');
            td.textContent = v;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);

        // Add or move marker to highlight candle
        if (marker) candleSeries.removeMarker(marker.id);
        marker = {
            id: 'selected-candle',
            time: param.time,
            position: 'aboveBar',
            color: 'red',
            shape: 'arrowUp',
            text: 'Selected',
        };
        candleSeries.setMarkers([marker]);
    });

    // Make chart responsive
    window.addEventListener('resize', () => {
        chart.applyOptions({ width: chartContainer.clientWidth });
    });
}

initChart();
