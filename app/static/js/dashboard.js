/**
 * Fraud Detection System (FDS) - Dashboard Client Helpers
 * Author: Alpha
 */

// Window resize listener for responsive Plotly graphs
window.addEventListener('resize', function () {
    const charts = ['trendChart', 'riskDistChart', 'merchantChart', 'shapChart', 'simShapChart'];
    charts.forEach(id => {
        const el = document.getElementById(id);
        if (el && window.Plotly) {
            Plotly.Plots.resize(el);
        }
    });
});
