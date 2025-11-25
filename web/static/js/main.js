// AI-Powered Refugee Crisis Intelligence System - JavaScript

// Global flag to track if initial load is complete
let initialLoadComplete = false;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    loadSystemStatus();
    loadAgents();
    loadMetrics();
    // Charts removed - using static visualizations instead
    initialLoadComplete = true;
});

// Load system status
async function loadSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        // Update header badge
        const statusBadge = document.getElementById('system-status');
        statusBadge.textContent = `● ${data.status === 'operational' ? 'System Operational' : 'System Offline'}`;

        // Update LLM backend badge
        const llmBadge = document.getElementById('llm-backend');
        llmBadge.textContent = data.llm_backend;
    } catch (error) {
        console.error('Error loading system status:', error);
    }
}

// Load agents information
async function loadAgents() {
    try {
        const response = await fetch('/api/agents');
        const agents = await response.json();

        const agentsGrid = document.getElementById('agents-grid');
        agentsGrid.innerHTML = '';

        for (const [key, agent] of Object.entries(agents)) {
            const agentCard = createAgentCard(agent);
            agentsGrid.appendChild(agentCard);
        }
    } catch (error) {
        console.error('Error loading agents:', error);
    }
}

// Create agent card element
function createAgentCard(agent) {
    const card = document.createElement('div');
    card.className = 'agent-card';
    card.innerHTML = `
        <div class="agent-icon">${agent.icon}</div>
        <h3 class="agent-title">${agent.name}</h3>
        <p class="agent-role">${agent.role}</p>
        <ul class="agent-capabilities">
            ${agent.capabilities.map(cap => `<li>${cap}</li>`).join('')}
        </ul>
        <span class="agent-status">${agent.status}</span>
    `;
    return card;
}

// Load metrics
async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();

        // Update hero stats
        document.getElementById('stat-predictions').textContent = data.total_predictions;
        document.getElementById('stat-accuracy').textContent = `${data.avg_prediction_accuracy}%`;
        document.getElementById('stat-threats').textContent = data.threats_detected;
        document.getElementById('stat-countries').textContent = data.countries_monitored;

        // Only animate numbers on initial load, not on refresh
        if (!initialLoadComplete) {
            animateNumbers();
        }
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// Animate stat numbers
function animateNumbers() {
    const stats = document.querySelectorAll('.stat-value');
    stats.forEach(stat => {
        const target = parseFloat(stat.textContent.replace(/[^0-9.]/g, ''));
        let current = 0;
        const increment = target / 50;
        const isPercentage = stat.textContent.includes('%');

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            stat.textContent = isPercentage ?
                `${Math.round(current * 10) / 10}%` :
                Math.round(current);
        }, 20);
    });
}

// Chart functions removed - using static visualizations instead for better performance

// Run workflow
async function runWorkflow(workflowType) {
    const demoOutput = document.getElementById('demo-output');
    const demoConsole = document.getElementById('demo-console');

    // Show demo output
    demoOutput.style.display = 'block';
    demoConsole.innerHTML = '<div class="demo-line demo-log">Initializing workflow...</div>';

    // Scroll to demo output
    demoOutput.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    try {
        // Start workflow
        const response = await fetch('/api/demo/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ workflow: workflowType })
        });

        if (!response.ok) {
            throw new Error('Failed to start workflow');
        }

        // Listen for server-sent events
        const eventSource = new EventSource('/api/demo/stream');

        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleDemoEvent(data, demoConsole);
        };

        eventSource.onerror = function() {
            eventSource.close();
        };

    } catch (error) {
        demoConsole.innerHTML += `<div class="demo-line demo-error">Error: ${error.message}</div>`;
    }
}

// Handle demo event
function handleDemoEvent(data, demoConsole) {
    let line = '';

    switch (data.type) {
        case 'start':
            line = `<div class="demo-line demo-success">🚀 Starting ${data.workflow.toUpperCase()} workflow...</div>`;
            break;

        case 'log':
            line = `<div class="demo-line demo-log">${escapeHtml(data.message)}</div>`;
            break;

        case 'result':
            const resultText = JSON.stringify(data.data, null, 2);
            line = `<div class="demo-line demo-result">📊 ${data.agent}: ${escapeHtml(resultText)}</div>`;
            break;

        case 'complete':
            line = `<div class="demo-line demo-success">${escapeHtml(data.message)}</div>`;
            line += `<div class="demo-line demo-log">Completed at: ${data.timestamp}</div>`;
            break;

        case 'error':
            line = `<div class="demo-line demo-error">❌ ${escapeHtml(data.message)}</div>`;
            break;
    }

    if (line) {
        demoConsole.innerHTML += line;
        demoConsole.scrollTop = demoConsole.scrollHeight;
    }
}

// Clear demo output
function clearDemo() {
    const demoConsole = document.getElementById('demo-console');
    demoConsole.innerHTML = '<div class="demo-line demo-log">Demo output cleared. Click a workflow to run again.</div>';
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Refresh only system status every 60 seconds (not metrics to avoid re-animation)
setInterval(() => {
    loadSystemStatus();
    // Don't reload metrics to avoid CPU-heavy re-animations
}, 60000);
