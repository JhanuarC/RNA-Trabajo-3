const API_BASE = "http://127.0.0.1:8000/api";

const category_names = [
    'churches', 'resorts', 'beaches', 'parks', 'theatres', 'museums', 'malls', 'zoo', 
    'restaurants', 'pubs_bars', 'local_services', 'burger_pizza', 'hotels', 'juice_bars', 
    'art_galleries', 'dance_clubs', 'swimming_pools', 'gyms', 'bakeries', 'beauty_spas', 
    'cafes', 'viewpoints', 'monuments', 'gardens'
];

let currentUserRole = null;
let demandChartInstance = null;

// INIT SLIDERS
function initSliders() {
    const container = document.getElementById('sliders-container');
    category_names.forEach((cat, index) => {
        const div = document.createElement('div');
        div.className = 'slider-item';
        div.innerHTML = `
            <label for="slider-${index}">${cat.replace('_', ' ')}: <span id="val-${index}">0</span></label>
            <input type="range" id="slider-${index}" min="0" max="5" value="0" step="0.5" oninput="document.getElementById('val-${index}').innerText = this.value">
        `;
        container.appendChild(div);
    });
}
initSliders();

// LOGIN LOGIC
function loginGuest() {
    currentUserRole = 'guest';
    document.getElementById('user-role-badge').innerText = 'Cliente Invitado';
    document.getElementById('user-role-badge').style.color = '#10b981';
    
    document.querySelectorAll('.admin-only').forEach(el => el.classList.add('hidden'));
    document.getElementById('login-view').classList.add('hidden');
    document.getElementById('dashboard-view').classList.remove('hidden');
    switchTab('module3');
}

function loginAdmin(e) {
    e.preventDefault();
    const user = document.getElementById('username').value;
    const pass = document.getElementById('password').value;
    
    if(user === 'admin' && pass === 'admin') {
        currentUserRole = 'admin';
        document.getElementById('user-role-badge').innerText = 'Administrador';
        document.getElementById('user-role-badge').style.color = '#ef4444';
        
        document.querySelectorAll('.admin-only').forEach(el => el.classList.remove('hidden'));
        document.getElementById('login-view').classList.add('hidden');
        document.getElementById('dashboard-view').classList.remove('hidden');
        document.getElementById('login-error').innerText = '';
        switchTab('module1');
    } else {
        document.getElementById('login-error').innerText = 'Usuario o contraseña incorrectos';
    }
}

function logout() {
    currentUserRole = null;
    document.getElementById('dashboard-view').classList.add('hidden');
    document.getElementById('login-view').classList.remove('hidden');
    document.getElementById('admin-login-form').reset();
}

// TAB SWITCHING
function switchTab(tabId) {
    document.querySelectorAll('.nav-links li').forEach(el => el.classList.remove('active'));
    document.getElementById('nav-' + tabId).classList.add('active');
    
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.getElementById(tabId).classList.remove('hidden');
}

// Country flag emojis for display
const COUNTRY_FLAGS = {
    'Argentina': '🇦🇷', 'Australia': '🇦🇺', 'Brazil': '🇧🇷', 'Canada': '🇨🇦',
    'China': '🇨🇳', 'Egypt': '🇪🇬', 'France': '🇫🇷', 'Germany': '🇩🇪',
    'Greece': '🇬🇷', 'India': '🇮🇳', 'Italy': '🇮🇹', 'Japan': '🇯🇵',
    'Kenya': '🇰🇪', 'Mexico': '🇲🇽', 'Morocco': '🇲🇦', 'New Zealand': '🇳🇿',
    'Peru': '🇵🇪', 'South Africa': '🇿🇦', 'Spain': '🇪🇸', 'Thailand': '🇹🇭',
    'USA': '🇺🇸', 'Vietnam': '🇻🇳'
};

// MODULE 3: RECOMMENDATION
document.getElementById('preferences-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const ratings = [];
    for(let i=0; i<category_names.length; i++) {
        ratings.push(parseFloat(document.getElementById(`slider-${i}`).value));
    }
    
    const resultsContainer = document.getElementById('recom-results');
    resultsContainer.innerHTML = '<div class="placeholder-text">Analizando perfil con IA...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/recommend_destinations`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({user_ratings: ratings})
        });
        const data = await response.json();
        
        resultsContainer.innerHTML = '';
        const topCountries = data.top_countries.slice(0, 5);
        topCountries.forEach((item, idx) => {
            const perc = (item.score * 100).toFixed(1);
            const flag = COUNTRY_FLAGS[item.country] || '🌍';
            const adminBtn = currentUserRole === 'admin' 
                ? `<button class="btn btn-outline btn-sm" onclick="goToDemand('${item.country}')">Ver Demanda →</button>`
                : '';
            resultsContainer.innerHTML += `
                <div class="dest-card">
                    <h4>${flag} ${item.country}</h4>
                    <div class="score">Afinidad: ${perc}%</div>
                    ${adminBtn}
                </div>
            `;
        });
    } catch(err) {
        resultsContainer.innerHTML = `<div class="error-msg">Error de conexión con el servidor. Verifica que el backend esté ejecutándose.</div>`;
    }
});

// Helper: jump to Module 1 with a pre-selected country
function goToDemand(country) {
    const select = document.getElementById('demand-country');
    select.value = country;
    switchTab('module1');
}

// MODULE 1: DEMAND PREDICTION
async function getDemand() {
    const country = document.getElementById('demand-country').value;
    const month = document.getElementById('demand-month').value;
    const horizon = document.getElementById('demand-horizon').value;
    
    // Show loader and clear chart if exists
    const btn = document.getElementById('btn-demand');
    btn.innerText = "Calculando...";
    btn.disabled = true;
    document.getElementById('demand-loader').classList.remove('hidden');
    
    if(demandChartInstance) {
        demandChartInstance.destroy();
        demandChartInstance = null;
    }
    
    try {
        const response = await fetch(`${API_BASE}/predict_demand`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({country: country, month: parseInt(month), horizon: parseInt(horizon)})
        });
        const data = await response.json();
        
        btn.innerText = "Proyectar Demanda";
        btn.disabled = false;
        document.getElementById('demand-loader').classList.add('hidden');
        renderChart(data);
    } catch(err) {
        console.error("Error fetching demand:", err);
        btn.innerText = "Proyectar Demanda";
        btn.disabled = false;
        document.getElementById('demand-loader').classList.add('hidden');
        alert("Error de conexión con el servidor. Verifica que el backend esté ejecutándose.");
    }
}

function renderChart(data) {
    const ctx = document.getElementById('demandChart').getContext('2d');
    if(demandChartInstance) {
        demandChartInstance.destroy();
    }
    
    demandChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.days,
            datasets: [
                { label: 'Bus de lujo', data: data.luxury_bus, borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', tension: 0.4, fill: true},
                { label: 'Van Turístico', data: data.tourist_van, borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', tension: 0.4, fill: true},
                { label: 'Tren Express', data: data.train_express, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', tension: 0.4, fill: true}
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, title: {display: true, text: 'Pasajeros Estimados', color: '#f8fafc'} }
            }
        }
    });
}

// MODULE 2: DRIVER SECURITY
let selectedFile = null;

function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('image-preview');
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            document.getElementById('btn-classify').disabled = false;
        }
        reader.readAsDataURL(file);
    }
}

async function classifyDriver() {
    if(!selectedFile) return;
    
    const resultDiv = document.getElementById('security-result');
    resultDiv.innerHTML = '<div class="result-placeholder">Analizando imagen con ResNet50...</div>';
    resultDiv.className = 'result-placeholder';
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch(`${API_BASE}/classify_driver`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        let stateClass = '';
        let titleClass = '';
        let label = '';
        
        if(data.class === 'safe_driving') {
            stateClass = 'result-safe';
            titleClass = 'class-label-safe';
            label = 'Conducción Segura';
        } else {
            stateClass = 'result-danger';
            titleClass = 'class-label-danger';
            
            if(data.class === 'talking_phone') label = 'Peligro: Hablando por Celular';
            else if(data.class === 'texting_phone') label = 'Peligro: Escribiendo en Celular';
            else if(data.class === 'turning') label = 'Peligro: Volteando hacia atrás';
            else label = 'Peligro: Distraído / Otras Actividades';
            
            // Play Audio
            document.getElementById('alert-audio').play();
        }
        
        const probPerc = (data.probability * 100).toFixed(2);
        
        resultDiv.className = stateClass;
        resultDiv.innerHTML = `
            <div class="${titleClass}">${label}</div>
            <div class="prob-text">Confianza de la Red: ${probPerc}%</div>
        `;
        
    } catch(err) {
        console.error("Error classifying:", err);
        resultDiv.innerHTML = `<div class="error-msg">Error de conexión con el servidor.</div>`;
        resultDiv.className = 'result-placeholder';
    }
}
