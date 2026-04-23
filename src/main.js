import './style.css';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

// UI Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const toggleCameraBtn = document.getElementById('toggle-camera-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const cameraPrompt = document.getElementById('camera-prompt');
const modelStatusBadge = document.getElementById('model-status');
const latencyVal = document.getElementById('latency-val');
const fpsVal = document.getElementById('fps-val');
const incidentLog = document.getElementById('incident-log');
const incidentCountBadge = document.getElementById('incident-count');

// State
let model = null;
let isDetecting = false;
let animationId = null;
let lastFrameTime = 0;
let incidentCount = 0;
let loggedIncidentsCache = new Set(); // To debounce logs

// Forbidden objects configuration
const FORBIDDEN_CLASSES = new Set([
  'cell phone', 
  'phone',
  'mobile phone',
  'glass',
  'scissors',
  'laptop', 
  'knife', 
  'wine glass', 
  'bottle', 
  'cup', 
  'fork',
  // Note: COCO-SSD doesn't have "vape" or "gun" directly, so we use proxies or standard classes.
]);

// Initialize Application
async function init() {
  try {
    modelStatusBadge.textContent = 'Loading Model...';
    modelStatusBadge.className = 'status-badge loading';
    
    model = await cocoSsd.load();
    
    modelStatusBadge.textContent = 'Ready';
    modelStatusBadge.className = 'status-badge ready';
    
    loadingOverlay.classList.add('hidden');
    cameraPrompt.classList.remove('hidden');
    
    toggleCameraBtn.disabled = false;
    toggleCameraBtn.addEventListener('click', toggleCamera);
    
    // Handle window resize for canvas scaling
    window.addEventListener('resize', resizeCanvas);
  } catch (error) {
    console.error('Failed to load model:', error);
    modelStatusBadge.textContent = 'Error';
    modelStatusBadge.style.color = '#ef4444';
  }
}

// Start / Stop Camera
async function toggleCamera() {
  if (isDetecting) {
    stopDetection();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false
    });
    
    video.srcObject = stream;
    
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.play();
        cameraPrompt.classList.add('hidden');
        toggleCameraBtn.textContent = 'Stop Camera';
        isDetecting = true;
        resizeCanvas();
        detectFrame(performance.now());
        resolve();
      };
    });
  } catch (error) {
    console.error('Error accessing webcam:', error);
    alert('Please allow webcam access to use the monitoring system.');
  }
}

function stopDetection() {
  if (animationId) cancelAnimationFrame(animationId);
  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  isDetecting = false;
  cameraPrompt.classList.remove('hidden');
  toggleCameraBtn.textContent = 'Start Camera';
  latencyVal.textContent = '-- ms';
  fpsVal.textContent = '--';
}

function resizeCanvas() {
  if (!video.videoWidth) return;
  
  // Match canvas dimensions to the displayed video dimensions
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

async function detectFrame(timestamp) {
  if (!isDetecting) return;
  
  const startTime = performance.now();
  
  // Calculate FPS
  const deltaTime = timestamp - lastFrameTime;
  lastFrameTime = timestamp;
  const fps = Math.round(1000 / deltaTime);
  
  if (fps > 0 && fps < 100) {
    fpsVal.textContent = fps;
  }
  
  try {
    const predictions = await model.detect(video);
    
    // Performance latency measurement
    const endTime = performance.now();
    const latency = Math.round(endTime - startTime);
    latencyVal.textContent = `${latency} ms`;
    
    drawPredictions(predictions);
    processIncidents(predictions);
    
  } catch (err) {
    console.error('Detection error:', err);
  }
  
  if (isDetecting) {
    animationId = requestAnimationFrame(detectFrame);
  }
}

function drawPredictions(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Calculate scaling factor between actual video size and displayed size
  const scaleX = canvas.width / video.videoWidth;
  const scaleY = canvas.height / video.videoHeight;
  
  predictions.forEach(prediction => {
    // The video CSS has `transform: scaleX(-1)` to mirror.
    // We must mirror the bounding box coordinates on the canvas as well.
    let [x, y, width, height] = prediction.bbox;
    
    const scaledX = (video.videoWidth - x - width) * scaleX; 
    const scaledY = y * scaleY;
    const scaledWidth = width * scaleX;
    const scaledHeight = height * scaleY;

    const isForbidden = FORBIDDEN_CLASSES.has(prediction.class);
    
    // Draw bounding box
    ctx.strokeStyle = isForbidden ? '#ef4444' : '#10b981'; // Red or Green
    ctx.lineWidth = 3;
    ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
    
    // Draw background for label
    ctx.fillStyle = isForbidden ? '#ef4444' : '#10b981';
    const text = `${prediction.class} - ${Math.round(prediction.score * 100)}%`;
    ctx.font = '16px Inter';
    const textWidth = ctx.measureText(text).width;
    
    ctx.fillRect(scaledX, scaledY > 25 ? scaledY - 25 : 0, textWidth + 10, 25);
    
    // Draw label text
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, scaledX + 5, scaledY > 25 ? scaledY - 8 : 17);
  });
}

function processIncidents(predictions) {
  predictions.forEach(prediction => {
    if (FORBIDDEN_CLASSES.has(prediction.class) && prediction.score > 0.60) {
      logIncident(prediction.class, prediction.score);
    }
  });
}

function logIncident(className, confidence) {
  const cacheKey = className;
  
  // Throttle logs so we don't spam the same object every frame
  if (loggedIncidentsCache.has(cacheKey)) return;
  
  loggedIncidentsCache.add(cacheKey);
  
  // Remove from cache after 5 seconds to log again if it reappears
  setTimeout(() => {
    loggedIncidentsCache.delete(cacheKey);
  }, 5000);
  
  // Update UI
  if (incidentCount === 0) {
    incidentLog.innerHTML = ''; // Clear placeholder
  }
  
  incidentCount++;
  incidentCountBadge.textContent = incidentCount;
  
  const li = document.createElement('li');
  li.className = 'log-item';
  
  const timeInfo = new Date().toLocaleTimeString();
  li.innerHTML = `
    <strong>Forbidden Object:</strong> ${className} (${Math.round(confidence * 100)}%)
    <span class="log-time">${timeInfo}</span>
  `;
  
  incidentLog.prepend(li);
}

// Start initialization
init();
