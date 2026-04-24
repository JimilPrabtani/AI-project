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
let frameCounter = 0;
const zoomCanvas = document.createElement('canvas');
const zoomCtx = zoomCanvas.getContext('2d');

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

const DETECTION_MAX_BOXES = 75;
const DETECTION_MIN_SCORE = 0.12;
const ZOOM_PASS_EVERY_N_FRAMES = 2;
const ZOOM_CROP_RATIO = 0.68;
const DEDUPE_IOU_THRESHOLD = 0.45;

const INCIDENT_MIN_SCORE_DEFAULT = 0.55;
const INCIDENT_MIN_SCORE_BY_CLASS = {
  'cell phone': 0.35,
  'phone': 0.35,
  'mobile phone': 0.35,
  'laptop': 0.45,
  'scissors': 0.50,
  'bottle': 0.55,
  'cup': 0.55,
  'fork': 0.55,
  'glass': 0.60,
  'wine glass': 0.60,
  'knife': 0.65,
};

const DRAW_MIN_SCORE_DEFAULT = 0.30;
const DRAW_MIN_SCORE_BY_CLASS = {
  'cell phone': 0.22,
  'phone': 0.22,
  'mobile phone': 0.22,
  'laptop': 0.35,
};

// Initialize Application
async function init() {
  try {
    modelStatusBadge.textContent = 'Loading Model...';
    modelStatusBadge.className = 'status-badge loading';
    
    // Use mobilenet_v2 instead of the lighter default for higher accuracy
    model = await cocoSsd.load({ base: 'mobilenet_v2' });
    
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

  frameCounter++;
  
  const startTime = performance.now();
  
  // Calculate FPS
  const deltaTime = timestamp - lastFrameTime;
  lastFrameTime = timestamp;
  const fps = Math.round(1000 / deltaTime);
  
  if (fps > 0 && fps < 100) {
    fpsVal.textContent = fps;
  }
  
  try {
    // Base pass on full frame + periodic zoomed pass to improve small-object recall.
    const fullFramePredictions = await model.detect(video, DETECTION_MAX_BOXES, DETECTION_MIN_SCORE);

    let zoomPredictions = [];
    if (frameCounter % ZOOM_PASS_EVERY_N_FRAMES === 0) {
      zoomPredictions = await detectZoomPass();
    }

    const mergedPredictions = mergePredictions(fullFramePredictions, zoomPredictions);
    const forbiddenPredictions = mergedPredictions.filter(prediction =>
      FORBIDDEN_CLASSES.has(prediction.class)
    );
    
    // Performance latency measurement
    const endTime = performance.now();
    const latency = Math.round(endTime - startTime);
    latencyVal.textContent = `${latency} ms`;
    
    drawPredictions(forbiddenPredictions);
    processIncidents(forbiddenPredictions);
    
  } catch (err) {
    console.error('Detection error:', err);
  }
  
  if (isDetecting) {
    animationId = requestAnimationFrame(detectFrame);
  }
}

async function detectZoomPass() {
  if (!video.videoWidth || !video.videoHeight || !zoomCtx) return [];

  const fullWidth = video.videoWidth;
  const fullHeight = video.videoHeight;
  zoomCanvas.width = fullWidth;
  zoomCanvas.height = fullHeight;

  const cropWidth = Math.round(fullWidth * ZOOM_CROP_RATIO);
  const cropHeight = Math.round(fullHeight * ZOOM_CROP_RATIO);
  const cropX = Math.round((fullWidth - cropWidth) / 2);
  const cropY = Math.round((fullHeight - cropHeight) / 2);

  zoomCtx.clearRect(0, 0, fullWidth, fullHeight);
  zoomCtx.drawImage(video, cropX, cropY, cropWidth, cropHeight, 0, 0, fullWidth, fullHeight);

  const zoomPredictions = await model.detect(zoomCanvas, DETECTION_MAX_BOXES, DETECTION_MIN_SCORE);
  const scaleX = cropWidth / fullWidth;
  const scaleY = cropHeight / fullHeight;

  return zoomPredictions.map(prediction => {
    const [x, y, width, height] = prediction.bbox;
    return {
      ...prediction,
      bbox: [
        x * scaleX + cropX,
        y * scaleY + cropY,
        width * scaleX,
        height * scaleY,
      ],
    };
  });
}

function mergePredictions(...predictionGroups) {
  const allPredictions = predictionGroups.flat();
  if (allPredictions.length === 0) return [];

  const sortedPredictions = allPredictions.slice().sort((a, b) => b.score - a.score);
  const merged = [];

  sortedPredictions.forEach(prediction => {
    const isDuplicate = merged.some(existing =>
      existing.class === prediction.class &&
      calculateIoU(existing.bbox, prediction.bbox) >= DEDUPE_IOU_THRESHOLD
    );

    if (!isDuplicate) {
      merged.push(prediction);
    }
  });

  return merged;
}

function calculateIoU(boxA, boxB) {
  const [ax, ay, aw, ah] = boxA;
  const [bx, by, bw, bh] = boxB;

  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);

  const intersectionWidth = Math.max(0, x2 - x1);
  const intersectionHeight = Math.max(0, y2 - y1);
  const intersectionArea = intersectionWidth * intersectionHeight;

  const areaA = aw * ah;
  const areaB = bw * bh;
  const unionArea = areaA + areaB - intersectionArea;

  if (unionArea <= 0) return 0;
  return intersectionArea / unionArea;
}

function getIncidentMinScore(className) {
  return INCIDENT_MIN_SCORE_BY_CLASS[className] ?? INCIDENT_MIN_SCORE_DEFAULT;
}

function getDrawMinScore(className) {
  return DRAW_MIN_SCORE_BY_CLASS[className] ?? DRAW_MIN_SCORE_DEFAULT;
}

function drawPredictions(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Calculate scaling factor between actual video size and displayed size
  const scaleX = canvas.width / video.videoWidth;
  const scaleY = canvas.height / video.videoHeight;
  
  predictions.forEach(prediction => {
    if (prediction.score < getDrawMinScore(prediction.class)) return;

    // The video CSS has `transform: scaleX(-1)` to mirror.
    // We must mirror the bounding box coordinates on the canvas as well.
    let [x, y, width, height] = prediction.bbox;
    
    const scaledX = (video.videoWidth - x - width) * scaleX; 
    const scaledY = y * scaleY;
    const scaledWidth = width * scaleX;
    const scaledHeight = height * scaleY;

    // Draw bounding box
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
    
    // Draw background for label
    ctx.fillStyle = '#ef4444';
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
    if (prediction.score >= getIncidentMinScore(prediction.class)) {
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
