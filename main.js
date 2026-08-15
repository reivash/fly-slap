import {
  FilesetResolver,
  FaceLandmarker,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const FLY_COUNT = 4;
const HIT_SPEED_THRESHOLD = 550; // px/sec, palm must be moving at least this fast to count as a slap
const HIT_RADIUS = 46; // px, how close the swipe path must pass to a fly

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const soundToggle = document.getElementById("soundToggle");

let audioCtx = null;
let playSlap = () => {};
let soundEnabled = true;
let score = 0;

soundToggle.addEventListener("click", () => {
  soundEnabled = !soundEnabled;
  soundToggle.textContent = soundEnabled ? "🔊" : "🔇";
});

function setStatus(msg) {
  statusEl.textContent = msg;
}

function createSlapSound(ctxAudio) {
  return function play(volume = 1) {
    const now = ctxAudio.currentTime;
    const duration = 0.09;
    const bufferSize = Math.floor(ctxAudio.sampleRate * duration);
    const buffer = ctxAudio.createBuffer(1, bufferSize, ctxAudio.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      const t = i / bufferSize;
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, 2);
    }
    const noise = ctxAudio.createBufferSource();
    noise.buffer = buffer;

    const bandpass = ctxAudio.createBiquadFilter();
    bandpass.type = "bandpass";
    bandpass.frequency.setValueAtTime(1800, now);
    bandpass.frequency.exponentialRampToValueAtTime(500, now + duration);
    bandpass.Q.value = 0.7;

    const gain = ctxAudio.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.85 * volume, now + 0.008);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

    noise.connect(bandpass).connect(gain).connect(ctxAudio.destination);
    noise.start(now);
    noise.stop(now + duration + 0.02);
  };
}

function drawFlyShape(c, wingPhase) {
  const flap = Math.sin(wingPhase) * 0.5 + 0.5;
  c.fillStyle = "rgba(225,232,255,0.55)";
  c.beginPath();
  c.ellipse(-2, -6 - flap * 2, 9, 4, -0.4, 0, Math.PI * 2);
  c.fill();
  c.beginPath();
  c.ellipse(-2, 6 + flap * 2, 9, 4, 0.4, 0, Math.PI * 2);
  c.fill();

  c.fillStyle = "#22212b";
  c.beginPath();
  c.ellipse(0, 0, 8, 5, 0, 0, Math.PI * 2);
  c.fill();
  c.beginPath();
  c.arc(7, 0, 4, 0, Math.PI * 2);
  c.fill();

  c.fillStyle = "#cc3333";
  c.beginPath();
  c.arc(8, -1.6, 1.6, 0, Math.PI * 2);
  c.arc(8, 1.6, 1.6, 0, Math.PI * 2);
  c.fill();
}

class Fly {
  constructor(region) {
    this.respawn(region);
    this.state = "alive";
    this.hitTimer = 0;
    this.wingPhase = Math.random() * Math.PI * 2;
    this.angle = 0;
  }

  respawn(region) {
    this.x = region.x0 + Math.random() * (region.x1 - region.x0);
    this.y = region.y0 + Math.random() * (region.y1 - region.y0);
    this.vx = (Math.random() - 0.5) * 40;
    this.vy = (Math.random() - 0.5) * 40;
    this.scale = 0.85 + Math.random() * 0.3;
    this.state = "alive";
  }

  hit() {
    if (this.state !== "alive") return false;
    this.state = "hit";
    this.hitTimer = 0.35;
    return true;
  }

  update(dt, region) {
    this.wingPhase += dt * 40;

    if (this.state === "hit") {
      this.hitTimer -= dt;
      this.y -= 70 * dt;
      this.x += this.vx * 0.6 * dt;
      this.scale *= 0.95;
      if (this.hitTimer <= 0) this.respawn(region);
      return;
    }

    this.vx += (Math.random() - 0.5) * 50 * dt;
    this.vy += (Math.random() - 0.5) * 50 * dt;
    this.vx *= 0.96;
    this.vy *= 0.96;

    const speed = Math.hypot(this.vx, this.vy);
    const maxSpeed = 75;
    if (speed > maxSpeed) {
      this.vx = (this.vx / speed) * maxSpeed;
      this.vy = (this.vy / speed) * maxSpeed;
    }

    if (this.x < region.x0) this.vx += 35 * dt * 10;
    if (this.x > region.x1) this.vx -= 35 * dt * 10;
    if (this.y < region.y0) this.vy += 35 * dt * 10;
    if (this.y > region.y1) this.vy -= 35 * dt * 10;

    this.x += this.vx * dt;
    this.y += this.vy * dt;

    if (speed > 4) this.angle = Math.atan2(this.vy, this.vx);
  }

  draw(c) {
    c.save();
    c.translate(this.x, this.y);
    c.rotate(this.angle);
    c.scale(this.scale, this.scale);
    if (this.state === "hit") {
      c.globalAlpha = Math.max(this.hitTimer / 0.35, 0);
    }
    drawFlyShape(c, this.wingPhase);
    c.restore();
  }
}

function pointToSegmentDistance(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  let t = lenSq === 0 ? 0 : ((px - ax) * dx + (py - ay) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx;
  const cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

let flies = [];
let faceRegion = null; // {x0,y0,x1,y1} in canvas pixel space, mirrored
const prevPalms = new Map(); // hand slot index -> {x,y,t}

function updateFaceRegion(faceLandmarks, w, h) {
  if (!faceLandmarks || faceLandmarks.length === 0) return;
  const pts = faceLandmarks[0];
  let minX = 1,
    maxX = 0,
    minY = 1,
    maxY = 0;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  // mirror x, expand region so flies have room to buzz around the head
  const marginX = (maxX - minX) * 0.9;
  const marginY = (maxY - minY) * 0.9;
  const nx0 = Math.max(0, minX - marginX);
  const nx1 = Math.min(1, maxX + marginX);
  const ny0 = Math.max(0, minY - marginY - 0.1);
  const ny1 = Math.min(1, maxY + marginY);

  faceRegion = {
    x0: w - nx1 * w,
    x1: w - nx0 * w,
    y0: ny0 * h,
    y1: ny1 * h,
  };
}

function processHands(handResult, w, h, dt) {
  const hits = [];
  if (!handResult || !handResult.landmarks) return hits;

  const seenSlots = new Set();

  handResult.landmarks.forEach((lm, i) => {
    const idxs = [0, 5, 9, 13, 17];
    let sx = 0,
      sy = 0;
    for (const idx of idxs) {
      sx += lm[idx].x;
      sy += lm[idx].y;
    }
    sx /= idxs.length;
    sy /= idxs.length;

    const x = w - sx * w;
    const y = sy * h;

    seenSlots.add(i);
    const prev = prevPalms.get(i);
    prevPalms.set(i, { x, y });

    if (!prev || dt <= 0) return;
    const dist = Math.hypot(x - prev.x, y - prev.y);
    const speed = dist / dt;
    if (speed < HIT_SPEED_THRESHOLD) return;

    for (const fly of flies) {
      if (fly.state !== "alive") continue;
      const d = pointToSegmentDistance(fly.x, fly.y, prev.x, prev.y, x, y);
      if (d < HIT_RADIUS) {
        hits.push(fly);
      }
    }
  });

  for (const key of [...prevPalms.keys()]) {
    if (!seenSlots.has(key)) prevPalms.delete(key);
  }

  return hits;
}

async function init() {
  setStatus("Loading vision models…");

  let vision;
  try {
    vision = await FilesetResolver.forVisionTasks(WASM_URL);
  } catch (err) {
    setStatus("Failed to load vision runtime. Check your connection.");
    console.error(err);
    return;
  }

  async function createLandmarker(Ctor, modelUrl, extraOptions) {
    try {
      return await Ctor.createFromOptions(vision, {
        baseOptions: { modelAssetPath: modelUrl, delegate: "GPU" },
        runningMode: "VIDEO",
        ...extraOptions,
      });
    } catch (err) {
      console.warn("GPU delegate failed, falling back to CPU", err);
      return await Ctor.createFromOptions(vision, {
        baseOptions: { modelAssetPath: modelUrl, delegate: "CPU" },
        runningMode: "VIDEO",
        ...extraOptions,
      });
    }
  }

  const [faceLandmarker, handLandmarker] = await Promise.all([
    createLandmarker(FaceLandmarker, FACE_MODEL_URL, { numFaces: 1 }),
    createLandmarker(HandLandmarker, HAND_MODEL_URL, { numHands: 2 }),
  ]);

  setStatus("Requesting camera access…");

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 960, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    setStatus(
      "Camera access denied or unavailable. Allow camera permission and reload."
    );
    console.error(err);
    return;
  }

  video.srcObject = stream;
  await video.play();

  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 960;

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  playSlap = createSlapSound(audioCtx);

  const region0 = { x0: 0, y0: 0, x1: canvas.width, y1: canvas.height };
  flies = Array.from({ length: FLY_COUNT }, () => new Fly(region0));

  overlay.classList.add("hidden");

  let lastTime = performance.now();

  function frame() {
    const now = performance.now();
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    const w = canvas.width;
    const h = canvas.height;

    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -w, 0, w, h);
    ctx.restore();

    const faceResult = faceLandmarker.detectForVideo(video, now);
    const handResult = handLandmarker.detectForVideo(video, now);

    if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
      updateFaceRegion(faceResult.faceLandmarks, w, h);
    }

    const region = faceRegion || {
      x0: w * 0.3,
      x1: w * 0.7,
      y0: h * 0.2,
      y1: h * 0.6,
    };

    const hits = processHands(handResult, w, h, dt);
    for (const fly of hits) {
      if (fly.hit()) {
        score += 1;
        scoreEl.textContent = `Swatted: ${score}`;
        if (soundEnabled && audioCtx) {
          if (audioCtx.state === "suspended") audioCtx.resume();
          playSlap();
        }
      }
    }

    for (const fly of flies) {
      fly.update(dt, region);
      fly.draw(ctx);
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
  setStatus("");
}

startBtn.addEventListener("click", () => {
  startBtn.disabled = true;
  init().catch((err) => {
    console.error(err);
    setStatus("Something went wrong starting the camera. See console for details.");
  });
});
