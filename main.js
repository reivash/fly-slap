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

// Hit detection: any overlap counts as a slap (touch), a fast-moving hand
// gets a bigger reach and a stronger launch (swipe). This dual mode makes
// slapping forgiving without requiring a precise fast swipe every time.
const HIT_TOUCH_RADIUS = 42;
const HIT_SWIPE_RADIUS = 95;
const HIT_SWIPE_SPEED = 260; // px/sec

const COMBO_WINDOW = 1300; // ms between hits to keep a combo alive

// Approach behavior: a fly far from the face swoops toward it in a curved
// arc rather than a straight line. Once inside ENTER_HOVER_DIST it switches
// to the slower chaotic hover.
const APPROACH_SPEED = 900; // px/sec
const APPROACH_SNAPPINESS = 12; // higher = velocity reaches approach speed faster
const ENTER_HOVER_DIST = 170;
const EXIT_HOVER_DIST = 260; // must wander this far back out to re-trigger a dash-in

// Arcing: the seek direction (both approach and hover) is rotated by a
// slowly oscillating angle so flight paths bend into broad curves instead
// of beelining or jittering in straight segments.
const ARC_AMPLITUDE_MIN = 0.9; // radians (~52°)
const ARC_AMPLITUDE_MAX = 1.4; // radians (~80°)
const ARC_SPEED_MIN = 0.6; // rad/sec
const ARC_SPEED_MAX = 1.2; // rad/sec

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const comboEl = document.getElementById("combo");
const soundToggle = document.getElementById("soundToggle");

let audioCtx = null;
let masterGain = null;
let playSlap = () => {};
let playPop = () => {};
let soundEnabled = true;

let score = 0;
let combo = 0;
let lastHitAt = 0;

soundToggle.addEventListener("click", () => {
  soundEnabled = !soundEnabled;
  soundToggle.textContent = soundEnabled ? "🔊" : "🔇";
  if (masterGain && audioCtx) {
    masterGain.gain.setTargetAtTime(
      soundEnabled ? 1 : 0,
      audioCtx.currentTime,
      0.05
    );
  }
});

function setStatus(msg) {
  statusEl.textContent = msg;
}

function updateScoreUI() {
  scoreEl.textContent = `Swatted: ${score}`;
}

function updateComboUI() {
  if (combo >= 2) {
    comboEl.textContent = `🔥 Combo x${combo}`;
    comboEl.classList.remove("hidden");
    comboEl.classList.remove("pop");
    // restart the pop animation
    void comboEl.offsetWidth;
    comboEl.classList.add("pop");
  } else {
    comboEl.classList.add("hidden");
  }
}

function registerHit(nowMs) {
  score += 1;
  combo = nowMs - lastHitAt < COMBO_WINDOW ? combo + 1 : 1;
  lastHitAt = nowMs;
  updateScoreUI();
  updateComboUI();
}

function maybeExpireCombo(nowMs) {
  if (combo > 0 && nowMs - lastHitAt > COMBO_WINDOW) {
    combo = 0;
    comboEl.classList.add("hidden");
  }
}

// ---------------------------------------------------------------------------
// Audio: soft synthesized impact/pop sounds plus a persistent buzz per fly
// ---------------------------------------------------------------------------

function createSlapSound(ctxAudio, destination) {
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

    noise.connect(bandpass).connect(gain).connect(destination);
    noise.start(now);
    noise.stop(now + duration + 0.02);
  };
}

function createPopSound(ctxAudio, destination) {
  return function play() {
    const now = ctxAudio.currentTime;
    const duration = 0.16;

    const osc = ctxAudio.createOscillator();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(520, now);
    osc.frequency.exponentialRampToValueAtTime(90, now + duration);

    const oscGain = ctxAudio.createGain();
    oscGain.gain.setValueAtTime(0.0001, now);
    oscGain.gain.exponentialRampToValueAtTime(0.7, now + 0.012);
    oscGain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

    osc.connect(oscGain).connect(destination);
    osc.start(now);
    osc.stop(now + duration + 0.02);

    const crackleSize = Math.floor(ctxAudio.sampleRate * 0.05);
    const buffer = ctxAudio.createBuffer(1, crackleSize, ctxAudio.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < crackleSize; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / crackleSize, 3);
    }
    const noise = ctxAudio.createBufferSource();
    noise.buffer = buffer;
    const noiseGain = ctxAudio.createGain();
    noiseGain.gain.setValueAtTime(0.4, now);
    noiseGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.05);
    noise.connect(noiseGain).connect(destination);
    noise.start(now);
  };
}

// ---------------------------------------------------------------------------
// Particles
// ---------------------------------------------------------------------------

let particles = [];

function spawnExplosion(x, y, count, color) {
  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const speed = 90 + Math.random() * 260;
    const maxLife = 0.3 + Math.random() * 0.3;
    particles.push({
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      life: maxLife,
      maxLife,
      size: 2 + Math.random() * 3.5,
      color,
    });
  }
}

function updateParticles(dt) {
  for (const p of particles) {
    p.vx *= 0.92;
    p.vy = p.vy * 0.92 + 320 * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.life -= dt;
  }
  particles = particles.filter((p) => p.life > 0);
}

function drawParticles(c) {
  for (const p of particles) {
    c.globalAlpha = Math.max(p.life / p.maxLife, 0);
    c.fillStyle = p.color;
    c.beginPath();
    c.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    c.fill();
  }
  c.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// Flies
// ---------------------------------------------------------------------------

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
  constructor(canvasSize, audioCtxRef, destination, onBorderHit) {
    this.audioCtx = audioCtxRef;
    this.onBorderHit = onBorderHit;
    this.wingPhase = Math.random() * Math.PI * 2;
    this.angle = 0;
    this.target = { x: 0, y: 0 };
    this.retargetTimer = 0;
    this.arcPhase = Math.random() * Math.PI * 2;
    this.randomizeArc();
    if (audioCtxRef) this.initAudio(destination);
    this.spawnAtEdge(canvasSize);
  }

  randomizeArc() {
    this.arcAmplitude =
      ARC_AMPLITUDE_MIN + Math.random() * (ARC_AMPLITUDE_MAX - ARC_AMPLITUDE_MIN);
    this.arcSpeed =
      (ARC_SPEED_MIN + Math.random() * (ARC_SPEED_MAX - ARC_SPEED_MIN)) *
      (Math.random() < 0.5 ? -1 : 1);
  }

  // Rotates a unit direction vector by `angle` radians, used to bend
  // straight seek/approach directions into broad curved arcs.
  static rotate(x, y, angle) {
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    return { x: x * cosA - y * sinA, y: x * sinA + y * cosA };
  }

  initAudio(destination) {
    const c = this.audioCtx;
    this.baseFreq = 170 + Math.random() * 60;
    this.osc1 = c.createOscillator();
    this.osc2 = c.createOscillator();
    this.osc1.type = "sawtooth";
    this.osc2.type = "sawtooth";
    this.osc1.frequency.value = this.baseFreq;
    this.osc2.frequency.value = this.baseFreq * 1.5;

    this.filter = c.createBiquadFilter();
    this.filter.type = "bandpass";
    this.filter.frequency.value = 260;
    this.filter.Q.value = 1.1;

    this.buzzGain = c.createGain();
    this.buzzGain.gain.value = 0;

    this.panner = c.createStereoPanner ? c.createStereoPanner() : null;

    this.osc1.connect(this.filter);
    this.osc2.connect(this.filter);
    if (this.panner) {
      this.filter.connect(this.buzzGain).connect(this.panner).connect(destination);
    } else {
      this.filter.connect(this.buzzGain).connect(destination);
    }
    this.osc1.start();
    this.osc2.start();
  }

  updateAudio(faceCenter, canvasWidth) {
    if (!this.audioCtx) return;
    const dist = Math.hypot(this.x - faceCenter.x, this.y - faceCenter.y);
    const proximity = Math.max(0, 1 - dist / 420);
    const now = this.audioCtx.currentTime;
    const targetGain = 0.02 + proximity * proximity * 0.4;
    this.buzzGain.gain.setTargetAtTime(targetGain, now, 0.05);
    const freq = this.baseFreq * (1 + proximity * 0.7);
    this.osc1.frequency.setTargetAtTime(freq, now, 0.08);
    this.osc2.frequency.setTargetAtTime(freq * 1.5, now, 0.08);
    if (this.panner) {
      const pan = Math.max(-1, Math.min(1, (this.x / canvasWidth) * 2 - 1));
      this.panner.pan.setTargetAtTime(pan, now, 0.12);
    }
  }

  muteAudio() {
    if (!this.audioCtx) return;
    this.buzzGain.gain.setTargetAtTime(0, this.audioCtx.currentTime, 0.03);
  }

  spawnAtEdge(canvasSize) {
    const { width: w, height: h } = canvasSize;
    const edge = Math.floor(Math.random() * 4);
    if (edge === 0) {
      this.x = -20;
      this.y = Math.random() * h;
    } else if (edge === 1) {
      this.x = w + 20;
      this.y = Math.random() * h;
    } else if (edge === 2) {
      this.x = Math.random() * w;
      this.y = -20;
    } else {
      this.x = Math.random() * w;
      this.y = h + 20;
    }
    this.vx = (Math.random() - 0.5) * 60;
    this.vy = (Math.random() - 0.5) * 60;
    this.scale = 0.85 + Math.random() * 0.3;
    this.target.x = this.x;
    this.target.y = this.y;
    this.retargetTimer = 0;
    this.state = "alive";
    this.approaching = true;
    this.randomizeArc();
  }

  hit(dirX, dirY) {
    if (this.state !== "alive") return false;
    this.state = "launched";
    const len = Math.hypot(dirX, dirY) || 1;
    const launchSpeed = 780 + Math.random() * 220;
    this.vx = (dirX / len) * launchSpeed;
    this.vy = (dirY / len) * launchSpeed - 140;
    this.muteAudio();
    return true;
  }

  update(dt, region, faceCenter, canvasSize) {
    this.wingPhase += dt * (55 + Math.hypot(this.vx, this.vy) * 0.25);
    this.arcPhase += dt * this.arcSpeed;

    if (this.state === "exploded") {
      this.respawnTimer -= dt;
      if (this.respawnTimer <= 0) this.spawnAtEdge(canvasSize);
      return;
    }

    if (this.state === "launched") {
      this.vy += 620 * dt;
      this.x += this.vx * dt;
      this.y += this.vy * dt;
      this.angle = Math.atan2(this.vy, this.vx);
      this.scale = Math.max(0.25, this.scale - dt * 0.5);
      const { width, height } = canvasSize;
      if (this.x < -8 || this.x > width + 8 || this.y < -8 || this.y > height + 8) {
        const bx = Math.min(Math.max(this.x, 4), width - 4);
        const by = Math.min(Math.max(this.y, 4), height - 4);
        if (this.onBorderHit) this.onBorderHit(bx, by);
        this.state = "exploded";
        this.respawnTimer = 0.45 + Math.random() * 0.35;
      }
      return;
    }

    // alive: decide whether we're still dashing in toward the face or
    // close enough to switch to the slower chaotic hover, with a bit of
    // hysteresis so it doesn't flicker between the two modes.
    const distToFace = Math.hypot(this.x - faceCenter.x, this.y - faceCenter.y);
    if (this.approaching && distToFace < ENTER_HOVER_DIST) this.approaching = false;
    if (!this.approaching && distToFace > EXIT_HOVER_DIST) this.approaching = true;

    if (this.approaching) {
      const toX = faceCenter.x - this.x;
      const toY = faceCenter.y - this.y;
      const toLen = Math.hypot(toX, toY) || 1;
      // bend the straight-line approach into a broad curve, tapering the
      // bend out as it nears the target so it still lands on the face
      const taper = Math.min(1, toLen / 480);
      const swing = Math.sin(this.arcPhase) * this.arcAmplitude * taper;
      const curved = Fly.rotate(toX / toLen, toY / toLen, swing);
      const desiredVx = curved.x * APPROACH_SPEED;
      const desiredVy = curved.y * APPROACH_SPEED;
      const ease = Math.min(1, APPROACH_SNAPPINESS * dt);
      this.vx += (desiredVx - this.vx) * ease;
      this.vy += (desiredVy - this.vy) * ease;
      this.x += this.vx * dt;
      this.y += this.vy * dt;
      this.angle = Math.atan2(this.vy, this.vx);
      this.retargetTimer = 0;
      this.updateAudio(faceCenter, canvasSize.width);
      return;
    }

    // close: dart erratically while continually re-aiming near the face
    this.retargetTimer -= dt;
    if (this.retargetTimer <= 0) {
      const angle = Math.random() * Math.PI * 2;
      const spread = Math.max(
        60,
        Math.min(region.x1 - region.x0, region.y1 - region.y0) * 0.55
      );
      const r = Math.random() * spread;
      this.target.x = faceCenter.x + Math.cos(angle) * r;
      this.target.y = faceCenter.y + Math.sin(angle) * r;
      this.retargetTimer = 0.12 + Math.random() * 0.3;
    }

    const toX = this.target.x - this.x;
    const toY = this.target.y - this.y;
    const toLen = Math.hypot(toX, toY) || 1;
    const swing = Math.sin(this.arcPhase) * this.arcAmplitude * 0.6;
    const curved = Fly.rotate(toX / toLen, toY / toLen, swing);
    const seekStrength = 300;
    this.vx += curved.x * seekStrength * dt;
    this.vy += curved.y * seekStrength * dt;

    // erratic jitter so it darts like an insect instead of gliding
    this.vx += (Math.random() - 0.5) * 460 * dt;
    this.vy += (Math.random() - 0.5) * 460 * dt;

    // occasional sharp flinch in a random direction
    if (Math.random() < dt * 1.3) {
      const a = Math.random() * Math.PI * 2;
      this.vx += Math.cos(a) * 220;
      this.vy += Math.sin(a) * 220;
    }

    this.vx *= 0.9;
    this.vy *= 0.9;

    const speed = Math.hypot(this.vx, this.vy);
    const maxSpeed = 270;
    if (speed > maxSpeed) {
      this.vx = (this.vx / speed) * maxSpeed;
      this.vy = (this.vy / speed) * maxSpeed;
    }

    this.x += this.vx * dt;
    this.y += this.vy * dt;

    if (speed > 8) this.angle = Math.atan2(this.vy, this.vx);

    this.updateAudio(faceCenter, canvasSize.width);
  }

  draw(c) {
    if (this.state === "exploded") return;
    c.save();
    c.translate(this.x, this.y);
    c.rotate(this.angle);
    c.scale(this.scale, this.scale);
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

function averagePoint(lm, idxs) {
  let x = 0,
    y = 0;
  for (const i of idxs) {
    x += lm[i].x;
    y += lm[i].y;
  }
  return { x: x / idxs.length, y: y / idxs.length };
}

let flies = [];
let faceRegion = null; // {x0,y0,x1,y1} in canvas pixel space, mirrored
const prevHandPoints = new Map(); // hand slot index -> array of {x,y} tracked points

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
  const events = [];
  if (!handResult || !handResult.landmarks) return events;

  const seenSlots = new Set();
  const hitFlies = new Set();

  handResult.landmarks.forEach((lm, i) => {
    const rawPoints = [averagePoint(lm, [0, 5, 9, 13, 17]), lm[8], lm[12]];
    const points = rawPoints.map((p) => ({ x: w - p.x * w, y: p.y * h }));

    seenSlots.add(i);
    const prevPoints = prevHandPoints.get(i);
    prevHandPoints.set(i, points);
    if (!prevPoints || dt <= 0) return;

    for (let pIdx = 0; pIdx < points.length; pIdx++) {
      const cur = points[pIdx];
      const prev = prevPoints[pIdx];
      if (!prev) continue;

      const dist = Math.hypot(cur.x - prev.x, cur.y - prev.y);
      const speed = dist / dt;
      const isSwipe = speed >= HIT_SWIPE_SPEED;
      const radius = isSwipe ? HIT_SWIPE_RADIUS : HIT_TOUCH_RADIUS;

      for (const fly of flies) {
        if (fly.state !== "alive" || hitFlies.has(fly)) continue;
        const d = pointToSegmentDistance(fly.x, fly.y, prev.x, prev.y, cur.x, cur.y);
        if (d < radius) {
          hitFlies.add(fly);
          const midX = (prev.x + cur.x) / 2;
          const midY = (prev.y + cur.y) / 2;
          const dirX = isSwipe ? cur.x - prev.x : fly.x - midX || 1;
          const dirY = isSwipe ? cur.y - prev.y : fly.y - midY;
          events.push({ fly, dirX, dirY });
        }
      }
    }
  });

  for (const key of [...prevHandPoints.keys()]) {
    if (!seenSlots.has(key)) prevHandPoints.delete(key);
  }

  return events;
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
  masterGain = audioCtx.createGain();
  masterGain.gain.value = soundEnabled ? 1 : 0;
  masterGain.connect(audioCtx.destination);

  playSlap = createSlapSound(audioCtx, masterGain);
  playPop = createPopSound(audioCtx, masterGain);

  const onBorderHit = (x, y) => {
    spawnExplosion(x, y, 26, "#ff8a5c");
    playPop();
  };

  const canvasSize = { width: canvas.width, height: canvas.height };
  flies = Array.from(
    { length: FLY_COUNT },
    () => new Fly(canvasSize, audioCtx, masterGain, onBorderHit)
  );

  overlay.classList.add("hidden");

  let lastTime = performance.now();

  function frame() {
    const now = performance.now();
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    const w = canvas.width;
    const h = canvas.height;
    const canvasSize = { width: w, height: h };

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
    const faceCenter = {
      x: (region.x0 + region.x1) / 2,
      y: (region.y0 + region.y1) / 2,
    };

    const hitEvents = processHands(handResult, w, h, dt);
    for (const evt of hitEvents) {
      if (evt.fly.hit(evt.dirX, evt.dirY)) {
        spawnExplosion(evt.fly.x, evt.fly.y, 14, "#fff3c4");
        if (soundEnabled) playSlap();
        registerHit(now);
      }
    }
    maybeExpireCombo(now);

    for (const fly of flies) {
      fly.update(dt, region, faceCenter, canvasSize);
      fly.draw(ctx);
    }

    updateParticles(dt);
    drawParticles(ctx);

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
