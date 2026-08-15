# 🪰 Fly Swat Cam

https://github.com/user-attachments/assets/b5c48f7a-4e51-4ad4-a19e-664239a80164

Watch on YouTube: https://youtu.be/7B8ITUgnBp0

Flies buzz around your face in your webcam feed. Swipe your hand through one to
slap it away, with a soft slap sound.

Runs entirely in the browser — no backend, no build step.

## How it works

- **Face tracking**: [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
  tracks your face each frame and defines a region around your head for flies
  to wander in.
- **Hand tracking**: [MediaPipe HandLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  tracks your hand(s). A fast swipe of the palm near a fly counts as a slap.
- **Flies**: drawn on a `<canvas>` with a simple wander/steering behavior so
  they stay near your face and flit around instead of moving in straight
  lines.
- **Sound**: a short "slap" sound is synthesized on the fly with the Web
  Audio API (filtered noise burst) — no audio files needed.

Both models and the WASM runtime load from a CDN (`jsdelivr` /
`storage.googleapis.com`), so an internet connection is required the first
time (results are cached by the browser after that).

## Running locally

Browsers block ES module imports and camera access from `file://` URLs, so
serve the folder over HTTP:

```bash
npx serve .
# or
python -m http.server 8080
```

Then open the printed `http://localhost:...` URL, click **Start camera**,
and allow camera access.

## Deploying

This is static HTML/CSS/JS, so it works as-is on GitHub Pages, Netlify,
Vercel, etc. — just point them at the repo root.

## Notes

- Camera video never leaves your device; all tracking runs locally in the
  browser (WebAssembly + optional WebGL/GPU delegate).
- Tune slap sensitivity in `main.js` via `HIT_TOUCH_RADIUS` (any overlap
  counts as a hit), `HIT_SWIPE_RADIUS`/`HIT_SWIPE_SPEED` (bigger reach when
  the hand is moving fast), and `HAND_TRACK_TIMEOUT` (how long a lost hand
  is still tracked to bridge fast-swipe motion blur).
