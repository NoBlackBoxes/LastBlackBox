# AI-Workshops : See and Speak

Demo 3: local Coral object detection with native NB3 browser video, confidence
boxes and stable terminal responses. Use `--silent` in shared workspaces.

## Speech restored and verified — November 16

The startup failure was a missing `ELEVENLABS_API_KEY`, followed by a missing
ElevenLabs SDK in `_tmp/LBB`. The original generic exception handler obscured
that cause. Current Coral service, camera and MAX98357A hardware all passed;
Coral was not reinstalled or reconfigured.

The ElevenLabs SDK was installed in the existing `_tmp/LBB` environment.
Speech now imports ElevenLabs directly and reuses the audio workshop's voice,
model and WAV conversion without importing its unrelated OpenAI module.
The key is loaded from `01_audio/.env` (Git-ignored), the process environment,
or `--env-file PATH` / `NB3_ENV_FILE`. Never pass a key as a command argument.
The speaker uses the stable name `plughw:CARD=MAX98357A,DEV=0`, currently card 3,
device 0; `NB3_ALSA_DEVICE` can override it. Failures print credential-redacted
tracebacks and record them in `_resources/python/last_error.log`.

Launch the full demonstration **with speech and live video**:

```bash
cd ~/NoBlackBoxes/LastBlackBox
_tmp/LBB/bin/python -B course/versions/ai-workshops/03_see_and_speak/_resources/python/main.py --stream --confidence 0.5
```

Open **http://quentin-NB3.local:8765**. Append `--silent` whenever audio should
be disabled. Press Ctrl+C to release the camera, HTTP sockets and Coral client.

Current real tests passed:

- Camera capture, live Coral inference and bottle detections at 87–91%.
- MAX98357A test tone; user confirmed physical audibility.
- `--say "I see a banana!"`: ElevenLabs HTTP 200, WAV conversion and ALSA playback.
- Full camera → Coral → “I see a bottle!” → ElevenLabs → speaker pipeline,
  with simultaneous MJPEG frames and live bounding-box JSON. The run received
  89 valid JPEG frames with maximum gap 0.074 seconds, and used one synthesis.
- User confirmed hearing both spoken phrases. Two paid TTS requests were
  made in total: the standalone speech test and the live detection test.
- Clean shutdown, HTTP port release, camera reopening and fresh Coral inference.
- Existing offline stability/cooldown, latest-frame and native-viewer tests.

The live-test log is `_resources/python/current_full_pipeline.log`. Actual
Mac browser rendering remains a viewing check; HTTP video and detection data
were tested on the current Pi. No OpenAI calls or unnecessary system changes
were made.

## Launch

From the repository root:

```bash
cd ~/NoBlackBoxes/LastBlackBox
_tmp/LBB/bin/python -B course/versions/ai-workshops/03_see_and_speak/_resources/python/main.py --silent --stream --confidence 0.5
```

Open **http://quentin-NB3.local:8765** on the Mac. Use the **Fullscreen** button
or Escape to exit fullscreen. `--port 9000` selects another port. Use NB3's LAN
IP if `.local` name resolution is unavailable. Stop other camera programs
before launching this demo; only one process should own the camera.

The default 800×600 configuration matches `boxes/vision/stream-NB3/stream.py`.
Once this works well on the Mac, test a sharper 1280×960 stream:

```bash
_tmp/LBB/bin/python -B course/versions/ai-workshops/03_see_and_speak/_resources/python/main.py --silent --stream --confidence 0.5 --width 1280 --height 960
```

Higher resolution needs more encoder CPU and network bandwidth. Fullscreen
preserves aspect ratio with letterboxing; it does not create image detail
beyond the selected camera resolution. SVG boxes/text remain sharp at any
window size. The raw MJPEG endpoint is `/stream/camera.mjpg`; overlays are a
separate layer in the viewing page, not burned into JPEGs.

## Reference architecture and changes

| Aspect | Working stream-NB3 | Previous demo | Current demo |
|---|---|---|---|
| Camera | NB3 Camera, 800×600, RGB888 | 640×480, encoder bypassed | Same NB3 Camera/start/MJPEGEncoder path as reference, 800×600 default |
| Video capture | `capture(mjpeg=True)` | Raw array each iteration | `capture(mjpeg=True)` |
| Video encoding | Camera encoder JPEG passed through | OpenCV JPEG quality 75 after inference/annotation | Camera JPEG passed through unchanged |
| Inference | None | Synchronous, blocked video loop | One asynchronous worker, up to 5 inferences/sec |
| Buffering | NB3 Output holds latest encoded JPEG | Latest raw-derived JPEG | NB3 Output plus one latest JPEG for delivery and one pending JPEG for inference |
| HTTP | NB3.Server multipart broadcast | Custom ThreadingHTTPServer | NB3.Server headers/framing/broadcast reused through a thin adapter |
| Browser | Native MJPEG image | JS multipart parsing/JPEG decoding/canvas | Native MJPEG image plus small SVG overlay |
| Cadence | Encoder output, .0333s delivery delay | Explicit 12 fps cap | Same delivery cadence as reference, independent of inference |

The previous path increased latency/CPU by coupling raw capture, Coral,
annotation and re-encoding. It also reduced resolution and added browser-side
JPEG decoding/canvas work. Those differences are plausible causes of the poor
experience; passing HTTP-client tests did not establish Mac browser quality.
The original implementation remains unchanged.

The current `stream.py` subclasses `NB3.Server.Server`, preserving its MJPEG
headers, frame packaging and broadcast. Socket writes run in a separate sender
thread with a two-second client timeout, so slow clients cannot block camera
acquisition or Coral. This is a small isolation layer around the reference
server, not a second video encoding system. A stalled client can still delay
other clients on that sender until its socket times out.

`vision.DetectorWorker` decodes only selected copies of the *same* encoder
JPEGs. It converts BGR to RGB, resizes and batches uint8 input according to
model metadata, then calls the existing `rpi_edgetpu` service `detect()`.
Pending frames are replaced rather than queued. Camera capture is never
performed by the worker or by browser clients.

The browser decodes MJPEG natively. It polls small detection JSON responses
at up to 5 Hz and positions normalized boxes in an SVG matching the video's
aspect ratio. No JS video parser, canvas, bitmap conversion or re-encoding is
used. Boxes can trail moving objects by the inference/poll delay; detections
older than one second are hidden. Camera-producer status and native image
errors drive reconnects; native MJPEG does not expose reliable per-frame
arrival events, so producer status does not prove browser rendering health.

## Incremental tests

Camera-only mode starts no Coral client and imports no speech SDKs:

```bash
_tmp/LBB/bin/python -B course/versions/ai-workshops/03_see_and_speak/_resources/python/main.py --silent --stream --camera-only
```

First compare this page with the original `stream-NB3` on the Mac. Stop it,
then launch the normal silent command to add inference. Compare perceived
smoothness, image detail and latency. If only the second is worse, terminal
metrics report video FPS and inference time independently.

Reproduce the hardware comparison and offline regressions:

```bash
_tmp/LBB/bin/python -B course/versions/ai-workshops/03_see_and_speak/_resources/python/test_robustness.py
_tmp/LBB/bin/python -B course/versions/ai-workshops/03_see_and_speak/_resources/python/test_viewer.py
```

The hardware test runs the original reference, then camera-only, then a
five-minute Coral phase, sequentially. It writes `stream_comparison.json` and
per-phase logs beside the Python scripts. It measures real camera/encoder
output over HTTP, CPU time, frame gaps, bandwidth and memory; it does not
claim that a Mac browser was tested. `test_viewer.py` now checks native-video
markup, unchanged JPEG pass-through, preprocessing/labels, stability/cooldown,
and bounded asynchronous inference. Earlier custom-canvas tests/logs are
historical and do not validate this architecture.

## NB3's local robot personality

Add `--humor` for playful, family-friendly responses. Without it the original
“I see a bottle!” behaviour is unchanged. No LLM, network or new dependency
is used to select jokes; only the existing ElevenLabs speech call is made
when speech is enabled. `--say TEXT` always speaks the supplied text literally.

Speech and live streaming:

```bash
cd ~/NoBlackBoxes/LastBlackBox
_tmp/LBB/bin/python -B course/versions/ai-workshops/03_see_and_speak/_resources/python/main.py --stream --humor --confidence 0.5
```

Open **http://quentin-NB3.local:8765**. Append `--only banana` or `--only bottle`
to restrict spoken subjects while keeping every detection visible. To rehearse
without audio or API calls, add `--silent`.

The personality is instantiated once per run. After the existing confidence,
stability and cooldown checks, NB3 randomly chooses a line for the selected
object, excluding the last line used for that class. Intervening objects do
not erase that memory. History resets when the program restarts. No joke is
selected during cooldown, and detection/streaming/overlays remain unchanged.

The complete editable collection is in `_resources/python/personality.py`.
Each object has three lines:

Validation: 11 regression tests passed (including CLI speech-import blocking,
matching labels, variety/per-object history, `--only`, stability/cooldown and
plain-mode compatibility). A live `--silent --humor --stream` run produced
three bottle jokes with no consecutive repeats and the eight-second cooldown.
Actual plain silent mode still said “I see a bottle!”. After those checks, one
real detection-triggered ElevenLabs/MAX98357A test spoke the spare-water/spare-
excuses joke successfully and shut down cleanly. No environments, camera,
Coral, streaming, voice settings or audio configuration were changed. The
local response selector uses no OpenAI or other LLM APIs.

### Banana

- A banana! Its biodegradable carrying case is a nice touch, humans.
- A banana! Curved, yellow, and tragically incompatible with my USB port.
- A banana! My fruit department would like a word, but it's just one very excited spreadsheet.

### Person

- A human! How do you get anything done without wheels or a charging socket?
- Human detected! Are you in charge, or do we both report to the snacks?
- A person! You get fingerprints; I get error messages.

### Bottle

- A bottle! Keep it away from my circuits; I'm funny enough without a short.
- A bottle! Humans carry spare water; I carry spare excuses.
- A bottle! Is that water, or have you finally invented robot tea?

### Cup

- A cup! A tiny bucket with excellent career prospects.
- A cup! Tea, coffee, or the reason nobody is allowed near the lab sink?
- A cup! Finally, a swimming pool for a very ambitious biscuit.

### Book

- A book! A tablet that refuses to run out of battery is deeply suspicious.
- A book! I'd turn the page, but my page-turning hardware is imaginary.
- A book! Thousands of words without a single software update; I respect that.

### Cell Phone

- A cell phone! A tiny rectangle that tells humans what to do is worth studying.
- A cell phone! Does it know you're here, or have you escaped?
- A cell phone! Please tell it I'm the interesting robot in the room.

### Laptop

- A laptop! That's my distant cousin, but we don't discuss the keyboard incident.
- A laptop! So many keys, and it still can't open a door.
- A laptop! We should network professionally, not with a cable.

### Chair

- A chair! Your legs have found their charging station.
- A chair! Four legs and zero ambition is honestly quite inspiring.
- A chair! I would sit down, but my wheels are emotionally attached to the floor.

### Dog

- A dog! Four legs, excellent sensors, and absolutely no respect for cables.
- A dog! Finally, someone who appreciates a good stick-based interface.
- A dog! We both fetch, but my data is considerably less muddy.

### Cat

- A cat! I recognise a fellow expert in ignoring instructions.
- A cat! Please tell your tiny supervisor I said hello.
- A cat! Excellent balance, questionable respect for expensive equipment.

### Apple

- An apple! How does nature keep making snacks without a charging port?
- An apple! Crunchy data storage, and humans insist on eating the evidence.
- An apple! I was promised a byte, but this seems like a trap.

### Teddy Bear

- A teddy bear! Finally, a colleague qualified to handle emotional support.
- A teddy bear! I respect the engineering of that soft exterior and mysterious inner stuffing.
- A teddy bear! Are you the safety officer, or just extremely huggable?

### Other detected objects

The exact detected COCO label replaces `{object}`:

- Object identified: {object}! My official scientific assessment: surprisingly object-shaped.
- Object detected: {object}! Does it come with instructions, or are we improvising?
- Today's discovery: {object}! Another entry for my extremely unofficial science journal.

## Detection and existing speech support

Restrict spoken responses to one object class with `--only`:

```bash
_tmp/LBB/bin/python -B course/versions/ai-workshops/03_see_and_speak/_resources/python/main.py --stream --confidence 0.5 --only bottle
```

Use `--only banana` for bananas, or quote a multiword COCO label such as
`--only "wine glass"`. Class names are case-insensitive; unknown labels are
rejected before opening hardware. Omit `--only` for the original behaviour.

This filters only responses: all qualifying detections remain in the browser
and terminal summaries. The most confident matching object is considered
even if another class has a higher score. Stability counts consecutive
matching detections; absence of the chosen class resets it. Confidence and
cooldown retain their existing settings. `--silent --only bottle` prints only
bottle responses without speech/API calls. `--only` cannot accompany `--say`,
which is a manual speech test rather than a detection-triggered response.

Defaults remain 65% confidence, three consecutive detections of the leading
class, and eight seconds of cooldown. Override with `--confidence`,
`--stable-frames` or `--cooldown`. Terminal summaries are limited to once/sec.
Silent responses appear only after stable *inference results*, never by
counting the same cached result multiple times. For example:

```text
banana: 87%
NB3 (silent): I see a banana!
```

`--silent` imports no speech dependencies, needs no API credentials and makes
no audio/cloud calls. Existing optional speech support is retained for future
use; it is not used in streaming tests. `--silent --say` is rejected. The
pre-existing audio workshop helpers are reused only if speech is explicitly
run without silent mode. No packages or system configuration are changed.

Models/labels remain the local files in `_tmp/coral/models/`; COCO labels
retain all 90 slots, including `n/a`, so banana is ID 51. The Coral service
runs separately and is not shut down when this demo closes its client.
Ctrl+C closes HTTP connections, the camera/encoder and the inference client.

## Validation results

Real hardware tests were run sequentially, with the original/camera-only
comparison completed before the Coral phase:

| Configuration | Duration | Received FPS | Longest frame gap | Demo process CPU (one core) |
|---|---:|---:|---:|---:|
| Original stream-NB3, 800×600 | 20s | 17.04 | 0.081s | 18.1% |
| Camera-only, 800×600 | 20s | 17.49 | 0.081s | 18.7% |
| Async Coral, 800×600 | 310s | 17.80 | 0.113s | 29.3% |
| Camera-only, 1280×960 | 20s | 17.83 | 0.102s | 26.2% |
| Async Coral, 1280×960 | 30s | 18.28 | 0.098s | 39.9% |

The five-minute phase delivered 5,517 valid JPEGs; model results were produced
independently, with the final decode/preprocess/inference measurement 32ms.
It used about 5.14 Mbps for video and 85 MB process RSS. The sharper 1280×960
Coral run used about 13.92 Mbps and 92 MB RSS. These are measurements in this
scene, not a guarantee on the presentation Wi-Fi. CPU figures cover this demo
process, including camera/encoder threads, not the separate inference service.

Inference did not reduce the measured video rate. That supports isolation of
the two paths, but normal run-to-run camera/scene variation explains the small
FPS differences; it is not evidence that inference accelerates video.

Offline regression tests passed JPEG identity, native-image/SVG markup,
preprocessing, COCO mapping, stable responses/cooldown and a deliberately
blocked inference worker. Submitting 99 new frames to that worker remained
nonblocking and retained only the newest pending JPEG. Shutdown released the
HTTP port, and a fresh camera/Coral inference test succeeded afterwards.

Reports: `_resources/python/stream_comparison.json` and
`_resources/python/stream_quality_comparison.json`; per-phase logs are beside
them. No audio, ElevenLabs or OpenAI APIs were used.

**Not yet verified:** actual Mac browser appearance, perceived latency,
fullscreen sharpness and long-run browser/network reliability. These remain
the user's acceptance check. Reload the page after upgrading to discard the
previous canvas viewer. Asynchronous boxes can trail moving objects; they are
hidden when over one second old. Camera/TPU driver failures still require
diagnosis, while a failed inference worker now leaves the video running.
