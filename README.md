# Ambulance Traffic Management Demo

## What this starter project does
- Displays a laptop webcam feed on a local website.
- Provides a dashboard with camera status and a traffic-light preview.
- Includes a 10-second manual demo trigger so you can demonstrate the signal workflow before an ambulance model is available.
## Roboflow Model Connected
- Model ID: `ambulance-wmxl5-6rgov/1`
- Workspace: `subhan-kalgond`
- Hosted AI inference is connected directly to Roboflow. When an ambulance appears in the webcam frame, the system detects it, outlines it in green, and turns the traffic signal to **GREEN**.

## Setup (Windows)
1. Install Python 3.10 or 3.11 from https://www.python.org/downloads/ and select "Add Python to PATH".
2. Extract this folder.
3. Open Command Prompt in this folder.
4. Create a virtual environment:
   `python -m venv .venv`
5. Activate it:
   `.venv\\Scripts\\activate`
6. Install dependencies:
   `python -m pip install -r requirements.txt`
7. Run:
   `python app.py`
8. Open http://127.0.0.1:5000 in Chrome or Edge and allow camera access if asked.

## Setup (macOS/Linux)
1. Install Python 3.10 or 3.11.
2. In Terminal, enter this folder.
3. Create and activate a virtual environment:
   `python3 -m venv .venv`
   `source .venv/bin/activate`
4. Install dependencies:
   `python -m pip install -r requirements.txt`
5. Run `python app.py`.
6. Open http://127.0.0.1:5000.

## Test the demo
- Check that the webcam feed appears.
- Click "Trigger ambulance event (10 sec)": the website signal preview should turn green.
- Click "Reset demo" to reset.
- The demo trigger validates the dashboard flow only; it does not test AI recognition.

## Add an ambulance detection model
1. Obtain/train a model on a properly licensed dataset containing ambulance images from varied distances, lighting, and angles.
2. The model must use the class name `ambulance`.
3. Place the model weights at `ambulance.pt` in this folder, or set `MODEL_PATH` to the model file path.
4. Install the optional package: `python -m pip install ultralytics`.
5. Restart `python app.py`.
6. Test with a range of camera images and record false positives, missed detections, and latency.

## Physical prototype parts
- Arduino Uno or ESP32 development board
- Red, yellow, and green LEDs
- 3 × 220–330 ohm resistors
- Breadboard and jumper wires
- USB cable for the board
- Optional cardboard/foam-board road intersection

Upload `arduino/traffic_light_model.ino` using Arduino IDE. This sketch listens for R/Y/G over serial. The supplied web starter does not yet send commands to the board; hardware integration is the next development step. A USB serial bridge and a tested safe signal-state controller must be added before the LEDs respond to camera events.

## Safety
This is an educational low-voltage prototype. Never connect it to a real public-road traffic controller. Real systems need authorization, fail-safe hardware, conflict monitoring, secure communications, and a safe transition policy. A camera alone cannot prove an ambulance is 800–1,000 m away; range needs optical, environmental, and field validation.
