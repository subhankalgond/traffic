// LOW-VOLTAGE MODEL ONLY. Do not connect this to real traffic signals.
// LED wiring: each LED anode -> resistor (220-330 ohm) -> pin; cathode -> GND.
const int RED_PIN = 8;
const int YELLOW_PIN = 9;
const int GREEN_PIN = 10;

void setRed() {
  digitalWrite(RED_PIN, HIGH);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
}
void setup() {
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  Serial.begin(9600);
  setRed();
}
void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'R') setRed();
    // Prototype sequence only. For an actual junction, use an engineered
    // all-red clearance and conflict-monitoring system before granting green.
    if (cmd == 'G') {
      digitalWrite(RED_PIN, LOW);
      digitalWrite(YELLOW_PIN, LOW);
      digitalWrite(GREEN_PIN, HIGH);
    }
    if (cmd == 'Y') {
      digitalWrite(RED_PIN, LOW);
      digitalWrite(YELLOW_PIN, HIGH);
      digitalWrite(GREEN_PIN, LOW);
    }
  }
}
