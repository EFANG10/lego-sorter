/*
 * LEGO Brick Sorter - Multi-Gate Arduino Controller
 * Communicates with the Autonomous LEGO Sorter vision pipeline over USB Serial.
 *
 * Commands:
 *   BIN:<n>\n   - Actuate gate for physical bin <n> (0 to NUM_BINS-1)
 *                 Bin -1 is overflow/reject (pass-through, no gate actuated)
 *   PING\n      - Reply PONG for connection handshake
 *   TEST:<n>\n  - Manually test-fire gate <n>
 */

#include <Arduino.h>
#include <Servo.h>

const long BAUD_RATE = 115200;
const int NUM_BINS = 6; // Configurable number of physical gate bins (0 to 5)

// Pin mappings for Servo gates or Solenoid relays
// Digital pins D2 through D7
const int GATE_PINS[NUM_BINS] = {2, 3, 4, 5, 6, 7};

// Actuation angles for standard 90/180 deg servos
const int SERVO_REST_ANGLE = 0;      // Gate closed (brick passes down main belt)
const int SERVO_DEFLECT_ANGLE = 65;  // Gate open (diverts brick into bin)

// Actuation duration in milliseconds
const unsigned long GATE_PULSE_MS = 400;

Servo gates[NUM_BINS];
unsigned long gateCloseTimes[NUM_BINS];
bool gateActive[NUM_BINS];

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial) { ; } // Wait for serial port

  for (int i = 0; i < NUM_BINS; i++) {
    gates[i].attach(GATE_PINS[i]);
    gates[i].write(SERVO_REST_ANGLE);
    gateActive[i] = false;
    gateCloseTimes[i] = 0;
  }

  Serial.println("READY: LEGO Sorter Gate Controller Online");
}

void triggerGate(int binId) {
  if (binId == -1) {
    // Overflow / Reject Bin: brick passes unobstructed to end of conveyor
    Serial.println("ACK: BIN:-1 (Pass-through to end of belt)");
    return;
  }

  if (binId >= 0 && binId < NUM_BINS) {
    gates[binId].write(SERVO_DEFLECT_ANGLE);
    gateActive[binId] = true;
    gateCloseTimes[binId] = millis() + GATE_PULSE_MS;
    Serial.print("ACK: BIN:");
    Serial.print(binId);
    Serial.println(" OPEN");
  } else {
    Serial.print("ERR: Invalid Bin ID ");
    Serial.println(binId);
  }
}

void loop() {
  unsigned long now = millis();

  // Check and close gates whose pulse time has expired (non-blocking)
  for (int i = 0; i < NUM_BINS; i++) {
    if (gateActive[i] && now >= gateCloseTimes[i]) {
      gates[i].write(SERVO_REST_ANGLE);
      gateActive[i] = false;
      Serial.print("ACK: BIN:");
      Serial.print(i);
      Serial.println(" CLOSED");
    }
  }

  // Read incoming serial commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() == 0) return;

    if (cmd == "PING") {
      Serial.println("PONG");
    } 
    else if (cmd.startsWith("BIN:")) {
      int binId = cmd.substring(4).toInt();
      triggerGate(binId);
    } 
    else if (cmd.startsWith("TEST:")) {
      int binId = cmd.substring(5).toInt();
      triggerGate(binId);
    } 
    else {
      Serial.print("ERR: Unknown Command: ");
      Serial.println(cmd);
    }
  }
}
