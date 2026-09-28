/*
  6-Axis Robot Arm Controller
  Arduino Uno + PCA9685 16-Channel PWM Servo Driver
  Joints 0-2: SG90  (light-duty, e.g. wrist/gripper)
  Joints 3-5: MG995 (high-torque, e.g. base/shoulder/elbow)

  WIRING
  ------
  PCA9685        Arduino Uno
  VCC       ->   5V   (logic power)
  GND       ->   GND
  SDA       ->   A4
  SCL       ->   A5
  V+        ->   External 5-6V supply, current-rated for all 6 servos
                 (SG90 ~ 100-250mA each, MG995 ~ 500-900mA stall each,
                  so budget at least 3-4A on the external rail)
  GND (V+ terminal block) -> External supply GND
  IMPORTANT: Arduino GND, PCA9685 GND, and external supply GND must all
  be tied together (common ground), but do NOT power servos from the
  Arduino 5V pin - it cannot supply enough current.

  Servo channels on PCA9685:
    CH0 = SG90  #1   (e.g. wrist roll)
    CH1 = SG90  #2   (e.g. wrist pitch)
    CH2 = SG90  #3   (e.g. gripper)
    CH3 = MG995 #1   (e.g. base rotate)
    CH4 = MG995 #2   (e.g. shoulder)
    CH5 = MG995 #3   (e.g. elbow)

  SERIAL PROTOCOL (from PC / Web Serial / ROS2 bridge)
  ------------------------------------------------------
  Send a line of 6 comma-separated integers (0-180), newline terminated:
      "90,45,90,120,60,90\n"
  representing target angles for joints 0..5 in order.
  The arm will smoothly interpolate from its current pose to the target.

  Optional commands:
    "HOME\n"   -> moves all joints to their configured home angle
    "SPEED,n\n"-> sets interpolation step delay in ms (higher = slower/smoother)

  Requires: Adafruit PWM Servo Driver Library
  (Arduino IDE -> Library Manager -> search "Adafruit PWM Servo Driver")
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define NUM_SERVOS 6
#define SERVO_FREQ 50 // analog servos run at ~50 Hz updates

// ---- Per-servo calibration -----------------------------------------
// Pulse length counts (out of 4096 @ 50Hz) corresponding to 0 deg and
// 180 deg. Datasheet-typical values are given below, but SG90/MG995
// units vary unit-to-unit - recalibrate each servo for best accuracy
// (see calibrate() helper at the bottom of this file).
//
//                    channel:   0(SG90) 1(SG90) 2(SG90) 3(MG995) 4(MG995) 5(MG995)
int SERVO_MIN[NUM_SERVOS] = {  102,    102,    102,     105,     105,     105 }; // 0 deg
int SERVO_MAX[NUM_SERVOS] = {  512,    512,    512,     500,     500,     500 }; // 180 deg

// Software joint limits (deg) - protects mechanism / wiring from over-travel
int JOINT_MIN[NUM_SERVOS] = {   0,   0,   0,   0,  10,   0 };
int JOINT_MAX[NUM_SERVOS] = { 180, 180, 180, 180, 170, 180 };

// Home / startup pose (deg)
int HOME_POSE[NUM_SERVOS] = { 90, 90, 90, 90, 90, 90 };

// ---- State -----------------------------------------------------------
float currentAngle[NUM_SERVOS];   // current commanded angle (float for smooth stepping)
int   targetAngle[NUM_SERVOS];
int   stepDelayMs = 15;           // ms between interpolation steps (SPEED command tunes this)
const float STEP_SIZE = 1.0;      // degrees per interpolation step

String inputLine = "";

int angleToPulse(int channel, float angleDeg) {
  angleDeg = constrain(angleDeg, JOINT_MIN[channel], JOINT_MAX[channel]);
  long pulse = map((long)(angleDeg * 10), 0, 1800,
                    SERVO_MIN[channel], SERVO_MAX[channel]);
  return (int)pulse;
}

void writeAngle(int channel, float angleDeg) {
  pwm.setPWM(channel, 0, angleToPulse(channel, angleDeg));
}

void setup() {
  Serial.begin(115200);
  Serial.println(F("6-Axis Arm Controller ready."));
  Serial.println(F("Send: a0,a1,a2,a3,a4,a5  (angles 0-180)"));

  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);
  delay(200);

  for (int i = 0; i < NUM_SERVOS; i++) {
    currentAngle[i] = HOME_POSE[i];
    targetAngle[i]  = HOME_POSE[i];
    writeAngle(i, currentAngle[i]);
  }
}

void goHome() {
  for (int i = 0; i < NUM_SERVOS; i++) targetAngle[i] = HOME_POSE[i];
}

void parseLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line.equalsIgnoreCase("HOME")) {
    goHome();
    Serial.println(F("OK HOME"));
    return;
  }

  if (line.startsWith("SPEED,")) {
    int val = line.substring(6).toInt();
    if (val > 0 && val < 200) {
      stepDelayMs = val;
      Serial.print(F("OK SPEED="));
      Serial.println(stepDelayMs);
    }
    return;
  }

  // Expect 6 comma separated integers
  int vals[NUM_SERVOS];
  int idx = 0;
  int start = 0;
  for (int i = 0; i <= line.length() && idx < NUM_SERVOS; i++) {
    if (i == line.length() || line.charAt(i) == ',') {
      String token = line.substring(start, i);
      vals[idx] = token.toInt();
      idx++;
      start = i + 1;
    }
  }

  if (idx == NUM_SERVOS) {
    for (int i = 0; i < NUM_SERVOS; i++) {
      targetAngle[i] = constrain(vals[i], JOINT_MIN[i], JOINT_MAX[i]);
    }
    Serial.println(F("OK"));
  } else {
    Serial.println(F("ERR expected 6 comma-separated angles"));
  }
}

void loop() {
  // --- read serial ---
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      parseLine(inputLine);
      inputLine = "";
    } else if (c != '\r') {
      inputLine += c;
    }
  }

  // --- smoothly step every joint toward its target ---
  bool moving = false;
  for (int i = 0; i < NUM_SERVOS; i++) {
    float diff = targetAngle[i] - currentAngle[i];
    if (abs(diff) > 0.01) {
      moving = true;
      float step = constrain(diff, -STEP_SIZE, STEP_SIZE);
      currentAngle[i] += step;
      writeAngle(i, currentAngle[i]);
    }
  }

  if (moving) delay(stepDelayMs);
}

/*
  ---------------------------------------------------------------------
  CALIBRATION HELPER (optional)
  ---------------------------------------------------------------------
  SG90 and MG995 units vary. To calibrate a channel precisely:
  1. Temporarily replace loop() with a sketch that sweeps pwm.setPWM()
     from ~100 to ~600 for the target channel while you watch the horn.
  2. Note the pulse count where the horn just reaches mechanical 0 deg
     and just reaches 180 deg (or its rated max, MG995 is often ~120-180
     deg depending on model) without straining.
  3. Enter those counts into SERVO_MIN[]/SERVO_MAX[] for that channel.
  Doing this per-servo materially improves accuracy versus the generic
  datasheet numbers above.
*/
