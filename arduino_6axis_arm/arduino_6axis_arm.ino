/*
 * 6-AXIS ROBOT ARM CONTROLLER
 * -----------------------------------------------------------
 * Arduino Uno + PCA9685 16-channel PWM/Servo driver (I2C)
 * 3x MG995 (high-torque: base, shoulder, elbow)
 * 3x SG90  (light-duty: wrist pitch, wrist roll, gripper)
 *
 * Wiring:
 *   Arduino UNO SDA (A4) -> PCA9685 SDA
 *   Arduino UNO SCL (A5) -> PCA9685 SCL
 *   Arduino UNO 5V       -> PCA9685 VCC (logic only, NOT V+)
 *   Arduino UNO GND      -> PCA9685 GND  (MUST share ground)
 *   PCA9685 V+ / servo rail -> external 5-6V, 3-5A supply
 *     (MG995 alone can pull >1A stalled; do NOT power servos
 *      from the Arduino 5V pin or from USB)
 *
 * Channel map (change PIN_* below if you wire differently):
 *   0 - Base rotation   (MG995)
 *   1 - Shoulder        (MG995)
 *   2 - Elbow           (MG995)
 *   3 - Wrist pitch     (SG90)
 *   4 - Wrist roll      (SG90)
 *   5 - Gripper         (SG90)
 *
 * Serial protocol (115200 baud, newline-terminated ASCII):
 *   P,a0,a1,a2,a3,a4,a5   -> move all 6 joints to these angles (0-180),
 *                            interpolated smoothly
 *   J,ch,angle            -> jog a single channel (0-5) to angle
 *   H                     -> go to home pose
 *   SPD,ms                -> set per-step delay in ms (lower = faster,
 *                            default 15ms). Keep >=10 for MG995 to limit
 *                            current spikes.
 *   ?                     -> print current pose
 *
 * Every accepted command replies "OK" ; unknown commands reply "ERR".
 * Requires: Adafruit PWM Servo Driver Library (Library Manager)
 * -----------------------------------------------------------
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define NUM_SERVOS 6
#define PIN_BASE      0
#define PIN_SHOULDER  1
#define PIN_ELBOW     2
#define PIN_WRIST_PITCH 3
#define PIN_WRIST_ROLL  4
#define PIN_GRIPPER   5

#define SERVO_FREQ 50  // analog servos run at 50 Hz

// Pulse-length calibration per channel, in PCA9685 12-bit ticks
// (4096 ticks per 20ms period @ 50Hz -> 1 tick = 4.88us)
// MG995 datasheet pulse range is wider/stronger than SG90; these are safe
// starting points -- tune per servo if travel looks clipped.
//                                 base  shldr  elbow  wpitch wroll  grip
int SERVO_MIN[NUM_SERVOS] = {     102,   102,   102,   105,   105,   105 }; // ~0 deg
int SERVO_MAX[NUM_SERVOS] = {     512,   512,   512,   500,   500,   500 }; // ~180 deg

// Home pose (degrees). Adjust to a safe, folded starting position for
// your specific arm geometry before powering the servo rail.
int homePose[NUM_SERVOS] = { 90, 90, 90, 90, 90, 90 };

int currentAngle[NUM_SERVOS];
int stepDelayMs = 15;

int angleToTicks(int channel, int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVO_MIN[channel], SERVO_MAX[channel]);
}

void writeAngle(int channel, int angle) {
  pwm.setPWM(channel, 0, angleToTicks(channel, angle));
}

// Move all 6 servos from currentAngle[] to target[] together, one degree
// step at a time, so joints arrive roughly simultaneously and current
// draw stays smooth instead of every servo slamming at once.
void movePoseSmooth(int target[NUM_SERVOS]) {
  int steps = 0;
  for (int i = 0; i < NUM_SERVOS; i++) {
    int d = abs(target[i] - currentAngle[i]);
    if (d > steps) steps = d;
  }
  for (int s = 1; s <= steps; s++) {
    for (int i = 0; i < NUM_SERVOS; i++) {
      int start = currentAngle[i];
      int end = target[i];
      int d = end - start;
      if (d == 0) continue;
      int stepsForThis = abs(d);
      // proportional pacing so all joints finish together
      int a = start + (long)d * s / max(steps, stepsForThis == 0 ? 1 : steps);
      writeAngle(i, a);
    }
    delay(stepDelayMs);
  }
  for (int i = 0; i < NUM_SERVOS; i++) {
    writeAngle(i, target[i]);
    currentAngle[i] = target[i];
  }
}

void jogSmooth(int channel, int target) {
  int start = currentAngle[channel];
  int d = target - start;
  int steps = abs(d);
  for (int s = 1; s <= steps; s++) {
    int a = start + (d * s) / steps;
    writeAngle(channel, a);
    delay(stepDelayMs);
  }
  writeAngle(channel, target);
  currentAngle[channel] = target;
}

void printPose() {
  Serial.print("POSE");
  for (int i = 0; i < NUM_SERVOS; i++) {
    Serial.print(",");
    Serial.print(currentAngle[i]);
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);
  delay(500);

  for (int i = 0; i < NUM_SERVOS; i++) {
    currentAngle[i] = homePose[i];
    writeAngle(i, homePose[i]);
  }
  Serial.println("READY 6AXIS_ARM");
}

String inBuf = "";

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line == "H") {
    movePoseSmooth(homePose);
    Serial.println("OK");
    return;
  }
  if (line == "?") {
    printPose();
    return;
  }
  if (line.startsWith("SPD,")) {
    int v = line.substring(4).toInt();
    if (v >= 2 && v <= 200) {
      stepDelayMs = v;
      Serial.println("OK");
    } else {
      Serial.println("ERR");
    }
    return;
  }
  if (line.startsWith("J,")) {
    // J,ch,angle
    int c1 = line.indexOf(',', 2);
    if (c1 < 0) { Serial.println("ERR"); return; }
    int ch = line.substring(2, c1).toInt();
    int ang = line.substring(c1 + 1).toInt();
    if (ch < 0 || ch >= NUM_SERVOS) { Serial.println("ERR"); return; }
    jogSmooth(ch, constrain(ang, 0, 180));
    Serial.println("OK");
    return;
  }
  if (line.startsWith("P,")) {
    // P,a0,a1,a2,a3,a4,a5
    int target[NUM_SERVOS];
    int idx = 2;
    for (int i = 0; i < NUM_SERVOS; i++) {
      int c = line.indexOf(',', idx);
      String tok = (i == NUM_SERVOS - 1) ? line.substring(idx)
                                          : line.substring(idx, c);
      if (tok.length() == 0) { Serial.println("ERR"); return; }
      target[i] = constrain(tok.toInt(), 0, 180);
      idx = c + 1;
    }
    movePoseSmooth(target);
    Serial.println("OK");
    return;
  }

  Serial.println("ERR");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(inBuf);
      inBuf = "";
    } else if (c != '\r') {
      inBuf += c;
    }
  }
}
