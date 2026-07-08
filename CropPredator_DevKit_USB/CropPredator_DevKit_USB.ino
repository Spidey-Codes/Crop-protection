/*
 * Smart Crop Predator Detection — ESP32 DevKit Sketch (USB Camera version)
 * ─────────────────────────────────────────────────────────────────────────
 * Communication:
 *   DevKit → PC : "MOTION\n"       when PIR fires
 *   DevKit → PC : JSON sensor data  every 30s
 *   PC → DevKit : "monkey\n"        detected class label
 *   PC → DevKit : "NO_DETECTION\n"  low confidence
 *   PC → DevKit : "COOLDOWN\n"      system in cooldown
 *
 * Pin mapping:
 *   GPIO4  → DHT22 DATA (+ 10kΩ pull-up to 3.3V)
 *   GPIO5  → HC-SR04 TRIG
 *   GPIO18 → HC-SR04 ECHO (via voltage divider: 1kΩ+2kΩ)
 *   GPIO34 → Soil moisture AOUT (analog, input only)
 *   GPIO13 → PIR sensor output
 *   GPIO21 → OLED SDA
 *   GPIO22 → OLED SCL
 *   GPIO26 → Relay IN (VCC of relay → VIN/5V)
 *   GPIO27 → Buzzer (via NPN transistor base)
 */

#include <DHT.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <FirebaseESP32.h>
#include <ArduinoJson.h>

// ── WiFi & Firebase ───────────────────────────────────────────────────────────
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#define FIREBASE_HOST "YOUR_PROJECT.firebaseio.com"
#define FIREBASE_AUTH "YOUR_FIREBASE_SECRET"

// ── Pins ──────────────────────────────────────────────────────────────────────
#define DHT_PIN    4
#define DHT_TYPE   DHT22
#define TRIG_PIN   5
#define ECHO_PIN   18
#define SOIL_PIN   34
#define PIR_PIN    13
#define RELAY_PIN  26
#define BUZZER_PIN 27
#define OLED_SDA   21
#define OLED_SCL   22

// ── OLED ──────────────────────────────────────────────────────────────────────
#define OLED_WIDTH  128
#define OLED_HEIGHT 64
#define OLED_ADDR   0x3C

// ── Timings ───────────────────────────────────────────────────────────────────
#define BUZZER_DURATION    5000    // ms
#define RELAY_DURATION     8000    // ms
#define PIR_DEBOUNCE_MS    500     // ignore re-triggers within 500ms
#define SENSOR_REPORT_MS   30000   // send sensor JSON to PC every 30s
#define PC_RESPONSE_TIMEOUT 8000   // wait up to 8s for PC to respond

// ── Objects ───────────────────────────────────────────────────────────────────
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_SSD1306 oled(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);
FirebaseData fbData;
FirebaseAuth fbAuth;
FirebaseConfig fbConfig;

// ── State ─────────────────────────────────────────────────────────────────────
unsigned long lastPIRTime       = 0;
unsigned long lastSensorReport  = 0;
bool          wifiOK            = false;
bool          firebaseOK        = false;
String        lastDetection     = "";

// =============================================================================
void setup() {
  Serial.begin(115200);    // USB Serial — talks to PC Python script

  pinMode(PIR_PIN,    INPUT);
  pinMode(RELAY_PIN,  OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(TRIG_PIN,   OUTPUT);
  pinMode(ECHO_PIN,   INPUT);

  digitalWrite(RELAY_PIN,  LOW);
  digitalWrite(BUZZER_PIN, LOW);

  dht.begin();
  Wire.begin(OLED_SDA, OLED_SCL);

  // OLED
  if (oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    oledPrint("Booting...", "", "");
  }

  // WiFi (optional — for Firebase only)
  oledPrint("Connecting WiFi", WIFI_SSID, "");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500); attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    wifiOK = true;
    Serial.println("[INFO] WiFi connected: " + WiFi.localIP().toString());
    oledPrint("WiFi OK", WiFi.localIP().toString().c_str(), "");

    // Firebase
    fbConfig.host = FIREBASE_HOST;
    fbConfig.signer.tokens.legacy_token = FIREBASE_AUTH;
    Firebase.begin(&fbConfig, &fbAuth);
    Firebase.reconnectWiFi(true);
    firebaseOK = true;
  } else {
    Serial.println("[WARN] WiFi failed — Firebase disabled");
    oledPrint("WiFi FAILED", "PC mode only", "");
  }

  delay(800);
  oledPrint("READY", "Waiting for motion", "");
  Serial.println("[INFO] DevKit ready — PC inference mode");
}

// =============================================================================
void loop() {
  bool pirHigh = digitalRead(PIR_PIN) == HIGH;
  unsigned long now = millis();

  // ── PIR triggered ────────────────────────────────────────────────────────
  if (pirHigh && (now - lastPIRTime) > PIR_DEBOUNCE_MS) {
    lastPIRTime = now;

    Serial.println("[INFO] PIR triggered");
    oledPrint("Motion!", "Waiting for PC...", "");

    // Tell PC to capture & infer
    Serial.println("MOTION");

    // Wait for PC response
    String response = waitForPCResponse(PC_RESPONSE_TIMEOUT);
    response.trim();
    Serial.println("[PC] " + response);

    if (response == "NO_DETECTION") {
      oledPrint("Low confidence", "No action", "");

    } else if (response == "COOLDOWN") {
      oledPrint("Cooldown active", "Try again soon", "");

    } else if (response.length() > 0) {
      // Valid class detected
      lastDetection = response;

      // Read sensors
      float temp     = dht.readTemperature();
      float humidity = dht.readHumidity();
      float dist_m   = readUltrasonic();
      int   soilRaw  = analogRead(SOIL_PIN);
      int   soilPct  = map(soilRaw, 0, 4095, 100, 0);

      // Show on OLED
      String line1 = "DETECTED:";
      String line2 = response;
      line2.toUpperCase();
      String line3 = "T:" + String(temp, 0) + "C H:" + String(humidity, 0) + "%";
      oledPrint(line1.c_str(), line2.c_str(), line3.c_str());

      // Trigger appropriate deterrent
      triggerDeterrent(response);

      // Log to Firebase (sensor data — class already logged by Python)
      if (firebaseOK) {
        logSensorsToFirebase(response, temp, humidity, dist_m, soilPct);
      }

      // Send sensor JSON to PC for its Firebase log
      sendSensorJSON(temp, humidity, dist_m, soilPct);
    }

    delay(1000);
    oledPrint("Monitoring...", lastDetection.length() > 0 ? ("Last: " + lastDetection).c_str() : "", "");
  }

  // ── Periodic sensor report to PC ─────────────────────────────────────────
  if ((now - lastSensorReport) > SENSOR_REPORT_MS) {
    lastSensorReport = now;
    float temp     = dht.readTemperature();
    float humidity = dht.readHumidity();
    float dist_m   = readUltrasonic();
    int   soilPct  = map(analogRead(SOIL_PIN), 0, 4095, 100, 0);
    sendSensorJSON(temp, humidity, dist_m, soilPct);
  }

  delay(50);
}

// =============================================================================
// Wait for a response from PC over Serial
// =============================================================================
String waitForPCResponse(unsigned long timeoutMs) {
  unsigned long start = millis();
  String response = "";
  while (millis() - start < timeoutMs) {
    if (Serial.available()) {
      response = Serial.readStringUntil('\n');
      response.trim();
      if (response.length() > 0) return response;
    }
    delay(10);
  }
  return "TIMEOUT";
}

// =============================================================================
// Deterrent logic — species-specific responses
// =============================================================================
void triggerDeterrent(String className) {
  Serial.println("[INFO] Triggering deterrent for: " + className);

  if (className == "monkey" || className == "wild_boar") {
    // Most destructive — relay (sprinkler) + buzzer together
    digitalWrite(RELAY_PIN,  HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    delay(RELAY_DURATION);
    digitalWrite(RELAY_PIN,  LOW);
    digitalWrite(BUZZER_PIN, LOW);

  } else if (className == "bird") {
    // Buzzer alone — birds scatter easily
    digitalWrite(BUZZER_PIN, HIGH);
    delay(BUZZER_DURATION);
    digitalWrite(BUZZER_PIN, LOW);

  } else if (className == "deer") {
    // Relay (LED strobe) — deer sensitive to sudden light
    digitalWrite(RELAY_PIN, HIGH);
    delay(RELAY_DURATION);
    digitalWrite(RELAY_PIN, LOW);

  } else if (className == "human") {
    // Alert only — short beep, no deterrent
    digitalWrite(BUZZER_PIN, HIGH);
    delay(300);
    digitalWrite(BUZZER_PIN, LOW);
    delay(200);
    digitalWrite(BUZZER_PIN, HIGH);
    delay(300);
    digitalWrite(BUZZER_PIN, LOW);
    Serial.println("[INFO] Human detected — alert only");
  }
}

// =============================================================================
// HC-SR04 ultrasonic distance (returns meters)
// =============================================================================
float readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  return (duration * 0.0343f) / 200.0f;  // cm → m
}

// =============================================================================
// Send sensor reading as JSON to PC
// =============================================================================
void sendSensorJSON(float temp, float humidity, float dist_m, int soilPct) {
  StaticJsonDocument<128> doc;
  doc["temp"]     = isnan(temp)     ? 0 : round(temp * 10) / 10.0;
  doc["humidity"] = isnan(humidity) ? 0 : round(humidity * 10) / 10.0;
  doc["distance"] = round(dist_m * 100) / 100.0;
  doc["soil_pct"] = soilPct;
  String out;
  serializeJson(doc, out);
  Serial.println(out);
}

// =============================================================================
// Firebase: log sensor readings alongside detection
// =============================================================================
void logSensorsToFirebase(String cls, float temp, float humidity,
                          float dist, int soil) {
  if (!Firebase.ready()) return;
  String path = "/crop_predator/sensors/" + String(millis());
  FirebaseJson json;
  json.set("class",    cls);
  json.set("temp",     temp);
  json.set("humidity", humidity);
  json.set("distance", dist);
  json.set("soil_pct", soil);
  if (!Firebase.setJSON(fbData, path, json)) {
    Serial.println("[WARN] Firebase: " + fbData.errorReason());
  }
}

// =============================================================================
// OLED helper
// =============================================================================
void oledPrint(const char* l1, const char* l2, const char* l3) {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);

  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.println("CROP PREDATOR SYSTEM");
  oled.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  oled.setCursor(0, 14); oled.println(l1);
  oled.setCursor(0, 30); oled.println(l2);
  oled.setCursor(0, 46); oled.println(l3);
  oled.display();
}
