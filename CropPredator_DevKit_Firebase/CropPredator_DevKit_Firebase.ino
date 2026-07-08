/*
 * Smart Crop Predator Detection — ESP32 DevKit
 * =============================================
 * Reads detection result TWO ways (whichever arrives first):
 *   1. USB Serial — Python sends "MONKEY\n" directly
 *   2. Firebase   — Python pushes to /crop_predator/latest
 *                   DevKit polls it every 2 seconds
 *
 * Triggers species-specific deterrents:
 *   monkey / wild_boar → relay + buzzer + red LED (6 sec)
 *   locust             → buzzer pulses + green LED
 *   human              → two short beeps only
 *
 * Sensors reported to PC via Serial JSON every 10 seconds:
 *   DHT22 (temp, humidity), HC-SR04 (distance),
 *   Soil moisture, MQ135 gas
 */

#include <DHT.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <FirebaseESP32.h>
#include <ArduinoJson.h>

// ── WiFi & Firebase ───────────────────────────────────────────
#define WIFI_SSID     "Ankit's M56"
#define WIFI_PASSWORD "12345678"
#define FIREBASE_HOST "https://crop-protection-6f8ca-default-rtdb.asia-southeast1.firebasedatabase.app/"
#define FIREBASE_AUTH "YOUR_FIREBASE_DATABASE_SECRET"

// ── Pins ──────────────────────────────────────────────────────
#define DHT_PIN     4
#define TRIG_PIN    5
#define ECHO_PIN    18
#define SOIL_PIN    34
#define MQ135_PIN   35
#define PIR_PIN     13
#define RELAY_PIN   26
#define BUZZER_PIN  27
#define RED_LED     25
#define GREEN_LED   32
// ── OLED ──────────────────────────────────────────────────────
Adafruit_SSD1306 oled(128, 64, &Wire, -1);

// ── DHT22 ─────────────────────────────────────────────────────
DHT dht(DHT_PIN, DHT22);

// ── Firebase ──────────────────────────────────────────────────
FirebaseData   fbData;
FirebaseAuth   fbAuth;
FirebaseConfig fbConfig;

// ── State ─────────────────────────────────────────────────────
String  incomingSerial  = "";
String  lastClass       = "";
unsigned long lastSensorMs    = 0;
unsigned long lastFirebaseMs  = 0;
unsigned long lastActionMs    = 0;
bool    wifiOK          = false;
bool    firebaseOK      = false;

#define SENSOR_INTERVAL_MS    10000   // send sensors every 10s
#define FIREBASE_POLL_MS       2000   // poll Firebase every 2s
#define ACTION_COOLDOWN_MS    25000   // ignore repeated triggers for 25s

// =============================================================================
void setup() {
  Serial.begin(115200);

  pinMode(RELAY_PIN,  OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_LED,    OUTPUT);
  pinMode(GREEN_LED,  OUTPUT);
  pinMode(PIR_PIN,    INPUT);
  pinMode(TRIG_PIN,   OUTPUT);
  pinMode(ECHO_PIN,   INPUT);

  // Ensure all outputs are LOW
  digitalWrite(RELAY_PIN,  LOW);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(RED_LED,    LOW);
  digitalWrite(GREEN_LED,  LOW);

  dht.begin();
  Wire.begin(OLED_SDA, OLED_SCL);

  if (oled.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    oledShow("Booting...", "", "");
  }

  // Connect WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  oledShow("WiFi connecting", WIFI_SSID, "");

  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 20) {
    delay(500);
    Serial.print(".");
    tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiOK = true;
    Serial.println("\nWiFi OK: " + WiFi.localIP().toString());
    oledShow("WiFi OK", WiFi.localIP().toString().c_str(), "");

    // Init Firebase
    fbConfig.host = FIREBASE_HOST;
    fbConfig.signer.tokens.legacy_token = FIREBASE_AUTH;
    Firebase.begin(&fbConfig, &fbAuth);
    Firebase.reconnectWiFi(true);
    firebaseOK = true;
    Serial.println("Firebase: connected");
  } else {
    Serial.println("\nWiFi FAILED — Firebase disabled");
    oledShow("WiFi FAILED", "Serial mode only", "");
  }

  delay(800);
  oledShow("READY", "Monitoring...", "");
  Serial.println("[DevKit] Ready");
}

// =============================================================================
void loop() {
  unsigned long now = millis();

  // ── 1. Read class label from PC over USB Serial ────────────────────────────
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      incomingSerial.trim();
      // Ignore JSON sensor ACK and empty strings
      if (incomingSerial.length() > 0 && !incomingSerial.startsWith("{")) {
        Serial.println("[DevKit] Serial received: " + incomingSerial);
        handleDetection(incomingSerial);
      }
      incomingSerial = "";
    } else {
      incomingSerial += c;
    }
  }

  // ── 2. Poll Firebase for new detection ────────────────────────────────────
  if (firebaseOK && (now - lastFirebaseMs) > FIREBASE_POLL_MS) {
    lastFirebaseMs = now;
    pollFirebase();
  }

  // ── 3. Send sensor data to PC every 10 seconds ────────────────────────────
  if ((now - lastSensorMs) > SENSOR_INTERVAL_MS) {
    lastSensorMs = now;
    sendSensorData();
  }
}

// =============================================================================
// Poll Firebase /crop_predator/latest and act if action == "TRIGGER"
// =============================================================================
void pollFirebase() {
  if (Firebase.getString(fbData, "/crop_predator/latest/action")) {
    String action = fbData.stringData();
    action.trim();

    if (action == "TRIGGER") {
      // Get the class
      if (Firebase.getString(fbData, "/crop_predator/latest/class")) {
        String cls = fbData.stringData();
        cls.toUpperCase();
        Serial.println("[Firebase] Trigger: " + cls);
        handleDetection(cls);

        // Mark as done so we don't trigger again
        Firebase.setString(fbData, "/crop_predator/latest/action", "DONE");
      }
    }
  }
}

// =============================================================================
// Handle a detection — show on OLED, trigger deterrent
// =============================================================================
void handleDetection(String cls) {
  unsigned long now = millis();

  // Cooldown check — don't trigger same class twice in 25 seconds
  if (cls == lastClass && (now - lastActionMs) < ACTION_COOLDOWN_MS) {
    Serial.println("[DevKit] Cooldown active — skipping");
    return;
  }

  lastClass    = cls;
  lastActionMs = now;

  oledShow("DETECTED:", cls.c_str(), "Triggering...");
  triggerDeterrent(cls);
  oledShow("Done", ("Last: " + cls).c_str(), "Monitoring...");
}

// =============================================================================
// Species-specific deterrent logic
// =============================================================================
void triggerDeterrent(String cls) {
  Serial.println("[DevKit] Deterrent for: " + cls);

  if (cls == "MONKEY" || cls == "WILD_BOAR") {
    // Aggressive — relay (sprinkler) + buzzer + red LED for 6 seconds
    digitalWrite(RED_LED,    HIGH);
    digitalWrite(RELAY_PIN,  HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    delay(6000);
    digitalWrite(RED_LED,    LOW);
    digitalWrite(RELAY_PIN,  LOW);
    digitalWrite(BUZZER_PIN, LOW);

  } else if (cls == "LOCUST") {
    // Rapid buzzer pulses + green LED
    digitalWrite(GREEN_LED, HIGH);
    for (int i = 0; i < 8; i++) {
      digitalWrite(BUZZER_PIN, HIGH); delay(150);
      digitalWrite(BUZZER_PIN, LOW);  delay(100);
    }
    digitalWrite(GREEN_LED, LOW);

  } else if (cls == "HUMAN") {
    // Two short beeps — acknowledge, no deterrent
    for (int i = 0; i < 2; i++) {
      digitalWrite(BUZZER_PIN, HIGH); delay(200);
      digitalWrite(BUZZER_PIN, LOW);  delay(200);
    }
    Serial.println("[DevKit] Human detected — alert only");
  }
}

// =============================================================================
// Read sensors and send as JSON to PC
// =============================================================================
void sendSensorData() {
  float temp  = dht.readTemperature();
  float hum   = dht.readHumidity();
  float dist  = readUltrasonic();
  int   soil  = map(analogRead(SOIL_PIN), 0, 4095, 100, 0);
  int   gas   = analogRead(MQ135_PIN);
  bool  pir   = digitalRead(PIR_PIN) == HIGH;

  // Send JSON to PC
  StaticJsonDocument<200> doc;
  doc["temp"]  = isnan(temp) ? 0 : round(temp * 10) / 10.0;
  doc["hum"]   = isnan(hum)  ? 0 : round(hum  * 10) / 10.0;
  doc["dist"]  = round(dist  * 100) / 100.0;
  doc["soil"]  = soil;
  doc["gas"]   = gas;
  doc["pir"]   = pir;

  String out;
  serializeJson(doc, out);
  Serial.println(out);

  // Also push to Firebase
  if (firebaseOK) {
    FirebaseJson json;
    json.set("temp",  doc["temp"].as<float>());
    json.set("hum",   doc["hum"].as<float>());
    json.set("dist",  doc["dist"].as<float>());
    json.set("soil",  soil);
    json.set("gas",   gas);
    json.set("pir",   pir);
    Firebase.setJSON(fbData, "/crop_predator/sensors", json);
  }

  // Update OLED with sensor readings
  String l1 = "T:" + String(isnan(temp) ? 0 : temp, 0) +
               "C H:" + String(isnan(hum) ? 0 : hum, 0) + "%";
  String l2 = "D:" + String(dist, 1) + "m S:" + String(soil) + "%";
  String l3 = pir ? "PIR:Motion!" : "PIR:Clear";
  oledShow(l1.c_str(), l2.c_str(), l3.c_str());
}

// =============================================================================
// HC-SR04 distance reading (returns cm)
// =============================================================================
float readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long dur = pulseIn(ECHO_PIN, HIGH, 30000);
  return dur * 0.0343f / 2.0f;
}

// =============================================================================
// OLED helper
// =============================================================================
void oledShow(const char* l1, const char* l2, const char* l3) {
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
