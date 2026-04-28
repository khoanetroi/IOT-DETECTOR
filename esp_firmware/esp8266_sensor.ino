/*
 * ESP8266 + DHT11 - IoT Sensor Simulator
 * =======================================
 * Vai trò: Đóng vai thiết bị cảm biến IoT (giống NestProtect, AwairAirQuality)
 * Hành vi: Gửi gói tin nhỏ (~50-100 bytes) mỗi 30 giây → tạo traffic pattern đặc trưng của Sensor
 * 
 * Cài đặt trong Arduino IDE:
 *   1. Board Manager: ESP8266 (http://arduino.esp8266.com/stable/package_esp8266com_index.json)
 *   2. Library: DHT sensor library (by Adafruit)
 *   3. Library: ESP8266WiFi (có sẵn)
 *   4. Library: ESP8266HTTPClient (có sẵn)
 * 
 * ĐỔI WiFi SSID + PASSWORD trước khi nạp!
 */

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <DHT.h>

// ─── CẤU HÌNH WiFi ────────────────────────────────
const char* WIFI_SSID     = "TEN_WIFI_CUA_BAN";      // ← ĐỔI
const char* WIFI_PASSWORD = "MAT_KHAU_WIFI";          // ← ĐỔI

// ─── CẤU HÌNH SERVER (IP máy laptop chạy AI) ──────
const char* SERVER_URL = "http://192.168.1.100:8000/api/data";  // ← ĐỔI IP laptop

// ─── CẤU HÌNH DHT11 ───────────────────────────────
#define DHTPIN D4          // Chân data của DHT11 nối vào D4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// ─── CẤU HÌNH THỜI GIAN ───────────────────────────
#define SEND_INTERVAL 30000  // Gửi mỗi 30 giây (30000ms)
                              // Cảm biến thật gửi rất thưa → đặc trưng nhận diện

unsigned long lastSendTime = 0;

void setup() {
    Serial.begin(115200);
    Serial.println("\n=== IoT Sensor (ESP8266 + DHT11) ===");

    // Khởi tạo DHT11
    dht.begin();

    // Kết nối WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Dang ket noi WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Da ket noi! IP: ");
    Serial.println(WiFi.localIP());
}

void loop() {
    unsigned long now = millis();

    if (now - lastSendTime >= SEND_INTERVAL) {
        lastSendTime = now;

        // Đọc cảm biến
        float temp = dht.readTemperature();
        float humi = dht.readHumidity();

        if (isnan(temp) || isnan(humi)) {
            // Nếu DHT11 lỗi, gửi giá trị giả
            temp = 25.0 + random(-20, 20) / 10.0;
            humi = 60.0 + random(-50, 50) / 10.0;
        }

        // Tạo payload JSON nhỏ (~60 bytes) - đặc trưng của sensor
        String payload = "{\"device\":\"sensor\",\"temp\":" + String(temp, 1) 
                        + ",\"humidity\":" + String(humi, 1) 
                        + ",\"uptime\":" + String(millis() / 1000) + "}";

        Serial.print("[SEND] ");
        Serial.println(payload);

        // Gửi HTTP POST
        if (WiFi.status() == WL_CONNECTED) {
            WiFiClient client;
            HTTPClient http;
            http.begin(client, SERVER_URL);
            http.addHeader("Content-Type", "application/json");
            
            int httpCode = http.POST(payload);
            Serial.print("[RESPONSE] Code: ");
            Serial.println(httpCode);
            
            http.end();
        }
    }

    // Không làm gì khác → traffic rất "im lặng" giữa các lần gửi
    // Đây chính là đặc trưng của Sensor IoT
}
