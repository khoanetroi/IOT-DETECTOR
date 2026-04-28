/*
 * ESP32-CAM - IoT Camera Simulator
 * =================================
 * Vai trò: Đóng vai Camera an ninh IoT (giống SamsungCamera, BelkinCamera)
 * Hành vi: Chụp ảnh liên tục, gửi gói tin lớn (~10-40KB) dồn dập → traffic bùng nổ
 * 
 * Cài đặt trong Arduino IDE:
 *   1. Board Manager: ESP32 (https://dl.espressif.com/dl/package_esp32_index.json)
 *   2. Board: AI Thinker ESP32-CAM
 *   3. Partition Scheme: Huge APP (3MB No OTA)
 * 
 * ĐỔI WiFi SSID + PASSWORD trước khi nạp!
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

// ─── CẤU HÌNH WiFi ────────────────────────────────
const char* WIFI_SSID     = "TEN_WIFI_CUA_BAN";      // ← ĐỔI
const char* WIFI_PASSWORD = "MAT_KHAU_WIFI";          // ← ĐỔI

// ─── CẤU HÌNH SERVER ──────────────────────────────
const char* SERVER_URL = "http://192.168.1.100:8000/api/image";  // ← ĐỔI IP laptop

// ─── CẤU HÌNH THỜI GIAN ───────────────────────────
#define CAPTURE_INTERVAL 2000  // Chụp ảnh mỗi 2 giây
                                // Camera thật gửi liên tục → traffic dồn dập

// ─── CẤU HÌNH CAMERA (AI Thinker ESP32-CAM) ───────
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

unsigned long lastCaptureTime = 0;
int frameCount = 0;

void initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size   = FRAMESIZE_VGA;    // 640x480 → ảnh ~20-40KB
    config.jpeg_quality = 12;
    config.fb_count     = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init FAILED: 0x%x\n", err);
        return;
    }
    Serial.println("Camera init OK!");
}

void setup() {
    Serial.begin(115200);
    Serial.println("\n=== IoT Camera (ESP32-CAM) ===");

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

    // Khởi tạo Camera
    initCamera();
}

void loop() {
    unsigned long now = millis();

    if (now - lastCaptureTime >= CAPTURE_INTERVAL) {
        lastCaptureTime = now;
        frameCount++;

        // Chụp ảnh
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("[ERROR] Chup anh that bai!");
            return;
        }

        Serial.printf("[FRAME %d] Size: %d bytes\n", frameCount, fb->len);

        // Gửi ảnh JPEG lên server qua HTTP POST
        // Traffic đặc trưng Camera: gói lớn (10-40KB), gửi liên tục
        if (WiFi.status() == WL_CONNECTED) {
            HTTPClient http;
            http.begin(SERVER_URL);
            http.addHeader("Content-Type", "image/jpeg");
            
            int httpCode = http.POST(fb->buf, fb->len);
            Serial.printf("[RESPONSE] Code: %d\n", httpCode);
            
            http.end();
        }

        esp_camera_fb_return(fb);
    }
}
