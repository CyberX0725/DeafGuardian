#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// 🚨 强制开启乐鑫官方 esp-nn 硬件级 AI 加速，请替换为你导出的实际 AI 库名
#define EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN 1
#include <acoustic_sentry_inferencing.h>

// ================= [ 1. 系统配置总线 ] =================
namespace Config
{
  const char *SSID = "AcousticNet";
  const char *PASSWORD = "12345678";
  const char *TARGET_IP = "192.168.169.1";
  const int TARGET_PORT = 8888;

  const unsigned long DISPLAY_HOLD_MS = 2000;
  const unsigned long HEARTBEAT_MS = 10000;
}

// ================= [ 2. 自适应时间总线 ] =================
namespace AI_Filter
{
  const int WINDOW_SIZE = 2; // 保持2帧的高灵敏度响应窗口
  int write_index = 0;

  float hist_fire[WINDOW_SIZE] = {0};
  float hist_bell[WINDOW_SIZE] = {0};
  float hist_knock[WINDOW_SIZE] = {0};
  float hist_glass[WINDOW_SIZE] = {0};
  float hist_alarm[WINDOW_SIZE] = {0};
  float hist_water[WINDOW_SIZE] = {0};

  // 长时环境基准线
  float base_fire = 0.0f;
  float base_bell = 0.0f;
  float base_knock = 0.0f;
  float base_glass = 0.0f;
  float base_alarm = 0.0f;
  float base_water = 0.0f;

  // 计算短时均值
  float update_and_get_short_avg(float *history, float new_val)
  {
    history[write_index] = new_val;
    float sum = 0;
    for (int i = 0; i < WINDOW_SIZE; i++)
      sum += history[i];
    return sum / (float)WINDOW_SIZE;
  }

  // EMA 跟踪慢漂移底噪
  void update_long_term_baseline(float &base, float current_raw)
  {
    base = (base * 0.98f) + (current_raw * 0.02f);
  }
}

WiFiUDP udp;
uint8_t myMac[6];
unsigned long lastHeartbeatTime = 0;

struct DisplayState
{
  unsigned long triggerTime = 0;
  String lockedSound = "";
  bool isDanger = false;
} screen;

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_SCL 11
#define OLED_SDA 12
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

#define I2S_SCK 5
#define I2S_WS 6
#define I2S_SD 7
#define I2S_PORT I2S_NUM_0

float *inference_buffer = nullptr;
int32_t *raw_i2s_buffer = nullptr;

int get_audio_channels_data(size_t offset, size_t length, float *out_ptr)
{
  memcpy(out_ptr, inference_buffer + offset, length * sizeof(float));
  return 0;
}

void send_protocol_packet(uint8_t event_id, uint8_t confidence)
{
  uint8_t packet[8];
  packet[0] = event_id;
  packet[1] = confidence;
  memcpy(&packet[2], myMac, 6);

  udp.beginPacket(Config::TARGET_IP, Config::TARGET_PORT);
  udp.write(packet, 8);
  udp.endPacket();
  Serial.printf("📡 [UDP 发射成功] Event_ID: 0x%02X, 置信度: %d%%\n", event_id, confidence);
}

void init_network()
{
  display.clearDisplay();
  display.setCursor(0, 5);
  display.println("Connecting WiFi...");
  display.display();
  WiFi.begin(Config::SSID, Config::PASSWORD);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(400);
    Serial.print(".");
  }
  WiFi.macAddress(myMac);
  udp.begin(Config::TARGET_PORT);
  Serial.println("\n[Network] 并网成功。");
}

void init_hardware()
{
  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
    for (;;)
      ;
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 10);
  display.println("Sentry Running...");
  display.display();

  const i2s_config_t i2s_config = {
      .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = EI_CLASSIFIER_FREQUENCY,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
      .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 128,
      .use_apll = false};
  const i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK, .ws_io_num = I2S_WS, .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = I2S_SD};
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_start(I2S_PORT);
}

void setup()
{
  Serial.begin(115200);
  init_hardware();
  init_network();
  inference_buffer = (float *)malloc(EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(float));
  raw_i2s_buffer = (int32_t *)malloc(128 * sizeof(int32_t));
  if (!inference_buffer || !raw_i2s_buffer)
    for (;;)
      ;
  delay(200);
}

void loop()
{
  size_t bytes_read = 0;
  unsigned long now = millis();

  if (now - lastHeartbeatTime >= Config::HEARTBEAT_MS)
  {
    send_protocol_packet(0x10, 0x00);
    lastHeartbeatTime = now;
  }

  // ① I2S 16位有损量程解包
  int i = 0;
  while (i < EI_CLASSIFIER_RAW_SAMPLE_COUNT)
  {
    i2s_read(I2S_PORT, raw_i2s_buffer, 128 * sizeof(int32_t), &bytes_read, portMAX_DELAY);
    int total_samples = bytes_read / sizeof(int32_t);
    for (int j = 0; j < total_samples; j += 2)
    {
      if (i < EI_CLASSIFIER_RAW_SAMPLE_COUNT)
      {
        inference_buffer[i] = (float)(raw_i2s_buffer[j] >> 16);
        i++;
      }
    }
  }

  // ② AI 内核前向传播
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
  signal.get_data = &get_audio_channels_data;
  ei_impulse_result_t result = {0};
  if (run_classifier(&signal, &result, false) != EI_IMPULSE_OK)
    return;

  // ③ 独立捕捉单帧原始概率值
  float raw_fire = 0.0f, raw_bell = 0.0f, raw_knock = 0.0f;
  float raw_glass = 0.0f, raw_alarm = 0.0f, raw_water = 0.0f;

  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
  {
    String lbl = String(result.classification[ix].label);
    float val = result.classification[ix].value;
    if (lbl == "fire_alarm")
      raw_fire = val;
    if (lbl == "doorbell")
      raw_bell = val;
    if (lbl == "door_knock")
      raw_knock = val;
    if (lbl == "glass_broken")
      raw_glass = val;
    if (lbl == "alarm")
      raw_alarm = val;
    if (lbl == "water")
      raw_water = val;
  }

  // ④ 短时滤波均值
  float sm_fire = AI_Filter::update_and_get_short_avg(AI_Filter::hist_fire, raw_fire);
  float sm_bell = AI_Filter::update_and_get_short_avg(AI_Filter::hist_bell, raw_bell);
  float sm_knock = AI_Filter::update_and_get_short_avg(AI_Filter::hist_knock, raw_knock);
  float sm_glass = AI_Filter::update_and_get_short_avg(AI_Filter::hist_glass, raw_glass);
  float sm_alarm = AI_Filter::update_and_get_short_avg(AI_Filter::hist_alarm, raw_alarm);
  float sm_water = AI_Filter::update_and_get_short_avg(AI_Filter::hist_water, raw_water);
  AI_Filter::write_index = (AI_Filter::write_index + 1) % AI_Filter::WINDOW_SIZE;

  // ⑤ 长时动态基准线
  AI_Filter::update_long_term_baseline(AI_Filter::base_fire, raw_fire);
  AI_Filter::update_long_term_baseline(AI_Filter::base_bell, raw_bell);
  AI_Filter::update_long_term_baseline(AI_Filter::base_knock, raw_knock);
  AI_Filter::update_long_term_baseline(AI_Filter::base_glass, raw_glass);
  AI_Filter::update_long_term_baseline(AI_Filter::base_alarm, raw_alarm);
  AI_Filter::update_long_term_baseline(AI_Filter::base_water, raw_water);

  // ⑥ 计算跳变差值
  float delta_fire = sm_fire - AI_Filter::base_fire;
  float delta_bell = sm_bell - AI_Filter::base_bell;
  float delta_knock = sm_knock - AI_Filter::base_knock;
  float delta_glass = sm_glass - AI_Filter::base_glass;
  float delta_alarm = sm_alarm - AI_Filter::base_alarm;
  float delta_water = sm_water - AI_Filter::base_water;

  // 🤖 诊断输出
  Serial.printf("⚡ [Delta] 火警:%d%% | 门铃:%d%% | 碎玻璃:%d%%\n",
                (int)(delta_fire * 100), (int)(delta_bell * 100), (int)(delta_glass * 100));

  // ⑦【自适应判定矩阵】
  uint8_t tx_event_id = 0x00;
  uint8_t confidence_score = 0;
  bool is_danger = false;
  String visual_name = "";

  const float DELTA_GATE = 0.35f;       // 其他常规异响保持 15% 的高灵敏度门槛
  const float GLASS_DELTA_GATE = 0.75f; // 🔥 针对碎玻璃单独大幅调高门槛至 35%，强行压制静息状态的误触发

  if (delta_fire >= DELTA_GATE)
  {
    tx_event_id = 0x01;
    confidence_score = (uint8_t)(sm_fire * 100);
    is_danger = true;
    visual_name = "ALERT: FIRE ALARM";
  }
  else if (delta_bell >= DELTA_GATE)
  {
    tx_event_id = 0x02;
    confidence_score = (uint8_t)(sm_bell * 100);
    visual_name = "ALERT: DOORBELL";
  }
  else if (delta_knock >= DELTA_GATE)
  {
    tx_event_id = 0x03;
    confidence_score = (uint8_t)(sm_knock * 100);
    visual_name = "ALERT: DOOR KNOCK";
  }
  else if (delta_glass >= GLASS_DELTA_GATE)
  { // 🔥 碎玻璃使用独立的硬核门槛
    tx_event_id = 0x05;
    confidence_score = (uint8_t)(sm_glass * 100);
    visual_name = "ALERT: GLASS BREAK";
  }
  else if (delta_alarm >= DELTA_GATE)
  {
    tx_event_id = 0x06;
    confidence_score = (uint8_t)(sm_alarm * 100);
    visual_name = "ALERT: PHONE RING";
  }
  else if (delta_water >= DELTA_GATE)
  {
    tx_event_id = 0x07;
    confidence_score = (uint8_t)(sm_water * 100);
    visual_name = "ALERT: WATER LEAK";
  }

  // ⑧ 执行并网发包
  if (tx_event_id != 0x00)
  {
    screen.lockedSound = visual_name;
    screen.isDanger = is_danger;
    screen.triggerTime = now;
    send_protocol_packet(tx_event_id, confidence_score);
  }

  // ⑨ OLED 紧凑型图像渲染总线
  display.clearDisplay();
  if (screen.lockedSound != "" && (now - screen.triggerTime < Config::DISPLAY_HOLD_MS))
  {
    display.invertDisplay(screen.isDanger);
    display.setTextSize(1); // 保持精简 1 号字
    display.setCursor(0, 12);
    display.println(screen.lockedSound);
  }
  else
  {
    screen.lockedSound = "";
    display.invertDisplay(false);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("AI SENTRY: ACTIVE");
    display.setCursor(0, 16);
    display.printf("G_d:%d%% | F_d:%d%%", (int)(delta_glass * 100), (int)(delta_fire * 100));
  }
  display.display();
}