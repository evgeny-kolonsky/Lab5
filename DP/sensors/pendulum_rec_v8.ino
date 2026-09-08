/*
 * pendulum_rec_v8.ino
 * Двойной маятник: узел оцифровки энкодера EM1-2-2500
 * Seeed XIAO nRF52840 (Sense), питание EM1 = 3.3V через high-side P-MOSFET.
 *
 * Подключение:
 *   EM1 pin 4 (Vcc)   -> Drain P-FET (Source -> 3V3, Gate: 100k на 3V3, 1k на D3)
 *   EM1 pin 1 (GND)   -> GND
 *   EM1 pin 3 (Ch A)  -> D0
 *   EM1 pin 5 (Ch B)  -> D1
 *   EM1 pin 2 (Index) -> D2
 *
 * Один таймер TIMER4 на 1 кГц является общей временной базой:
 *   - запись:      каждый (1000/recRate)-й тик        (50/100/200/500 Гц)
 *   - живой поток: каждый 10-й тик                    (100 Гц)
 * Живые отсчёты уходят пачками с номером первого отсчёта, поэтому
 * график в браузере строится по времени ПРИБОРА и не зависит от
 * джиттера BLE.
 *
 * Пакеты в браузер:
 *   L,<tick0>,<c0>,<d1>,<d2>,...      живой поток; время отсчёта i = (tick0+i)*10 мс
 *   T,<count>,<vbat_mv>,<state>,<samples>,<inv>,<zerr>,<edgeA_ps>,<edgeB_ps>,<zTotal>,<connInt>,<mtu>
 *                                     статус 4 Гц; connInt в единицах 1.25 мс
 *   state: 0=датчик выключен 1=живой просмотр 2=запись
 *
 * Команды (BLE NUS или USB Serial):
 *   S статус | O датчик вкл | F датчик выкл | R[,hz] запись | E стоп
 *   D выгрузка | X очистить буфер
 */

#include <bluefruit.h>

// ЕДИНСТВЕННОЕ, что меняется от прибора к прибору: его номер.
// Роль (центр / колено) назначается в интерфейсе, а не в прошивке,
// чтобы запасные приборы можно было ставить на любое место.
#define NODE_ID        1               // 1..99, у каждого прибора свой

#define PIN_A          D0
#define PIN_B          D1
#define PIN_Z          D2
#define PIN_ENC_EN     D3              // LOW = энкодер включен

#define COUNTS_PER_REV 10000L          // 2500 CPR x4
#define ENC_SETTLE_MS  100

#define BASE_HZ        1000            // общая временная база
#define LIVE_DIV       10              // живой поток 100 Гц
#define LIVE_RING      64
#define LIVE_BATCH     10              // отсчётов в пакете (=100 мс)
#define LIVE_FLUSH_MS  100             // период отправки живого потока

// Бюджет RAM (256 КБ всего). Если линковщик пожалуется на нехватку памяти —
// уменьшайте эти два числа, они независимы.
#define MAX_SAMPLES    24000           // энкодер, int16 -> 48 КБ (2 мин @200 Гц)
#define IMU_REC_HZ     50              // частота записи IMU
#define IMU_SAMPLES    6000            // IMU, 6*int16 -> 72 КБ (2 мин @50 Гц)
#define IMU_LIVE_DIV   4               // в эфир каждый 4-й отсчёт IMU (12.5 Гц)
#define MAX_Z_MARKS    256            // колено за минуту хаоса делает много оборотов
#define DEFAULT_RATE   200
#define CHUNK_SAMPLES  100

#define STATUS_PERIOD  250             // статус 4 Гц
#define IDLE_OFF_MS    (10UL*60UL*1000UL)

#ifndef PIN_VBAT
#define PIN_VBAT       31
#endif
#ifndef VBAT_ENABLE
#define VBAT_ENABLE    14
#endif
#define VBAT_LOW_MV    3500

BLEUart bleuart;
char deviceName[12];                   // "DP-01"

// ---------- IMU (LSM6DS3TR-C на плате XIAO Sense) ----------
// требуется библиотека "Seeed Arduino LSM6DS3" из Library Manager
#include "LSM6DS3.h"
#include "Wire.h"
LSM6DS3 imu(I2C_MODE, 0x6A);
bool imuOk = false;

// ---------- квадратура ----------
volatile int32_t  encCount     = 0;
volatile uint32_t invalidCount = 0;
volatile uint32_t zErrCount    = 0;
volatile uint32_t zCountTotal  = 0;
volatile int32_t  zMaxDev      = 0;    // худшее расхождение между индексами, отсчётов

volatile uint32_t edgeA        = 0;    // фронтов по каналу A
volatile uint32_t edgeB        = 0;    // фронтов по каналу B
volatile bool     zSeen        = false;
volatile int32_t  lastZcount   = 0;
volatile uint8_t  prevState    = 0;

static const int8_t QTAB[16] = {
   0, -1, +1,  2,
  +1,  0,  2, -1,
  -1,  2,  0, +1,
   2, +1, -1,  0
};

// Чтение A и B одним обращением к регистру порта.
// digitalRead() в ядре Arduino слишком медленный для десятков кГц фронтов:
// на быстром вращении колена это приводило к пропускам.
NRF_GPIO_Type* gpioPort = NULL;
uint8_t bitA = 0, bitB = 0;
bool    fastIO = false;

void setupFastIO() {
  uint32_t aa = g_ADigitalPinMap[PIN_A];
  uint32_t ab = g_ADigitalPinMap[PIN_B];
  if ((aa < 32) == (ab < 32)) {          // оба канала на одном порту
    gpioPort = (aa < 32) ? NRF_P0 : NRF_P1;
    bitA = aa & 31;
    bitB = ab & 31;
    fastIO = true;
  }
}

inline uint8_t readAB() {
  if (fastIO) {
    uint32_t in = gpioPort->IN;
    return (uint8_t)((((in >> bitA) & 1) << 1) | ((in >> bitB) & 1));
  }
  return (uint8_t)((digitalRead(PIN_A) << 1) | digitalRead(PIN_B));
}

inline void quadStep() {
  uint8_t s = readAB();
  int8_t d = QTAB[(prevState << 2) | s];
  prevState = s;
  if (d == 2) invalidCount++;
  else        encCount += d;
}

void isrA() { edgeA++; quadStep(); }
void isrB() { edgeB++; quadStep(); }

// ---------- буферы ----------
static int16_t  deltaBuf[MAX_SAMPLES];
volatile uint32_t sampleIdx  = 0;
volatile bool     recording  = false;
volatile int32_t  lastSample = 0;
volatile uint32_t clipCount  = 0;
int32_t  startCount = 0;
uint32_t recRate    = DEFAULT_RATE;
volatile uint32_t recDiv = BASE_HZ / DEFAULT_RATE;

// IMU: ax,ay,az,gx,gy,gz в единицах мg и мdps
static int16_t  imuBuf[IMU_SAMPLES][6];
volatile uint32_t imuIdx  = 0;
volatile bool     imuDue  = false;      // выставляет таймер, читает основной цикл
volatile uint32_t imuCntDown = 1;
int16_t  imuLast[6] = {0,0,0,0,0,0};
uint32_t imuLiveCnt = 0;

int32_t  recStartTick = 0;              // тик базового таймера в момент старта записи

volatile uint32_t zMarkSample[MAX_Z_MARKS];
volatile int32_t  zMarkCount[MAX_Z_MARKS];
volatile uint8_t  zMarks = 0;

// живой кольцевой буфер
volatile int32_t  liveRing[LIVE_RING];
volatile uint32_t liveWrTick = 0;      // номер следующего живого отсчёта
volatile uint32_t liveRdTick = 0;      // номер первого неотправленного

volatile uint32_t baseTick   = 0;
volatile uint32_t recCntDown  = 1;
volatile uint32_t liveCntDown = 1;

void isrZ() {
  zCountTotal++;
  if (readAB() != 0) zErrCount++;
  if (zSeen) {
    // между индексами счётчик обязан пройти целое число оборотов;
    // отклонение = число потерянных (или лишних) отсчётов
    int32_t rem = (encCount - lastZcount) % COUNTS_PER_REV;
    if (rem < 0) rem += COUNTS_PER_REV;
    int32_t dev = (rem > COUNTS_PER_REV / 2) ? (COUNTS_PER_REV - rem) : rem;
    if (dev != 0) zErrCount++;
    if (dev > zMaxDev) zMaxDev = dev;
  }
  zSeen = true;
  lastZcount = encCount;

  if (recording && zMarks < MAX_Z_MARKS) {
    zMarkSample[zMarks] = sampleIdx;
    zMarkCount[zMarks]  = encCount;
    zMarks++;
  }
}

// ---------- базовый таймер 1 кГц ----------
void baseTimerStart() {
  NRF_TIMER4->TASKS_STOP  = 1;
  NRF_TIMER4->MODE        = TIMER_MODE_MODE_Timer;
  NRF_TIMER4->BITMODE     = TIMER_BITMODE_BITMODE_32Bit;
  NRF_TIMER4->PRESCALER   = 4;                    // 1 МГц
  NRF_TIMER4->CC[0]       = 1000000UL / BASE_HZ;  // 1 мс
  NRF_TIMER4->SHORTS      = TIMER_SHORTS_COMPARE0_CLEAR_Msk;
  NRF_TIMER4->EVENTS_COMPARE[0] = 0;
  NRF_TIMER4->INTENSET    = TIMER_INTENSET_COMPARE0_Msk;
  NVIC_SetPriority(TIMER4_IRQn, 3);
  NVIC_ClearPendingIRQ(TIMER4_IRQn);
  NVIC_EnableIRQ(TIMER4_IRQn);
  NRF_TIMER4->TASKS_CLEAR = 1;
  NRF_TIMER4->TASKS_START = 1;
}

void baseTimerStop() {
  NRF_TIMER4->TASKS_STOP = 1;
  NRF_TIMER4->INTENCLR   = TIMER_INTENCLR_COMPARE0_Msk;
  NVIC_DisableIRQ(TIMER4_IRQn);
}

extern "C" void TIMER4_IRQHandler(void) {
  if (!NRF_TIMER4->EVENTS_COMPARE[0]) return;
  NRF_TIMER4->EVENTS_COMPARE[0] = 0;

  baseTick++;
  int32_t c = encCount;

  // --- живой поток ---
  if (--liveCntDown == 0) {
    liveCntDown = LIVE_DIV;
    liveRing[liveWrTick % LIVE_RING] = c;
    liveWrTick++;
    if (liveWrTick - liveRdTick > LIVE_RING)      // потребитель отстал
      liveRdTick = liveWrTick - LIVE_RING;
  }

  // --- IMU по сетке таймера ---
  if (--imuCntDown == 0) {
    imuCntDown = BASE_HZ / IMU_REC_HZ;
    imuDue = true;
  }

  // --- запись ---
  if (recording) {
    if (--recCntDown == 0) {
      recCntDown = recDiv;
      if (sampleIdx >= MAX_SAMPLES) { recording = false; return; }
      int32_t d = c - lastSample;
      if (d >  32767) { d =  32767; clipCount++; }
      if (d < -32768) { d = -32768; clipCount++; }
      deltaBuf[sampleIdx++] = (int16_t)d;
      lastSample = c;
    }
  }
}

// ---------- питание энкодера ----------
bool encoderOn    = false;
bool sensorWanted = false;

void encoderPowerOn() {
  if (encoderOn) return;
  digitalWrite(PIN_ENC_EN, LOW);
  delay(ENC_SETTLE_MS);
  pinMode(PIN_A, INPUT);
  pinMode(PIN_B, INPUT);
  pinMode(PIN_Z, INPUT);
  prevState = readAB();
  attachInterrupt(digitalPinToInterrupt(PIN_A), isrA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_B), isrB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_Z), isrZ, RISING);
  // фронты энкодера критичнее всего по времени: даём им высший
  // доступный приложению приоритет (0,1,4 заняты SoftDevice)
  NVIC_SetPriority(GPIOTE_IRQn, 2);
  encoderOn = true;
  liveCntDown = LIVE_DIV;
  baseTimerStart();
}

void encoderPowerOff() {
  if (!encoderOn) return;
  baseTimerStop();
  detachInterrupt(digitalPinToInterrupt(PIN_A));
  detachInterrupt(digitalPinToInterrupt(PIN_B));
  detachInterrupt(digitalPinToInterrupt(PIN_Z));
  pinMode(PIN_A, INPUT_PULLDOWN);
  pinMode(PIN_B, INPUT_PULLDOWN);
  pinMode(PIN_Z, INPUT_PULLDOWN);
  digitalWrite(PIN_ENC_EN, HIGH);
  encoderOn = false;
}

void applyPowerState() {
  bool need = sensorWanted || recording;
  if (need && !encoderOn)  encoderPowerOn();
  if (!need && encoderOn)  encoderPowerOff();
}

// ---------- батарея ----------
uint16_t readVbatMv() {
  digitalWrite(VBAT_ENABLE, LOW);
  delay(2);
  uint32_t acc = 0;
  for (int i = 0; i < 8; i++) acc += analogRead(PIN_VBAT);
  digitalWrite(VBAT_ENABLE, HIGH);
  float raw = acc / 8.0f;
  return (uint16_t)(raw * (3.6f / 4096.0f) * 2.961f * 1000.0f);
}

// ---------- вывод ----------
Stream* replyTo = NULL;          // NULL = BLE

void sendLine(const char* s) {
  // строка и '\n' одним вызовом: раздельные print() уходят разными
  // BLE-пакетами и разделитель может потеряться
  char out[200];
  int n = snprintf(out, sizeof(out), "%s\n", s);
  if (n <= 0) return;
  if (n > (int)sizeof(out) - 1) n = sizeof(out) - 1;
  if (replyTo) { replyTo->write((const uint8_t*)out, n); return; }
  bleuart.write((const uint8_t*)out, n);
}

void bleWriteAll(const uint8_t* p, size_t n) {
  size_t sent = 0;
  uint32_t guard = millis();
  while (sent < n) {
    if (!replyTo && !Bluefruit.connected()) return;
    size_t w = replyTo ? replyTo->write(p + sent, n - sent)
                       : bleuart.write(p + sent, n - sent);
    if (w == 0) {
      if (millis() - guard > 5000) return;
      delay(2);
    } else {
      sent += w;
      guard = millis();
    }
  }
}

// ---------- запись ----------
void startRecording(uint32_t hz) {
  if (recording) return;
  if (hz != 50 && hz != 100 && hz != 200 && hz != 500) hz = DEFAULT_RATE;
  recRate = hz;
  encoderPowerOn();                    // идемпотентно, запускает и таймер
  noInterrupts();
  recDiv       = BASE_HZ / hz;
  recCntDown   = recDiv;
  sampleIdx    = 0;
  zMarks       = 0;
  invalidCount = 0;
  zErrCount    = 0;
  zMaxDev      = 0;
  clipCount    = 0;
  zSeen        = false;
  lastSample   = encCount;
  startCount   = encCount;
  imuIdx       = 0;
  recStartTick = baseTick;
  recording    = true;
  interrupts();
  // на время записи просим внешний кварц: точность временной сетки
  // с внутреннего RC (~1%) поднимается до единиц-десятков ppm
  sd_clock_hfclk_request();
}

void stopRecording() {
  if (recording) sd_clock_hfclk_release();
  recording = false;
  applyPowerState();
}

// ---------- команды ----------
uint8_t stateCode() { return recording ? 2 : (encoderOn ? 1 : 0); }

void connInfo(uint16_t& ci, uint16_t& mtu) {
  ci = 0; mtu = 0;
  BLEConnection* c = Bluefruit.Connection(0);
  if (c && c->connected()) { ci = c->getConnectionInterval(); mtu = c->getMtu(); }
}

void doStatus() {
  char buf[200];
  uint16_t ci, mtu; connInfo(ci, mtu);
  snprintf(buf, sizeof(buf),
    "STATUS,%s,%u,%u,%lu,%u,%lu,%lu,%lu,%lu,%ld,%u,%u,%u,%lu,%ld",
    deviceName, stateCode(), readVbatMv(),
    (unsigned long)sampleIdx, MAX_SAMPLES, (unsigned long)recRate,
    (unsigned long)invalidCount, (unsigned long)zErrCount,
    (unsigned long)zCountTotal, (long)encCount, ci, mtu,
    NODE_ID, (unsigned long)imuIdx, (long)lastZcount);
  sendLine(buf);
}

void doDump() {
  if (recording) { sendLine("ERR,recording"); return; }
  uint32_t n = sampleIdx;
  if (n == 0) { sendLine("ERR,empty"); return; }

  char buf[110];
  snprintf(buf, sizeof(buf), "DUMP,%lu,%lu,%ld,%u,%lu,%lu,%ld,%ld,%lu",
           (unsigned long)n, (unsigned long)recRate, (long)startCount,
           readVbatMv(), (unsigned long)invalidCount, (unsigned long)zErrCount,
           (long)recStartTick, (long)zMaxDev, (unsigned long)zCountTotal);
  sendLine(buf);

  for (uint8_t i = 0; i < zMarks; i++) {
    snprintf(buf, sizeof(buf), "Z,%lu,%ld",
             (unsigned long)zMarkSample[i], (long)zMarkCount[i]);
    sendLine(buf);
  }

  uint32_t nchunks = (n + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES;
  snprintf(buf, sizeof(buf), "BIN,%lu,%u", (unsigned long)nchunks, CHUNK_SAMPLES);
  sendLine(buf);
  delay(20);

  static uint8_t frame[8 + CHUNK_SAMPLES * 2];
  uint32_t idx = 0;
  for (uint32_t c = 0; c < nchunks; c++) {
    uint16_t cnt = (n - idx > CHUNK_SAMPLES) ? CHUNK_SAMPLES : (uint16_t)(n - idx);
    uint16_t len = cnt * 2;
    frame[0] = 0xA5; frame[1] = 0x5A;
    frame[2] = c & 0xFF;   frame[3] = (c >> 8) & 0xFF;
    frame[4] = len & 0xFF; frame[5] = (len >> 8) & 0xFF;
    uint16_t sum = 0;
    for (uint16_t i = 0; i < cnt; i++) {
      int16_t v = deltaBuf[idx + i];
      uint8_t lo = v & 0xFF, hi = (v >> 8) & 0xFF;
      frame[6 + i * 2]     = lo;
      frame[6 + i * 2 + 1] = hi;
      sum += lo; sum += hi;
    }
    frame[6 + len]     = sum & 0xFF;
    frame[6 + len + 1] = (sum >> 8) & 0xFF;
    bleWriteAll(frame, 8 + len);
    idx += cnt;
    if (!replyTo && (c % 8 == 7)) delay(4);
  }
  delay(30);

  // --- вторая часть: IMU ---
  uint32_t m = imuIdx;
  snprintf(buf, sizeof(buf), "IMU,%lu,%u", (unsigned long)m, IMU_REC_HZ);
  sendLine(buf);
  if (m) {
    const uint16_t IMU_PER_CHUNK = 20;                 // 20*6*2 = 240 байт
    uint32_t ich = (m + IMU_PER_CHUNK - 1) / IMU_PER_CHUNK;
    snprintf(buf, sizeof(buf), "IBIN,%lu,%u", (unsigned long)ich, IMU_PER_CHUNK);
    sendLine(buf);
    delay(20);
    static uint8_t iframe[8 + IMU_PER_CHUNK * 12];
    uint32_t k = 0;
    for (uint32_t c = 0; c < ich; c++) {
      uint16_t cnt = (m - k > IMU_PER_CHUNK) ? IMU_PER_CHUNK : (uint16_t)(m - k);
      uint16_t len = cnt * 12;
      iframe[0] = 0xA5; iframe[1] = 0x5A;
      iframe[2] = c & 0xFF;   iframe[3] = (c >> 8) & 0xFF;
      iframe[4] = len & 0xFF; iframe[5] = (len >> 8) & 0xFF;
      uint16_t sum = 0;
      for (uint16_t i = 0; i < cnt; i++) {
        for (uint8_t a = 0; a < 6; a++) {
          int16_t v = imuBuf[k + i][a];
          uint8_t lo = v & 0xFF, hi = (v >> 8) & 0xFF;
          iframe[6 + (i * 6 + a) * 2]     = lo;
          iframe[6 + (i * 6 + a) * 2 + 1] = hi;
          sum += lo; sum += hi;
        }
      }
      iframe[6 + len]     = sum & 0xFF;
      iframe[6 + len + 1] = (sum >> 8) & 0xFF;
      bleWriteAll(iframe, 8 + len);
      k += cnt;
      if (!replyTo && (c % 8 == 7)) delay(4);
    }
    delay(30);
  }
  sendLine("END");
}

void handleCommand(char* cmd, Stream* src) {
  replyTo = src;
  while (*cmd == ' ') cmd++;
  char* e = cmd + strlen(cmd);
  while (e > cmd && (e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\n')) *--e = 0;
  if (!*cmd) { replyTo = NULL; return; }

  char buf[64];
  switch (cmd[0]) {
    case 'S': case 's': doStatus(); break;

    case 'O': case 'o':
      sensorWanted = true;  applyPowerState(); sendLine("OK,live"); break;

    case 'F': case 'f':
      sensorWanted = false; applyPowerState(); sendLine("OK,off");  break;

    case 'R': case 'r': {
      uint32_t hz = DEFAULT_RATE;
      char* comma = strchr(cmd, ',');
      if (comma) hz = strtoul(comma + 1, NULL, 10);
      startRecording(hz);
      snprintf(buf, sizeof(buf), "OK,rec,%lu,%lu",
               (unsigned long)recRate, (unsigned long)recStartTick);
      sendLine(buf);
      break;
    }

    case 'E': case 'e':
      stopRecording();
      snprintf(buf, sizeof(buf), "OK,stop,%lu", (unsigned long)sampleIdx);
      sendLine(buf);
      break;

    case 'D': case 'd': doDump(); break;

    case 'X': case 'x':
      if (!recording) {
        noInterrupts(); sampleIdx = 0; zMarks = 0; interrupts();
        sendLine("OK,cleared");
      } else sendLine("ERR,recording");
      break;

    default: sendLine("ERR,cmd");
  }
  replyTo = NULL;
}

void pumpChannel(Stream* s, bool isUsb, char* lineBuf, uint8_t& lineLen) {
  while (s->available()) {
    char c = s->read();
    if (c == '\n' || c == '\r') {
      if (lineLen) {
        lineBuf[lineLen] = 0;
        handleCommand(lineBuf, isUsb ? s : NULL);
        lineLen = 0;
      }
    } else if (lineLen < 31) {
      lineBuf[lineLen++] = c;
    }
  }
}

// ---------- живой поток ----------
void flushLive(uint32_t now) {
  static uint32_t lastFlush = 0;
  uint32_t wr, rd;
  noInterrupts(); wr = liveWrTick; rd = liveRdTick; interrupts();
  uint32_t pending = wr - rd;
  if (pending == 0) return;
  // ждём полную пачку либо истечения периода — иначе уходят пакеты по 1 отсчёту
  if (pending < LIVE_BATCH && (now - lastFlush) < LIVE_FLUSH_MS) return;
  lastFlush = now;
  if (pending > LIVE_BATCH) pending = LIVE_BATCH;

  char buf[160];
  int p = snprintf(buf, sizeof(buf), "L,%lu", (unsigned long)rd);
  int32_t prev = 0;
  for (uint32_t i = 0; i < pending; i++) {
    int32_t v;
    noInterrupts(); v = liveRing[(rd + i) % LIVE_RING]; interrupts();
    if (i == 0) { p += snprintf(buf + p, sizeof(buf) - p, ",%ld", (long)v); }
    else        { p += snprintf(buf + p, sizeof(buf) - p, ",%ld", (long)(v - prev)); }
    prev = v;
    if (p > (int)sizeof(buf) - 16) { pending = i + 1; break; }
  }
  buf[p++] = '\n';
  bleuart.write((const uint8_t*)buf, p);

  noInterrupts(); liveRdTick = rd + pending; interrupts();
}

// ---------- IMU ----------
void imuPower(bool on) {
#ifdef PIN_LSM6DS3TR_C_POWER
  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);
  digitalWrite(PIN_LSM6DS3TR_C_POWER, on ? HIGH : LOW);
#endif
  if (on && !imuOk) {
    delay(20);
    imuOk = (imu.begin() == 0);
  }
  if (!on) imuOk = false;
}

// читается в основном цикле по флагу от таймера — сетка задаётся таймером,
// а блокирующий обмен по I2C не мешает прерываниям энкодера
void serviceImu() {
  if (!imuDue) return;
  imuDue = false;
  if (!imuOk) return;

  int16_t v[6];
  v[0] = (int16_t)(imu.readFloatAccelX() * 1000.0f);   // мg
  v[1] = (int16_t)(imu.readFloatAccelY() * 1000.0f);
  v[2] = (int16_t)(imu.readFloatAccelZ() * 1000.0f);
  v[3] = (int16_t)(imu.readFloatGyroX()  * 10.0f);     // 0.1 град/с
  v[4] = (int16_t)(imu.readFloatGyroY()  * 10.0f);
  v[5] = (int16_t)(imu.readFloatGyroZ()  * 10.0f);
  memcpy(imuLast, v, sizeof(v));

  if (recording && imuIdx < IMU_SAMPLES) {
    memcpy(imuBuf[imuIdx], v, sizeof(v));
    imuIdx++;
  }

  if (++imuLiveCnt >= IMU_LIVE_DIV && Bluefruit.connected()) {
    imuLiveCnt = 0;
    char buf[96];
    int len = snprintf(buf, sizeof(buf), "I,%lu,%d,%d,%d,%d,%d,%d\n",
                       (unsigned long)(liveWrTick),
                       v[0], v[1], v[2], v[3], v[4], v[5]);
    if (len > 0) bleuart.write((const uint8_t*)buf, len);
  }
}

// ---------- LED ----------
void ledSet(bool r, bool g, bool b) {
  digitalWrite(LED_RED,   r ? LOW : HIGH);
  digitalWrite(LED_GREEN, g ? LOW : HIGH);
  digitalWrite(LED_BLUE,  b ? LOW : HIGH);
}

// после подключения ещё раз просим короткий интервал: Windows часто
// игнорирует PPCP, но иногда соглашается на явный запрос
void connectCallback(uint16_t conn_handle) {
  BLEConnection* c = Bluefruit.Connection(conn_handle);
  if (c) c->requestConnectionParameter(12, 0, 200);   // 15 мс, latency 0, таймаут 2 с
}

void setup() {
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  ledSet(false, false, false);

  pinMode(PIN_ENC_EN, OUTPUT);
  digitalWrite(PIN_ENC_EN, HIGH);
  pinMode(PIN_A, INPUT_PULLDOWN);
  pinMode(PIN_B, INPUT_PULLDOWN);
  pinMode(PIN_Z, INPUT_PULLDOWN);

  pinMode(VBAT_ENABLE, OUTPUT);
  digitalWrite(VBAT_ENABLE, HIGH);
  analogReadResolution(12);

  snprintf(deviceName, sizeof(deviceName), "DP-%02d", NODE_ID);

  setupFastIO();

  Serial.begin(115200);
  Wire.begin();
  imuPower(true);

  // максимальная пропускная способность: должно стоять ДО begin()
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);
  Bluefruit.begin();
  Bluefruit.setTxPower(4);
  Bluefruit.setName(deviceName);
  // просим короткий интервал соединения 7.5-15 мс вместо ~50 мс по умолчанию
  Bluefruit.Periph.setConnInterval(6, 12);
  Bluefruit.Periph.setConnectCallback(connectCallback);
  bleuart.begin();

  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(160, 1600);
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);
}

void loop() {
  static char    bleLine[32], usbLine[32];
  static uint8_t bleLen = 0, usbLen = 0;
  static uint32_t lastStatus = 0, lastVbat = 0, lastMove = 0, lastEdgeCalc = 0;
  static uint32_t prevEdgeA = 0, prevEdgeB = 0, rateA = 0, rateB = 0;
  static int32_t  lastSeenCount = 0;
  static uint16_t vbatMv = 4000;
  static bool     wasConnected = false;

  uint32_t now = millis();

  if (!recording && sampleIdx >= MAX_SAMPLES) stopRecording();  // автостоп

  pumpChannel(&bleuart, false, bleLine, bleLen);
  if (Serial) pumpChannel(&Serial, true, usbLine, usbLen);

  bool conn = Bluefruit.connected();
  if (conn && !wasConnected) { sensorWanted = true; lastMove = now; }
  if (!conn && wasConnected && !recording) sensorWanted = false;
  wasConnected = conn;

  if (encCount != lastSeenCount) { lastSeenCount = encCount; lastMove = now; }
  if (recording) lastMove = now;
  if (sensorWanted && !recording && (now - lastMove > IDLE_OFF_MS)) sensorWanted = false;

  applyPowerState();
  serviceImu();


  if (now - lastVbat > 10000) { vbatMv = readVbatMv(); lastVbat = now; }

  // частоты фронтов по каналам (диагностика линий)
  if (now - lastEdgeCalc >= 1000) {
    uint32_t ea, eb;
    noInterrupts(); ea = edgeA; eb = edgeB; interrupts();
    uint32_t dt = now - lastEdgeCalc;
    rateA = (ea - prevEdgeA) * 1000UL / dt;
    rateB = (eb - prevEdgeB) * 1000UL / dt;
    prevEdgeA = ea; prevEdgeB = eb;
    lastEdgeCalc = now;
  }

  if (conn) {
    // событие индекса: консоль по нему привязывает ноль к механике оси
    static uint32_t lastZrep = 0;
    if (zCountTotal != lastZrep) {
      lastZrep = zCountTotal;
      char zb[64];
      int zl = snprintf(zb, sizeof(zb), "Zv,%ld,%lu,%ld\n",
                        (long)lastZcount, (unsigned long)zCountTotal, (long)zMaxDev);
      if (zl > 0) bleuart.write((const uint8_t*)zb, zl);
    }

    flushLive(now);                               // живой поток пачками

    if (now - lastStatus >= STATUS_PERIOD) {      // статус 4 Гц
      lastStatus = now;
      char buf[176];
      noInterrupts();
      int32_t c = encCount; uint32_t n = sampleIdx;
      uint32_t iv = invalidCount, ze = zErrCount, zt = zCountTotal;
      interrupts();
      uint16_t ci, mtu; connInfo(ci, mtu);
      int len = snprintf(buf, sizeof(buf),
        "T,%ld,%u,%u,%lu,%lu,%lu,%lu,%lu,%lu,%u,%u,%u,%ld\n",
        (long)c, vbatMv, stateCode(), (unsigned long)n,
        (unsigned long)iv, (unsigned long)ze,
        (unsigned long)rateA, (unsigned long)rateB, (unsigned long)zt, ci, mtu,
        imuOk ? 1 : 0, (long)zMaxDev);
      if (len > 0) bleuart.write((const uint8_t*)buf, len);
    }
  }

  // индикация
  bool lowBat = vbatMv < VBAT_LOW_MV;
  bool flash;
  if (recording) {
    flash = (now % 1000) < 40;
    ledSet(lowBat && flash, !lowBat && flash, false);
  } else if (conn) {
    flash = (now % 2000) < 40;
    ledSet(lowBat && flash, false, !lowBat && flash);
  } else {
    flash = lowBat && (now % 5000) < 30;
    ledSet(flash, false, false);
  }

  delay(conn ? 2 : 60);
}
