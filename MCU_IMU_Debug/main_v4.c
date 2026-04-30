/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main_v4.c
  * @brief          : Extended I2C Diagnostic — v4
  *
  * HOW TO USE:
  *   Rename this file to main.c and drop it into the STM32CubeIDE project,
  *   replacing the existing main.c. All other generated files (i2c.c, gpio.c,
  *   usb_device.c, etc.) stay unchanged.
  *
  * WHAT'S NEW IN V4 vs V3:
  *
  *   TEST 9  — GPIO Alternate Function Register Dump
  *     Reads GPIOB->MODER, GPIOB->AFR[1], GPIOB->OTYPER directly from the
  *     peripheral registers to confirm PB8/PB9 are genuinely in AF4 (I2C1)
  *     open-drain mode. If these registers are wrong the hardware I2C peripheral
  *     is internally active but physically disconnected from the pins — that
  *     would produce exactly the ACK failure seen in v3 with no wiring fault.
  *
  *   TEST 10 — I2C Peripheral Enable (PE) Bit Check
  *     Reads CR1 bit 0. PE=0 means the peripheral is disabled and nothing is
  *     actually transmitted regardless of what HAL functions are called.
  *
  *   TEST 11 — Pure Bit-Bang I2C WHO_AM_I
  *     Implements I2C entirely in GPIO software — no hardware I2C peripheral
  *     involved whatsoever. Bypasses TIMINGR, CR1/CR2, HAL config, all of it.
  *     INTERPRETATION:
  *       bitbang_ack_addr=1, bitbang_whoami=0x6C → IMU alive; bug is in
  *         hardware I2C peripheral config (TIMINGR wrong, AF not set, etc.)
  *       bitbang_ack_addr=0 → IMU doesn't respond to any I2C; issue is
  *         physical at the IMU (bad solder, wrong CS pin, damage) — definitive.
  *
  *   TEST 12 — HAL_I2C_Master_Transmit + Master_Receive (STOP+START)
  *     HAL_I2C_Mem_Read uses a repeated START between the write and read
  *     phases. This test issues a full STOP after writing the register address
  *     then a fresh START for the read — a different protocol path that some
  *     I2C implementations handle differently. Tests both 0x6A and 0x6B.
  *
  *   TEST 13 — Extended Address Scan (more retries, longer timeout)
  *     Same full scan as v3 but with 5 retries and 100ms per address instead
  *     of 2 retries and 50ms. Catches marginal or slow-to-respond devices.
  *
  * READING RESULTS:
  *   Same workflow as v3: grep the .map file for variable names, open the
  *   Memory tab in STM32CubeProgrammer, read at the listed addresses.
  *   test_phase=99 (not 8 like v3) means all tests completed successfully.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* NOTE: usb_device.h is included to match the CubeIDE project structure.
   MX_USB_Device_Init() is intentionally NOT called — USB CDC was removed
   because CDC_Transmit_FS() blocks on USB enumeration, which never happens
   on this board due to the D- hardware fault. The include is harmless. */
#include "main.h"
#include "i2c.h"
#include "usb_device.h"
#include "gpio.h"

/* USER CODE BEGIN Includes */
#include <string.h>
/* USER CODE END Includes */

/* USER CODE BEGIN PV */

// ================================================================
// DIAGNOSTIC VARIABLES — V3 ORIGINALS (all preserved unchanged)
// ================================================================

// GPIO pull-up / drive tests (Test 4)
volatile uint8_t pb8_idle_high    = 0;  // 1 = external pull-up on PB8 works
volatile uint8_t pb9_idle_high    = 0;  // 1 = external pull-up on PB9 works
volatile uint8_t pb8_can_drive_low = 0; // 1 = chip can sink current on PB8
volatile uint8_t pb9_can_drive_low = 0; // 1 = chip can sink current on PB9

// I2C peripheral register snapshots (Tests 2, 8)
volatile uint32_t i2c_isr_before = 0;
volatile uint32_t i2c_isr_after  = 0;
volatile uint32_t i2c_cr1        = 0;
volatile uint32_t i2c_cr2        = 0;
volatile uint32_t i2c_timingr    = 0;
// NOTE on TIMINGR: for 64 MHz PCLK1 → 100 kHz standard-mode I2C, CubeMX
// typically generates 0x10909CEC (with analog filter enabled). If the value
// you read differs significantly, CubeMX may have been configured with the
// wrong PCLK1 frequency and the generated waveform timing is off.

// Address scan results (Test 7)
volatile uint8_t devices_found[16] = {0};
volatile uint8_t num_devices       = 0;

// WHO_AM_I results — hardware I2C (Tests 5, 6)
volatile uint8_t  whoami_6A     = 0;    // Should be 0x6C if IMU alive at 0x6A
volatile uint8_t  whoami_6B     = 0;    // Should be 0x6C if IMU alive at 0x6B
volatile uint8_t  status_6A     = 0xFF; // HAL_StatusTypeDef: 0=OK, 1=ERR, 2=BUSY, 3=TIMEOUT
volatile uint8_t  status_6B     = 0xFF;
volatile uint32_t error_after_6A = 0;  // hi2c1.ErrorCode: 0x04=ACK failure
volatile uint32_t error_after_6B = 0;

// General progress / liveness
volatile uint32_t loop_count = 0;  // Increments in while(1); confirms code is running
volatile uint8_t  test_phase = 0;  // Last completed test phase (99 = all done)

// Clock verification (Test 1)
volatile uint32_t pclk1_freq = 0;  // Should be 0x03D09000 = 64 MHz
volatile uint32_t hclk_freq  = 0;  // Should be 64 MHz


// ================================================================
// NEW V4 DIAGNOSTIC VARIABLES
// ================================================================

// --- Test 9: GPIO Alternate Function Register Dump ---
// These are raw peripheral register reads. Compare against expected values
// to confirm PB8/PB9 are genuinely in I2C alternate function mode.
volatile uint32_t gpiob_moder  = 0; // bits 19:16 expected: 0b10101010 (AF for PB8, PB9)
volatile uint32_t gpiob_afr1   = 0; // bits  7:0  expected: 0x44       (AF4 for PB8, AF4 for PB9)
volatile uint32_t gpiob_otyper = 0; // bits  9:8  expected: 0b11        (open-drain for both)

// Decoded pass/fail for each sub-check (1=correct, 0=wrong)
volatile uint8_t pb8_is_af_mode    = 0; // MODER bits 17:16 == 0b10 ?
volatile uint8_t pb9_is_af_mode    = 0; // MODER bits 19:18 == 0b10 ?
volatile uint8_t pb8_is_af4        = 0; // AFR[1] bits  3:0 == 0x4 ?
volatile uint8_t pb9_is_af4        = 0; // AFR[1] bits  7:4 == 0x4 ?
volatile uint8_t pb8_is_opendrain  = 0; // OTYPER bit 8 == 1 ?
volatile uint8_t pb9_is_opendrain  = 0; // OTYPER bit 9 == 1 ?

// --- Test 10: I2C Peripheral Enable bit ---
volatile uint8_t i2c_pe_bit = 0; // CR1 bit 0; must be 1 for any I2C activity

// --- Test 11: Bit-Bang I2C ---
// Granular ACK tracking so you can see exactly where the transaction fails.
volatile uint8_t bitbang_ack_addr   = 0; // 1 = slave ACKed the address byte (0x6A+W)
volatile uint8_t bitbang_ack_reg    = 0; // 1 = slave ACKed the register byte (0x0F)
volatile uint8_t bitbang_ack_rdaddr = 0; // 1 = slave ACKed the read-address byte (0x6A+R)
volatile uint8_t bitbang_whoami     = 0; // Byte read back — should be 0x6C
volatile uint8_t bitbang_6B_ack     = 0; // 1 = 0x6B also ACKs (quick check of alt address)

// --- Test 12: STOP+START alternative to HAL_I2C_Mem_Read ---
volatile uint8_t  alt_status_write_6A = 0xFF; // HAL status for the transmit leg at 0x6A
volatile uint8_t  alt_status_read_6A  = 0xFF; // HAL status for the receive leg at 0x6A
volatile uint8_t  alt_whoami_6A       = 0;    // Should be 0x6C
volatile uint32_t alt_error_6A        = 0;
volatile uint8_t  alt_status_write_6B = 0xFF;
volatile uint8_t  alt_status_read_6B  = 0xFF;
volatile uint8_t  alt_whoami_6B       = 0;
volatile uint32_t alt_error_6B        = 0;

// --- Test 13: Extended address scan ---
volatile uint8_t ext_num_devices      = 0;
volatile uint8_t ext_devices_found[16] = {0};

/* USER CODE END PV */

void SystemClock_Config(void);
void PeriphCommonClock_Config(void);

/* USER CODE BEGIN 0 */

// ================================================================
// HELPER FUNCTIONS — V3 ORIGINALS (all preserved unchanged)
// ================================================================

// Temporarily reconfigure PB8/PB9 as GPIO to test pull-ups and drive strength.
// Called before any I2C transaction; reset_i2c() must follow to restore AF4.
void test_gpio_state(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Input mode (no internal pull) — external pull-ups should hold lines high
    GPIO_InitStruct.Pin  = GPIO_PIN_8 | GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
    HAL_Delay(10);

    pb8_idle_high = (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_8) == GPIO_PIN_SET) ? 1 : 0;
    pb9_idle_high = (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9) == GPIO_PIN_SET) ? 1 : 0;

    // Open-drain output — verify chip can sink current (drive low)
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_OD;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_RESET);
    HAL_Delay(1);
    pb8_can_drive_low = (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_8) == GPIO_PIN_RESET) ? 1 : 0;
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET);
    HAL_Delay(1);

    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_RESET);
    HAL_Delay(1);
    pb9_can_drive_low = (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9) == GPIO_PIN_RESET) ? 1 : 0;
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_SET);
    HAL_Delay(1);
}

// Bit-bang 9 SCL pulses to release any stuck I2C slave.
// A slave stuck mid-transfer holds SDA low; 9 clocks guarantees it sees its
// byte boundary and releases the bus before the STOP condition.
void clear_i2c_bus(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    GPIO_InitStruct.Pin   = GPIO_PIN_8;
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_OD;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    GPIO_InitStruct.Pin  = GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    for (int i = 0; i < 9; i++) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_RESET);
        HAL_Delay(1);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET);
        HAL_Delay(1);
    }
}

// Full peripheral reset: DeInit sets handle state to RESET, which causes the
// subsequent HAL_I2C_Init (inside MX_I2C1_Init) to call HAL_I2C_MspInit and
// re-apply the GPIO alternate function configuration.
void reset_i2c(void) {
    HAL_I2C_DeInit(&hi2c1);
    HAL_Delay(10);
    MX_I2C1_Init();
    HAL_Delay(10);
}

// Original address scan: 2 retries, 50 ms timeout, range 0x08–0x77.
void run_full_scan(void) {
    num_devices = 0;
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        if (hi2c1.ErrorCode != 0) reset_i2c();
        HAL_StatusTypeDef result = HAL_I2C_IsDeviceReady(&hi2c1, addr << 1, 2, 50);
        if (result == HAL_OK) {
            if (num_devices < 16) devices_found[num_devices++] = addr;
        }
    }
}


// ================================================================
// NEW: TEST 9 — GPIO ALTERNATE FUNCTION REGISTER DUMP
//
// Reading GPIOB->MODER and GPIOB->AFR[1] directly is the only way
// to confirm the pins are in AF4 mode. The v3 test only confirmed
// electrical functionality (pull-ups, drive strength) — it did NOT
// verify that the MCU's I2C peripheral is actually connected to the
// physical pins. If HAL_I2C_MspInit failed silently or was skipped,
// the I2C peripheral generates clocks internally but they never
// reach PB8/PB9. That produces HAL_I2C_ERROR_AF with a clean bus.
// ================================================================
void check_gpio_config(void) {
    gpiob_moder  = GPIOB->MODER;
    gpiob_afr1   = GPIOB->AFR[1]; // AFR[1] covers pins 8–15
    gpiob_otyper = GPIOB->OTYPER;

    // PB8 (SCL): MODER bits 17:16 must be 0b10 (alternate function)
    pb8_is_af_mode = (((gpiob_moder >> 16) & 0x3) == 0x2) ? 1 : 0;

    // PB9 (SDA): MODER bits 19:18 must be 0b10 (alternate function)
    pb9_is_af_mode = (((gpiob_moder >> 18) & 0x3) == 0x2) ? 1 : 0;

    // PB8: AFR[1] bits 3:0 must be 0x4 (AF4 = I2C1_SCL on STM32WB55)
    pb8_is_af4 = (((gpiob_afr1 >> 0) & 0xF) == 0x4) ? 1 : 0;

    // PB9: AFR[1] bits 7:4 must be 0x4 (AF4 = I2C1_SDA on STM32WB55)
    pb9_is_af4 = (((gpiob_afr1 >> 4) & 0xF) == 0x4) ? 1 : 0;

    // PB8: OTYPER bit 8 must be 1 (open-drain; I2C requires open-drain)
    pb8_is_opendrain = (((gpiob_otyper >> 8) & 0x1) == 1) ? 1 : 0;

    // PB9: OTYPER bit 9 must be 1 (open-drain)
    pb9_is_opendrain = (((gpiob_otyper >> 9) & 0x1) == 1) ? 1 : 0;
}


// ================================================================
// NEW: TEST 11 — PURE BIT-BANG I2C WHO_AM_I
//
// All I2C signaling is done with raw GPIO writes. The hardware I2C
// peripheral (I2C1) is completely bypassed — TIMINGR, CR1/CR2,
// analog/digital filters, DMA, none of it is involved.
//
// The clock rate is ~500 Hz (1 ms per half-bit) — far below the
// LSM6DSO's maximum of 1 MHz, so timing cannot be the reason for
// a no-ACK here.
//
// In open-drain output mode, HAL_GPIO_ReadPin reads the actual pin
// voltage via IDR even while the ODR is set — no mode switching
// needed. When SDA is "released" (ODR=1, FET off), the slave can
// override by pulling low. Reading IDR at that moment gives the
// slave's response.
// ================================================================

static inline void bb_scl_high(void) { HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET);   HAL_Delay(1); }
static inline void bb_scl_low(void)  { HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_RESET); HAL_Delay(1); }
static inline void bb_sda_high(void) { HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_SET);   HAL_Delay(1); }
static inline void bb_sda_low(void)  { HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_RESET); HAL_Delay(1); }

// Send one byte MSB-first; return 1 if slave ACKed (pulled SDA low on 9th clock)
static uint8_t bb_send_byte(uint8_t byte) {
    for (int i = 7; i >= 0; i--) {
        if ((byte >> i) & 0x1) bb_sda_high(); else bb_sda_low();
        bb_scl_high();
        bb_scl_low();
    }
    // Release SDA so slave can ACK (pull low) or NACK (leave high)
    bb_sda_high();
    bb_scl_high();
    // ACK = slave holds SDA low; NACK = SDA stays high (pulled up)
    uint8_t acked = (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9) == GPIO_PIN_RESET) ? 1 : 0;
    bb_scl_low();
    return acked;
}

// Read one byte MSB-first; send NACK to signal last byte read
static uint8_t bb_recv_byte_nack(void) {
    uint8_t data = 0;
    for (int i = 7; i >= 0; i--) {
        bb_sda_high(); // release so slave drives the bit
        bb_scl_high();
        if (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9) == GPIO_PIN_SET) data |= (1 << i);
        bb_scl_low();
    }
    // NACK: master holds SDA high through the 9th clock
    bb_sda_high();
    bb_scl_high();
    bb_scl_low();
    return data;
}

// I2C START: SDA falls while SCL is high
static void bb_start(void) {
    bb_sda_high(); bb_scl_high();
    bb_sda_low();
    bb_scl_low();
}

// I2C STOP: SCL rises while SDA is low, then SDA rises
static void bb_stop(void) {
    bb_sda_low();
    bb_scl_high();
    bb_sda_high();
    HAL_Delay(2);
}

void bitbang_i2c_test(void) {
    // Configure PB8 and PB9 as open-drain outputs (bus idle = both high)
    GPIO_InitTypeDef g = {0};
    g.Pin   = GPIO_PIN_8 | GPIO_PIN_9;
    g.Mode  = GPIO_MODE_OUTPUT_OD;
    g.Pull  = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &g);

    bb_sda_high(); bb_scl_high();
    HAL_Delay(5); // settle

    // ---- Address 0x6A: full WHO_AM_I read ----
    bb_start();
    bitbang_ack_addr = bb_send_byte((0x6A << 1) | 0); // 0xD4 = addr + WRITE

    if (bitbang_ack_addr) {
        bitbang_ack_reg = bb_send_byte(0x0F); // WHO_AM_I register address

        // Repeated START for the read phase
        bb_sda_high(); bb_scl_high(); // setup for repeated start
        bb_sda_low();  bb_scl_low();

        bitbang_ack_rdaddr = bb_send_byte((0x6A << 1) | 1); // 0xD5 = addr + READ

        if (bitbang_ack_rdaddr) {
            bitbang_whoami = bb_recv_byte_nack(); // Expected: 0x6C
        }
    }
    bb_stop();
    HAL_Delay(5);

    // ---- Address 0x6B: just check for ACK (in case SDO/SA0 pin is high) ----
    bb_start();
    bitbang_6B_ack = bb_send_byte((0x6B << 1) | 0); // 0xD6 = addr + WRITE
    bb_stop();
    HAL_Delay(5);
    // Note: if bitbang_6B_ack=1 here, repeat the full WHO_AM_I read above
    // with 0x6B substituted — hardware not worth adding until we know if
    // the IMU responds at all.
}


// ================================================================
// NEW: TEST 12 — STOP+START (Master_Transmit + Master_Receive)
//
// HAL_I2C_Mem_Read internally issues a repeated START (Sr) between
// writing the register address and reading the data. Most I2C slaves
// support repeated start, but the LSM6DSO datasheet explicitly shows
// both repeated-start and stop+start read sequences as valid. Testing
// the stop+start path eliminates any repeated-start implementation
// issue in the HAL or the peripheral's state machine.
// ================================================================
void try_stop_start_read(void) {
    uint8_t reg = 0x0F; // WHO_AM_I register

    // --- Address 0x6A ---
    reset_i2c();
    // Phase 1: send register address (write transaction, full STOP at end)
    alt_status_write_6A = (uint8_t)HAL_I2C_Master_Transmit(
        &hi2c1, 0x6A << 1, &reg, 1, 500);
    alt_error_6A = hi2c1.ErrorCode;

    // Phase 2: read 1 byte (fresh START, no repeated start)
    if (alt_status_write_6A == HAL_OK) {
        alt_status_read_6A = (uint8_t)HAL_I2C_Master_Receive(
            &hi2c1, 0x6A << 1, &alt_whoami_6A, 1, 500);
    }

    // --- Address 0x6B ---
    reset_i2c();
    alt_status_write_6B = (uint8_t)HAL_I2C_Master_Transmit(
        &hi2c1, 0x6B << 1, &reg, 1, 500);
    alt_error_6B = hi2c1.ErrorCode;

    if (alt_status_write_6B == HAL_OK) {
        alt_status_read_6B = (uint8_t)HAL_I2C_Master_Receive(
            &hi2c1, 0x6B << 1, &alt_whoami_6B, 1, 500);
    }
}


// ================================================================
// NEW: TEST 13 — EXTENDED ADDRESS SCAN
//
// Same range as the original scan (0x08–0x77) but with 5 retries
// per address and a 100 ms timeout. The original used 2 retries and
// 50 ms. This catches devices that need more attempt cycles to wake
// up, e.g. after the peripheral resets that precede this test.
// ================================================================
void run_extended_scan(void) {
    ext_num_devices = 0;
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        if (hi2c1.ErrorCode != 0) reset_i2c();
        HAL_StatusTypeDef result = HAL_I2C_IsDeviceReady(&hi2c1, addr << 1, 5, 100);
        if (result == HAL_OK) {
            if (ext_num_devices < 16) ext_devices_found[ext_num_devices++] = addr;
        }
    }
}

/* USER CODE END 0 */

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    PeriphCommonClock_Config();
    MX_GPIO_Init();
    MX_I2C1_Init();

    /* USER CODE BEGIN 2 */

    HAL_Delay(500); // Power-on settle; LSM6DSO boot time is ~10 ms, this is plenty

    // ========== TEST 1: Clock frequencies (v3 unchanged) ==========
    test_phase   = 1;
    pclk1_freq   = HAL_RCC_GetPCLK1Freq(); // Expected: 0x03D09000 = 64 MHz
    hclk_freq    = HAL_RCC_GetHCLKFreq();  // Expected: 64 MHz

    // ========== TEST 2: I2C peripheral register snapshot (v3 unchanged) ==========
    test_phase   = 2;
    i2c_cr1      = hi2c1.Instance->CR1;
    i2c_cr2      = hi2c1.Instance->CR2;
    i2c_timingr  = hi2c1.Instance->TIMINGR;
    i2c_isr_before = hi2c1.Instance->ISR;

    // ========== TEST 3: 9-pulse bus clear (v3 unchanged) ==========
    test_phase = 3;
    clear_i2c_bus();
    HAL_Delay(10);
    // MX_I2C1_Init without prior DeInit does NOT re-run MspInit (GPIO config).
    // That is intentional here — the subsequent reset_i2c() calls before each
    // actual I2C attempt do the full DeInit+Init cycle that restores AF4.
    MX_I2C1_Init();
    HAL_Delay(10);

    // ========== TEST 4: GPIO pull-up / drive test (v3 unchanged) ==========
    test_phase = 4;
    test_gpio_state();
    HAL_Delay(10);
    MX_I2C1_Init(); // Same note as Test 3 — AF4 restoration happens in reset_i2c()
    HAL_Delay(10);

    // ========== TEST 5: WHO_AM_I at 0x6A via Mem_Read (v3 unchanged) ==========
    test_phase = 5;
    reset_i2c(); // Full DeInit+Init: restores PB8/PB9 to AF4 open-drain
    status_6A = (uint8_t)HAL_I2C_Mem_Read(&hi2c1, 0x6A << 1, 0x0F,
                                            I2C_MEMADD_SIZE_8BIT,
                                            (uint8_t*)&whoami_6A, 1, 1000);
    error_after_6A = hi2c1.ErrorCode;
    HAL_Delay(50);

    // ========== TEST 6: WHO_AM_I at 0x6B via Mem_Read (v3 unchanged) ==========
    test_phase = 6;
    reset_i2c();
    status_6B = (uint8_t)HAL_I2C_Mem_Read(&hi2c1, 0x6B << 1, 0x0F,
                                            I2C_MEMADD_SIZE_8BIT,
                                            (uint8_t*)&whoami_6B, 1, 1000);
    error_after_6B = hi2c1.ErrorCode;
    HAL_Delay(50);

    // ========== TEST 7: Original address scan (v3 unchanged) ==========
    test_phase = 7;
    reset_i2c();
    run_full_scan();

    // ========== TEST 8: Final ISR state (v3 unchanged) ==========
    test_phase = 8;
    i2c_isr_after = hi2c1.Instance->ISR;

    // ========== TEST 9 (NEW): GPIO alternate function register dump ==========
    // Run after a clean reset_i2c() so pins SHOULD be in AF4. If they're not,
    // that's the root cause of the ACK failures in tests 5–7.
    test_phase = 9;
    reset_i2c();
    check_gpio_config();

    // ========== TEST 10 (NEW): I2C peripheral enable bit ==========
    // Captured immediately after clean init. PE=0 means the peripheral is off.
    i2c_pe_bit = (uint8_t)(I2C1->CR1 & 0x01); // Must be 1

    // ========== TEST 11 (NEW): Bit-bang I2C WHO_AM_I ==========
    // This reconfigures PB8/PB9 as plain GPIO — hardware I2C is idle.
    // reset_i2c() afterward restores AF4 before the next hardware tests.
    test_phase = 11;
    bitbang_i2c_test();
    reset_i2c();
    HAL_Delay(10);

    // ========== TEST 12 (NEW): STOP+START alternative ==========
    test_phase = 12;
    try_stop_start_read();

    // ========== TEST 13 (NEW): Extended address scan ==========
    test_phase = 13;
    reset_i2c();
    run_extended_scan();

    // ========== All tests complete ==========
    // test_phase=99 (not 8 as in v3) makes it unambiguous that this is v4
    // and that all tests finished without hanging.
    test_phase = 99;

    /* USER CODE END 2 */

    while (1)
    {
        /* USER CODE BEGIN 3 */
        loop_count++;   // Confirm main loop is alive; increments ~10x/sec
        HAL_Delay(100);
        /* USER CODE END 3 */
    }
}

// ================================================================
// SYSTEM CLOCK CONFIG — copied verbatim from v3 / CubeMX generated
// ================================================================
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    HAL_PWR_EnableBkUpAccess();
    __HAL_RCC_LSEDRIVE_CONFIG(RCC_LSEDRIVE_LOW);
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE | RCC_OSCILLATORTYPE_LSE
                                     | RCC_OSCILLATORTYPE_MSI | RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSEState            = RCC_HSE_ON;
    RCC_OscInitStruct.LSEState            = RCC_LSE_ON;
    RCC_OscInitStruct.HSIState            = RCC_HSI_ON;
    RCC_OscInitStruct.MSIState            = RCC_MSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.MSICalibrationValue = RCC_MSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.MSIClockRange       = RCC_MSIRANGE_6;
    RCC_OscInitStruct.PLL.PLLState        = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource       = RCC_PLLSOURCE_MSI;
    RCC_OscInitStruct.PLL.PLLM           = RCC_PLLM_DIV1;
    RCC_OscInitStruct.PLL.PLLN           = 32;
    RCC_OscInitStruct.PLL.PLLP           = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ           = RCC_PLLQ_DIV2;
    RCC_OscInitStruct.PLL.PLLR           = RCC_PLLR_DIV2;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK4 | RCC_CLOCKTYPE_HCLK2
                                | RCC_CLOCKTYPE_HCLK  | RCC_CLOCKTYPE_SYSCLK
                                | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource      = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider     = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider    = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB2CLKDivider    = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.AHBCLK2Divider    = RCC_SYSCLK_DIV2;
    RCC_ClkInitStruct.AHBCLK4Divider    = RCC_SYSCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK) Error_Handler();
}

void PeriphCommonClock_Config(void)
{
    RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_SMPS;
    PeriphClkInitStruct.SmpsClockSelection   = RCC_SMPSCLKSOURCE_HSI;
    PeriphClkInitStruct.SmpsDivSelection     = RCC_SMPSCLKDIV_RANGE0;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) Error_Handler();
}

void Error_Handler(void)
{
    __disable_irq();
    while (1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {}
#endif
