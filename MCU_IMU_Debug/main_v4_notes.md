# main_v4.c — Extended I2C Diagnostic: Changes, Rationale, and Usage

## Context

After running the v3 diagnostic firmware, the results were:

- Pull-ups confirmed working (`pb8_idle_high = 1`, `pb9_idle_high = 1`)
- Chip can drive lines low (`pb8_can_drive_low = 1`, `pb9_can_drive_low = 1`)
- Clocks correct (`pclk1_freq = 64 MHz`)
- All 8 test phases completed (`test_phase = 8`)
- **Zero devices found on full bus scan (`num_devices = 0`)**
- **Both addresses fail with `error_after_6A = 0x00000004` = `HAL_I2C_ERROR_AF` (ACK failure)**

ACK failure means the MCU successfully sends a START condition and the 7-bit address byte, but no slave pulls SDA low on the 9th clock pulse. The bus is electrically healthy. The IMU is simply not responding.

The v3 firmware exhausted the obvious software checks. v4 adds four new tests targeting the remaining software-diagnosable failure modes — things that v3 proved weren't electrical but couldn't rule out on the firmware side.

---

## How to Use main_v4.c

1. In STM32CubeIDE, rename `main_v4.c` to `main.c` and replace the existing `main.c` in `Core/Src/`.
2. All other generated files (`i2c.c`, `gpio.c`, `usb_device.c`, etc.) stay **unchanged**.
3. Build → flash via ST-LINK SWD using the normal workflow (see `stlink_workflow.md`).
4. Wait ~10 seconds after flashing for all tests to complete (the extended scan adds time).
5. Reconnect CubeProgrammer in **Hot plug + Software reset** mode.
6. Grep the `.map` file for variable names to find their RAM addresses, then read via the Memory tab.

**Key signal that all tests finished:** `test_phase = 99` (v3 used `8` — the change makes it immediately obvious which firmware version ran).

---

## What Changed and Why

### Variables Added

All new variables are `volatile` globals, same as v3, readable via SWD at their `.map` addresses.

---

### Test 9 — GPIO Alternate Function Register Dump

**What it does:**
Reads three raw GPIOB peripheral registers directly — `GPIOB->MODER`, `GPIOB->AFR[1]`, and `GPIOB->OTYPER` — and decodes each bit field into a pass/fail variable.

**New variables:**

| Variable | Expected value | What it checks |
|---|---|---|
| `gpiob_moder` | raw register | MODER bits 19:16 |
| `gpiob_afr1` | raw register | AFR[1] bits 7:0 |
| `gpiob_otyper` | raw register | OTYPER bits 9:8 |
| `pb8_is_af_mode` | 1 | PB8 (SCL) is in alternate function mode |
| `pb9_is_af_mode` | 1 | PB9 (SDA) is in alternate function mode |
| `pb8_is_af4` | 1 | PB8 is mapped to AF4 (= I2C1_SCL on STM32WB55) |
| `pb9_is_af4` | 1 | PB9 is mapped to AF4 (= I2C1_SDA on STM32WB55) |
| `pb8_is_opendrain` | 1 | PB8 output type is open-drain (required for I2C) |
| `pb9_is_opendrain` | 1 | PB9 output type is open-drain |

**Why this matters:**

v3 confirmed the pins are electrically functional as GPIO — they can be driven low and the pull-ups hold them high. But that test only exercises the pins as ordinary GPIO. It says nothing about whether the I2C peripheral is actually routed to them.

The hardware I2C peripheral (I2C1) talks to the physical world through the alternate function mux. If `MODER` or `AFR[1]` is wrong — whether because `HAL_I2C_MspInit()` was skipped, ran on wrong parameters, or got overwritten by something else in the init sequence — the peripheral generates I2C clocks internally but they never reach the physical PB8/PB9 pins. From the MCU's perspective everything looks normal. From the IMU's perspective it sees nothing. The result is exactly the `HAL_I2C_ERROR_AF` observed in v3.

This is the highest-priority new test because it's three lines of code and could immediately identify the root cause.

**How to interpret:**
- Any `pb*_is_af_mode = 0` or `pb*_is_af4 = 0` → confirmed software root cause. The fix is ensuring `HAL_I2C_MspInit()` runs correctly after every init cycle.
- All six decoded flags = 1 → pin configuration is correct; move on to the bit-bang test.

---

### Test 10 — I2C Peripheral Enable (PE) Bit

**What it does:**
Reads bit 0 of `I2C1->CR1` after a clean `reset_i2c()` call and stores it in `i2c_pe_bit`.

**New variable:** `i2c_pe_bit` — must be `1`.

**Why this matters:**

The `CR1` register is already captured in v3 as `i2c_cr1`, but it's captured at the start of the test sequence before any bus clearing or peripheral resets. The PE bit can change state during error recovery. Capturing it again after a clean re-init, immediately before the new tests run, gives a definitive answer on whether the peripheral is actually enabled at the point the I2C transactions are attempted.

If `PE = 0`, the peripheral is disabled. All HAL calls will time out or error regardless of what's on the bus.

---

### Test 11 — Pure Bit-Bang I2C WHO_AM_I

**What it does:**
Implements the I2C protocol entirely in software using raw GPIO writes on PB8 (SCL) and PB9 (SDA). The hardware I2C peripheral is completely idle during this test. The clock rate is approximately 500 Hz (1 ms per half-bit via `HAL_Delay(1)`) — far below the LSM6DSO's maximum of 1 MHz FM+, so timing cannot cause a failure here.

The test attempts a full WHO_AM_I read at address `0x6A`: START → address byte (write) → register byte (0x0F) → repeated START → address byte (read) → receive one byte → NACK → STOP. It then does a quick address ping at `0x6B` to check the alternate address.

**New variables:**

| Variable | Expected value | Meaning |
|---|---|---|
| `bitbang_ack_addr` | 1 | IMU ACKed the 0x6A address byte |
| `bitbang_ack_reg` | 1 | IMU ACKed the register address byte |
| `bitbang_ack_rdaddr` | 1 | IMU ACKed the read-mode address byte |
| `bitbang_whoami` | `0x6C` | WHO_AM_I register value (LSM6DSO fixed ID) |
| `bitbang_6B_ack` | 0 or 1 | Whether 0x6B also responds |

**Why this matters:**

This test is the definitive software/hardware discriminator:

- **If `bitbang_ack_addr = 1` and `bitbang_whoami = 0x6C`:** The IMU is alive and responding correctly. The issue is entirely within the hardware I2C peripheral configuration — TIMINGR wrong, alternate function not set, filter misconfigured. No hardware rework needed.

- **If `bitbang_ack_addr = 0`:** The IMU does not respond to any I2C transaction regardless of how it's generated. The issue is physical at the IMU — bad solder joint on the LGA-14L, CS pin not pulled high (IMU stuck in SPI mode), or a damaged chip. No amount of firmware changes will fix this.

This single variable answers the question of whether to keep debugging firmware or go back to the hardware.

**Implementation note:** In open-drain output mode, `HAL_GPIO_ReadPin` reads the actual pin voltage through the STM32's IDR register even while the ODR is controlling the output. This means SDA can be sampled without switching the pin between input and output modes — the slave's ACK (pulling SDA low) is visible immediately when the master releases SDA (sets ODR high, FET turns off, pull-up takes over).

---

### Test 12 — STOP+START Alternative to HAL_I2C_Mem_Read

**What it does:**
Uses `HAL_I2C_Master_Transmit` followed by a separate `HAL_I2C_Master_Receive` call instead of `HAL_I2C_Mem_Read`. Tests both 0x6A and 0x6B.

**New variables:**

| Variable | Expected value | Meaning |
|---|---|---|
| `alt_status_write_6A` | `0` (HAL_OK) | Transmit leg succeeded |
| `alt_status_read_6A` | `0` (HAL_OK) | Receive leg succeeded |
| `alt_whoami_6A` | `0x6C` | Data read back |
| `alt_error_6A` | `0` | No I2C error after transmit |
| *(same for 6B)* | | |

**Why this matters:**

`HAL_I2C_Mem_Read` internally generates a **repeated START** (Sr) between the write phase (sending the register address) and the read phase (clocking in data). The I2C spec allows repeated start, and the LSM6DSO datasheet shows it as valid — but it is a different state machine path through both the master and slave. Some I2C implementations have subtle bugs in repeated-start handling.

Using `Master_Transmit` + `Master_Receive` as two separate calls issues a full **STOP** after the write, then a fresh **START** for the read. This is a simpler transaction sequence and exercises a different code path through the HAL's I2C state machine.

If this test succeeds where `HAL_I2C_Mem_Read` fails, the issue is in the HAL's repeated-start implementation or the peripheral's handling of it.

---

### Test 13 — Extended Address Scan

**What it does:**
Same full scan as v3 (addresses 0x08–0x77) but with **5 retries** and **100 ms timeout** per address, compared to v3's 2 retries and 50 ms.

**New variables:** `ext_num_devices`, `ext_devices_found[16]`

**Why this matters:**

The original scan results (`num_devices = 0`) are reliable, but running a more patient scan after all the peripheral resets and state changes from the new tests gives a clean second data point. If any device appears here but not in the original scan, it suggests the earlier tests disturbed the bus state in a way that prevented detection.

---

## Diagnostic Decision Tree for v4 Results

```
bitbang_ack_addr = 1?
├── YES → IMU is physically alive. Issue is in hardware I2C peripheral config.
│         Check: pb8_is_af4, pb9_is_af4, i2c_pe_bit, i2c_timingr value.
│         If bitbang_whoami = 0x6C → confirmed IMU works, just need correct HAL config.
└── NO  → IMU does not respond to any I2C regardless of how it's generated.
          Issue is physical at the IMU. Go back to hardware:
          - Reflow LGA-14L pads
          - Verify CS (pin 13) is at 3.3V (I2C mode select)
          - Replace IMU if damaged

pb8_is_af_mode = 0 or pb8_is_af4 = 0?
└── YES → Confirmed: HAL_I2C_MspInit() did not correctly configure AF4.
          Fix: ensure reset_i2c() (DeInit + Init) is called before I2C transactions,
          not just MX_I2C1_Init() alone.

i2c_pe_bit = 0?
└── YES → I2C peripheral disabled. HAL init is not completing correctly.

alt_status_write_6A = 0 but status_6A != 0?
└── YES → STOP+START works but repeated-START (Mem_Read) doesn't.
          HAL repeated-start bug or peripheral state machine issue.
```

---

## TIMINGR Reference

The `i2c_timingr` variable (captured in Test 2, unchanged from v3) holds the I2C timing register value. For **64 MHz PCLK1 → 100 kHz standard-mode I2C** with the analog filter enabled, STM32CubeMX typically generates `0x10909CEC`. If the value you read differs significantly, CubeMX may have been configured with the wrong input clock frequency when the project was set up, producing a malformed I2C waveform even though the peripheral appears to be running normally.

To verify: open the CubeMX `.ioc` file, check the I2C1 configuration page, and confirm the shown timing matches `0x10909CEC` for your clock tree.
