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

---

## Step-by-Step: Reading v4 Results After Flashing

This section walks through the entire process from the moment the firmware is flashed through reading and interpreting every variable. Follow these steps in order.

---

### Step 1 — Flash the firmware

Follow the standard SWD flashing procedure from `stlink_workflow.md`:

- CubeProgrammer settings: Port = SWD, Frequency = 1800 kHz, Mode = **Under reset**, Reset mode = **Hardware reset**
- Click **Connect** — chip info should populate (STM32WB5x/35xx, Device ID 0x495)
- In the **Erasing & Programming** tab, use the **top** Start Programming button (NOT the bottom "Automatic Mode" button)
- Browse to the `.hex` file, check **Verify programming** and **Run after programming**, uncheck **Skip flash erase**
- Click **Start Programming** and wait for "Download verified successfully"

---

### Step 2 — Let the firmware run

After the flash completes, **do not reconnect immediately**. The firmware needs time to run through all 13 test phases. The extended address scan (Test 13) tries every I2C address with 5 retries at 100 ms each — this alone takes about 37 seconds in the worst case.

**Wait at least 60 seconds** before reconnecting to be safe.

---

### Step 3 — Reconnect in read mode

Change CubeProgrammer settings to read the live RAM without resetting the chip:

- Mode: **Hot plug**
- Reset mode: **Software reset**
- Click **Connect**

The chip should show as connected. Your code is still running — `loop_count` is incrementing in the background.

---

### Step 4 — Find variable addresses in the .map file

Every global variable's RAM address is listed in the `.map` file generated during the build. Open a terminal and run:

```bash
grep -E "test_phase|loop_count|pclk1_freq|hclk_freq|i2c_cr1|i2c_cr2|i2c_timingr|i2c_isr|i2c_pe_bit|pb8_idle|pb9_idle|pb8_can|pb9_can|pb8_is|pb9_is|gpiob_moder|gpiob_afr1|gpiob_otyper|whoami_6|status_6|error_after|num_devices|devices_found|bitbang|alt_status|alt_whoami|alt_error|ext_num|ext_devices" \
  /Users/albertwang/Desktop/WB55_IMU_USB/Debug/WB55_IMU_USB.map
```

You'll get output like:

```
0x20000008    test_phase
0x2000000c    loop_count
0x20000010    pclk1_freq
...
```

The address on the left is where that variable lives in RAM. Note the lowest address — you'll read a block starting from there.

---

### Step 5 — Open the Memory tab and read RAM

In CubeProgrammer:

1. Click the **"+"** tab at the top → select **Open memory tab**
2. In the Address field, type the lowest variable address you found (typically somewhere around `0x20000000`)
3. Set Size to `300` (enough to cover all variables)
4. Set Data width to **8-bit**
5. Click **Read**

A hex dump appears. Each row is 16 bytes. Values are in hexadecimal.

---

### Step 6 — Decode the values

Multi-byte variables (uint32_t) are stored **little-endian**: the least significant byte is at the lowest address. For example, if you see bytes `00 90 09 10` at an address, the 32-bit value is `0x10090900` (read the bytes right-to-left).

Use the addresses from Step 4 to locate each variable. The sections below tell you what to look for.

---

### Step 7 — Confirm the firmware ran completely

**Variable:** `test_phase`
**Location:** find in .map output

| Value | Meaning |
|---|---|
| `99` | All 13 tests completed — this is what you want |
| `1`–`13` | Firmware stalled at that test phase — something hung |
| `0` | Code never started — flash failed or chip not running |

**Variable:** `loop_count`
**Location:** find in .map output

This increments every 100 ms in the `while(1)` loop. If `test_phase = 99` and `loop_count` is nonzero, the chip is definitely running and all tests finished. If `loop_count = 0` and `test_phase = 0`, the chip is not executing code — check your flash and SWD connection.

---

### Step 8 — Verify clocks (Test 1)

**Variables:** `pclk1_freq`, `hclk_freq`

| Variable | Expected hex | Expected decimal | If wrong |
|---|---|---|---|
| `pclk1_freq` | `0x03D09000` | 64,000,000 | Clock config failed; check SystemClock_Config |
| `hclk_freq` | `0x03D09000` | 64,000,000 | Same |

If either clock reads differently (e.g. `0x007A1200` = 8 MHz), the PLL didn't start correctly. This would also affect the I2C timing register — TIMINGR would have been calculated for the wrong clock speed.

---

### Step 9 — Verify I2C peripheral configuration (Tests 2 and 10)

**Variable:** `i2c_timingr`

Expected value: `0x10909CEC` for 64 MHz PCLK1 → 100 kHz standard mode.

If the value differs significantly (e.g. `0x00303D5B` which is the 8 MHz value), CubeMX generated timing for the wrong clock. The I2C waveform on the bus would have wrong pulse widths — potentially unreadable by the IMU.

**Variable:** `i2c_pe_bit`

| Value | Meaning |
|---|---|
| `1` | I2C peripheral is enabled — correct |
| `0` | I2C peripheral is disabled — HAL init did not complete; all transactions were dead on arrival |

**Variable:** `i2c_cr1` (captured early in Test 2)

Bit 0 is PE. In the raw 32-bit value, if the last hex digit is odd (e.g. `0x00000001`), PE is set. If even (e.g. `0x00000000`), PE is not set. Focus on `i2c_pe_bit` for a cleaner read.

---

### Step 10 — Check GPIO pull-ups and drive (Test 4, v3 originals)

**Variables:** `pb8_idle_high`, `pb9_idle_high`, `pb8_can_drive_low`, `pb9_can_drive_low`

These are 1-byte values.

| Variable | Expected | If 0 |
|---|---|---|
| `pb8_idle_high` | `1` | No pull-up on SCL, or SCL shorted to GND |
| `pb9_idle_high` | `1` | No pull-up on SDA, or SDA shorted to GND |
| `pb8_can_drive_low` | `1` | MCU can't sink current on PB8 — pin or chip issue |
| `pb9_can_drive_low` | `1` | MCU can't sink current on PB9 — pin or chip issue |

If all four are `1`, the bus is electrically healthy and the MCU can physically control the lines. This was already confirmed in v3.

---

### Step 11 — Read GPIO alternate function registers (Test 9 — NEW)

These are the most important new checks. They confirm whether PB8/PB9 are actually connected to the I2C peripheral at the silicon level.

**Variables:** `pb8_is_af_mode`, `pb9_is_af_mode`, `pb8_is_af4`, `pb9_is_af4`, `pb8_is_opendrain`, `pb9_is_opendrain`

All six are 1-byte values. All six should be `1`.

| Variable | Expected | If 0 — what it means |
|---|---|---|
| `pb8_is_af_mode` | `1` | PB8 is not in alternate function mode — it's GPIO, not I2C SCL. The peripheral is disconnected from the pin. |
| `pb9_is_af_mode` | `1` | PB9 is not in alternate function mode — it's GPIO, not I2C SDA. |
| `pb8_is_af4` | `1` | PB8 is in alternate function mode but mapped to the wrong function — not I2C1. |
| `pb9_is_af4` | `1` | PB9 is in alternate function mode but mapped to the wrong function. |
| `pb8_is_opendrain` | `1` | PB8 is push-pull, not open-drain. I2C requires open-drain. With push-pull, SCL would fight the pull-up resistor, or fight the slave trying to clock-stretch. |
| `pb9_is_opendrain` | `1` | PB9 is push-pull, not open-drain. SDA would fight the pull-up and prevent the slave from ACKing (slave ACK works by pulling SDA low — it can't do that against a push-pull output). |

**If any of these is 0:** This is the root cause. The fix is not hardware — it's ensuring the init sequence calls `HAL_I2C_DeInit()` before `MX_I2C1_Init()` (i.e., using `reset_i2c()` not bare `MX_I2C1_Init()`), so that `HAL_I2C_MspInit()` runs and correctly reconfigures the GPIO.

**If all six are 1:** Pin configuration is correct. The I2C peripheral is physically connected to the pins. Move to Step 12.

**Raw register values for cross-checking** (`gpiob_moder`, `gpiob_afr1`, `gpiob_otyper`):

For `gpiob_moder`: bits 19:16 should be `1010`. In the full 32-bit register this means the nibble covering PB8–PB9 reads as `0xA_____` in the upper portion. The exact expected bit pattern for bits 19:16 is `1010` = `0xA` when isolated.

For `gpiob_afr1`: the lowest byte (bits 7:0) should be `0x44` — AF4 for PB8 in the low nibble, AF4 for PB9 in the high nibble.

For `gpiob_otyper`: bits 9:8 should both be `1`. In the full 16-bit register, bits 9 and 8 set means the value has `0x300` in it (i.e., `xxxxxxxx_11xxxxxx` in binary when looking at the low 10 bits).

---

### Step 12 — Read bit-bang I2C results (Test 11 — NEW)

This is the key diagnostic result. No hardware I2C peripheral involved — pure GPIO.

**Variables:** `bitbang_ack_addr`, `bitbang_ack_reg`, `bitbang_ack_rdaddr`, `bitbang_whoami`, `bitbang_6B_ack`

All are 1-byte values.

**Scenario A — IMU is alive (best case):**

| Variable | Value |
|---|---|
| `bitbang_ack_addr` | `1` |
| `bitbang_ack_reg` | `1` |
| `bitbang_ack_rdaddr` | `1` |
| `bitbang_whoami` | `0x6C` |

This means the IMU is physically functional and responding correctly over I2C. The v3 failures were caused entirely by a software/configuration issue in the hardware I2C peripheral. Go back to Step 11 and fix whatever GPIO or TIMINGR issue is present there. No hardware rework needed.

**Scenario B — IMU not at 0x6A but try 0x6B:**

| Variable | Value |
|---|---|
| `bitbang_ack_addr` | `0` |
| `bitbang_6B_ack` | `1` |

The IMU is alive but at address 0x6B, meaning the SDO/SA0 pin (LSM6DSO pin 16) is pulled high rather than low. This is a wiring/PCB issue — the address pin is at a different voltage than expected. The firmware already tries both addresses, but this tells you which one is actually correct.

**Scenario C — ACK on address but wrong WHO_AM_I:**

| Variable | Value |
|---|---|
| `bitbang_ack_addr` | `1` |
| `bitbang_whoami` | not `0x6C` (e.g. `0x00` or `0xFF`) |

Something is responding on the bus but it's not returning the LSM6DSO's fixed ID. Possible causes: a different I2C device at address 0x6A on the board, or the IMU is damaged and returns garbage. Check `bitbang_whoami` — `0xFF` usually means SDA is stuck high (open circuit on the read path), `0x00` means SDA is stuck low.

**Scenario D — No ACK anywhere (worst case):**

| Variable | Value |
|---|---|
| `bitbang_ack_addr` | `0` |
| `bitbang_6B_ack` | `0` |

The IMU does not respond to any I2C communication, regardless of the method used. This is definitively a hardware problem at or near the IMU. No firmware change will fix this. The next steps are physical:

1. Check voltage on IMU CS pin (pin 13) — must be 3.3V for I2C mode. If it's 0V or floating, the IMU is in SPI mode and ignores I2C entirely.
2. Check voltage on IMU VDD and VDDIO (pins 12 and 9) — must be 3.3V.
3. Reflow the LGA-14L pads — hand-soldered LGA packages commonly have cold joints under the chip where you can't see them.
4. Replace the IMU if reflowing doesn't help.

---

### Step 13 — Read hardware I2C WHO_AM_I results (Tests 5, 6, 12)

These are the original v3 tests plus the new STOP+START alternative. If bit-bang worked (Step 12 Scenario A), these tell you exactly which protocol path in the hardware peripheral is broken.

**Variables from Tests 5/6 (Mem_Read, repeated START):**

| Variable | Expected if working | Meaning if not |
|---|---|---|
| `status_6A` | `0` (HAL_OK) | `1` = HAL_ERROR, `2` = HAL_BUSY, `3` = HAL_TIMEOUT |
| `error_after_6A` | `0` | `0x04` = ACK failure, `0x01` = bus error, `0x02` = arbitration lost |
| `whoami_6A` | `0x6C` | Value read from WHO_AM_I register |
| *(same for 6B)* | | |

**Variables from Test 12 (Master_Transmit + Receive, STOP+START):**

| Variable | Expected if working | Meaning if not |
|---|---|---|
| `alt_status_write_6A` | `0` (HAL_OK) | Transmit phase failed — slave didn't ACK address |
| `alt_error_6A` | `0` | Error code after transmit |
| `alt_status_read_6A` | `0` (HAL_OK) | Receive phase failed |
| `alt_whoami_6A` | `0x6C` | Value received |
| *(same for 6B)* | | |

**How to compare Tests 5/6 vs Test 12:**

- Both fail with `error = 0x04`: ACK failure in both protocol paths. Either GPIO AF is wrong (check Step 11) or the IMU doesn't respond at all (confirmed by bit-bang in Step 12).
- Test 12 succeeds but Tests 5/6 fail: the STOP+START path works but repeated START doesn't. This is a HAL or peripheral state machine bug. Workaround: replace `HAL_I2C_Mem_Read` calls with the `Master_Transmit` + `Master_Receive` pattern in any future firmware.
- Tests 5/6 succeed but Test 12 fails: unexpected — the more complex protocol path works but the simpler one doesn't. Likely a peripheral state issue between tests; re-flash and check again.

---

### Step 14 — Read address scan results (Tests 7 and 13)

**Variables from Test 7 (original scan):** `num_devices`, `devices_found[0]` through `devices_found[15]`

**Variables from Test 13 (extended scan):** `ext_num_devices`, `ext_devices_found[0]` through `ext_devices_found[15]`

| Outcome | Meaning |
|---|---|
| Both scans: `num_devices = 0`, `ext_num_devices = 0` | No I2C devices found at any address. Consistent with ACK failure on all addresses. |
| Test 7: 0 devices, Test 13: 1+ devices | A device exists but needed more retries to respond. The address in `ext_devices_found[0]` is where it is. |
| Both scans: 1+ devices | The device was found. Read `devices_found[0]` — that's the IMU's actual I2C address (should be `0x6A` = `106` decimal or `0x6B` = `107` decimal). |
| Unexpected address (not 0x6A or 0x6B) | A different I2C device is present. Investigate what else is on the board at that address. |

`devices_found` stores raw 7-bit addresses. `0x6A` in that array means the IMU responded at address 0x6A.

---

### Quick Reference: All Variables and Expected Values

| Variable | Expected | Type |
|---|---|---|
| `test_phase` | `99` | uint8 |
| `loop_count` | nonzero, increasing | uint32 |
| `pclk1_freq` | `0x03D09000` | uint32 |
| `hclk_freq` | `0x03D09000` | uint32 |
| `i2c_timingr` | `0x10909CEC` | uint32 |
| `i2c_pe_bit` | `1` | uint8 |
| `pb8_idle_high` | `1` | uint8 |
| `pb9_idle_high` | `1` | uint8 |
| `pb8_can_drive_low` | `1` | uint8 |
| `pb9_can_drive_low` | `1` | uint8 |
| `pb8_is_af_mode` | `1` | uint8 |
| `pb9_is_af_mode` | `1` | uint8 |
| `pb8_is_af4` | `1` | uint8 |
| `pb9_is_af4` | `1` | uint8 |
| `pb8_is_opendrain` | `1` | uint8 |
| `pb9_is_opendrain` | `1` | uint8 |
| `gpiob_afr1` (bits 7:0) | `0x44` | uint32 |
| `bitbang_ack_addr` | `1` | uint8 |
| `bitbang_ack_reg` | `1` | uint8 |
| `bitbang_ack_rdaddr` | `1` | uint8 |
| `bitbang_whoami` | `0x6C` | uint8 |
| `status_6A` | `0` | uint8 |
| `error_after_6A` | `0` | uint32 |
| `whoami_6A` | `0x6C` | uint8 |
| `alt_status_write_6A` | `0` | uint8 |
| `alt_status_read_6A` | `0` | uint8 |
| `alt_whoami_6A` | `0x6C` | uint8 |
| `num_devices` | `1` | uint8 |
| `ext_num_devices` | `1` | uint8 |
