# Project Context & Handoff Notes

Comprehensive context for continuing work on this senior design project. This document is intended to be fed to another AI assistant or human collaborator to bring them up to speed.

## Project Overview

Senior design project (Albert Wang). Custom PCB with:

- **STM32WB55CGU6** in UFQFPN48 package (1 MB flash, dual-core M4+M0+, BLE-capable)
- **LSM6DSOTR IMU** in LGA-14L package (accel + gyro, I2C/SPI capable)
- 32 MHz HSE crystal, 32.768 kHz LSE crystal
- Two USB-C connectors (USB#1 = power/charging, USB#2 = data)
- USB-C CC1/CC2 each with 5.1kΩ pulldowns to GND

### Project Paths

- Project root: `/Users/albertwang/Desktop/WB55_IMU_USB`
- Hex file: `/Users/albertwang/Desktop/WB55_IMU_USB/Debug/WB55_IMU_USB.hex`
- Map file: `/Users/albertwang/Desktop/WB55_IMU_USB/Debug/WB55_IMU_USB.map`
- Main source: `/Users/albertwang/Desktop/WB55_IMU_USB/Core/Src/main.c`

### Dev Environment

- macOS, MacBook Pro
- STM32CubeIDE 2.1.1 (build)
- STM32CubeProgrammer v2.22.0 (flash + memory inspection)
- ST-LINK V2 clone (SN: 30303030..., FW: V2J37S7), hardwired to board with 5 soldered wires

## Pin Assignments

| Function | Pin | Notes |
|---|---|---|
| I2C1 SCL | PB8 | 4.7kΩ external pull-up |
| I2C1 SDA | PB9 | 4.7kΩ external pull-up |
| USB D- | PA11 | Hardware issue on this board |
| USB D+ | PA12 | Working |
| SWDIO | PA13 | Schematic error: also tied to USB#1 D+ |
| SWCLK | PA14 | |
| NRST | Chip pin 7 | |
| BOOT0 | Chip pin 47 | |

### Schematic Error To Note

**PA13 (SWDIO) was incorrectly connected to USB#1's D+ on the original PCB.** This causes a USB#1 + SWD conflict. Not blocking, but requires unplugging USB#1 during SWD operations.

## What Happened (Chronological Summary)

### Phase 1: USB DFU Debugging On Original Board (FAILED)

- Goal: flash via USB DFU bootloader (BOOT0 high)
- USB-C connector had cold solder joints on CC pins
- Wall charger worked in 1 of 4 cable orientations (overly-strong pull-ups muscled through bad joint)
- Mac with spec-compliant pull-ups couldn't enumerate at all
- Scope on D+ showed Mac sending bus reset + 3 SETUP retries (textbook enumeration attempt), then giving up
- D- showed no signal during these attempts → suggested broken D- trace/solder
- Verified: BOOT0 was actually high at chip pin, voltage rails all correct (3.22V), NRST high, chip wasn't dead

### Phase 2: Bodge To USB#2 On A Different Board (PARTIAL SUCCESS)

- Tried jumping the broken USB-C connector to a working USB-C#2 connector on another board
- Got further (Mac saw something via USB-A → USB-C adapter chain) but enumeration still incomplete
- Scope confirmed D+ pulses but D- still flat → D- still not properly connected somewhere in the chain
- Mac terminal (`log stream`, `ioreg`, `system_profiler`) showed nothing — USB stack didn't even register the device

### Phase 3: Abandoned USB, Switched To ST-LINK SWD (SUCCESS)

- Hardwired ST-LINK V2 clone to board: GND, 3.3V, PA13, PA14, NRST
- Initial connection failed because CubeProgrammer defaulted to JTAG; STM32WB55 only supports SWD
- After switching to SWD + 1800 kHz + Under reset + Hardware reset, connection succeeded
- Chip identified as STM32WB5x/35xx, Device ID 0x495, Cortex-M4
- Successfully flashed first firmware version

### Phase 4: Code Iteration Via SWD

**v1 (original USB CDC code):** Hung on `CDC_Transmit_FS` waiting for USB enumeration that never happened. `whoami` stayed at 0x00, code never reached I2C read.

**v2 (USB CDC removed):** Added debug globals (`whoami`, `i2c_status`, `live_ax/ay/az`, `loop_counter`, `status_code`). Build initially failed because `SystemClock_Config` and `Error_Handler` were accidentally deleted — added back full PLL config. Build succeeded: 32200 bytes text. Result: `i2c_status=0x01` (HAL_ERROR), `whoami=0x00`, `loop_counter=1541` (proves loop running, but I2C failing).

**v3 (current diagnostic):** Comprehensive I2C diagnostic with peripheral reset between attempts, bus clearing pulses, GPIO state tests, full address scan, captures CR1/CR2/TIMINGR/ISR registers.

## Current Diagnosis: I2C ACK FAILURE

After running v3 diagnostic firmware, results:

### ✅ What Works

- `pb8_idle_high = 1` (pull-up working on SCL)
- `pb9_idle_high = 1` (pull-up working on SDA)
- `pb8_can_drive_low = 1` (chip can sink PB8)
- `pb9_can_drive_low = 1` (chip can sink PB9)
- `test_phase = 8` (all tests completed)
- `loop_count = 956` (main loop running)
- `pclk1_freq = 0x03D09000 = 64 MHz` ✓
- `hclk_freq = 64 MHz` ✓

### ❌ What Doesn't Work

- `num_devices = 0` (NO devices found in full bus scan)
- `whoami_6A = 0`, `whoami_6B = 0`
- ⭐ **`error_after_6A = 0x00000004 = HAL_I2C_ERROR_AF` (ACK FAILURE)**
- ⭐ **`error_after_6B = 0x00000004 = HAL_I2C_ERROR_AF`**

### Interpretation

The I2C bus is **electrically perfect**. STM32 successfully drives START + address byte. But **no slave acknowledges on the 9th clock pulse**. The IMU is simply not responding.

## Most Likely Root Causes (In Priority Order)

1. **SDA/SCL swapped at IMU on PCB** — PB8 should go to LSM6DSO pin 14 (SCL), PB9 to pin 15 (SDA). If swapped, IMU sees scrambled data.

2. **IMU CS pin not tied to 3.3V** — LSM6DSO pin 13 must be HIGH for I2C mode (else SPI mode, won't respond to I2C).

3. **IMU SDO/SA0 (pin 16) state** — 0V = address 0x6A, 3.3V = 0x6B (both tested in firmware, both failed).

4. **Bad solder on IMU LGA-14L** — hand-soldered LGA easy to damage/misalign.

5. **IMU damaged** from soldering heat or ESD.

6. **Trace broken** between chip PB8/PB9 and IMU (despite continuity test passing — DC continuity can pass even with high-impedance fault).

## LSM6DSO LGA-14L Pinout Reference

| Pin | Function |
|---|---|
| 7-8 | GND |
| 9 | VDDIO |
| 10 | INT2 |
| 11 | INT1 |
| 12 | VDD |
| 13 | CS (must be HIGH for I2C mode) |
| 14 | SCL/SPC |
| 15 | SDA/SDI/SDO |
| 16 | SDO/SA0 (LOW=0x6A address, HIGH=0x6B) |

## Pending Action Items

### Immediate Hardware Verification

- [ ] Continuity test PB8 → IMU pin 14 (SCL): must beep
- [ ] Continuity test PB9 → IMU pin 15 (SDA): must beep
- [ ] Verify NOT swapped (PB8 to SDA pin would be the bug)
- [ ] IMU CS pin (13) voltage: must read 3.3V
- [ ] IMU SDO/SA0 pin (16) voltage: 0V for 0x6A address
- [ ] IMU SCL pin (14) voltage chip running: should be 3.3V idle
- [ ] IMU SDA pin (15) voltage chip running: should be 3.3V idle
- [ ] If IMU side reads 0V but chip side reads 3.3V → broken trace

### If Hardware Tests Reveal Issue

- [ ] Bodge wire to fix SDA/SCL swap (if confirmed)
- [ ] Tie CS to 3.3V via wire (if floating)
- [ ] Reflow LSM6DSO LGA pads (if cold joint)
- [ ] Replace IMU (if damaged)

### After I2C Works

- [ ] Re-flash with non-diagnostic firmware (read accel/gyro into `live_ax/ay/az/gx/gy/gz` globals)
- [ ] Verify `whoami = 0x6C`
- [ ] Verify `live_*` values change when board moved
- [ ] Optional: fix D- with bodge wire for USB CDC functionality
- [ ] Optional: SWO printf alternative if USB CDC needed
- [ ] BLE NOT recommended unless project requires it (1-2 weeks of work)

## Reference Documents

- WB55 datasheet: DS11929
- LSM6DSO datasheet: DS12140
- USB-C connector: USB4110 spec from gct.co
- AN2606: STM32 system memory boot
