# I2C Diagnostic Firmware - How It Works

A diagnostic firmware for the STM32WB55 + LSM6DSO IMU board that tests the I2C bus and stores results in global RAM variables. Since USB CDC isn't available on this board (D- hardware issue), all output is read via SWD/ST-LINK by inspecting memory addresses in STM32CubeProgrammer.

## What This Code Is

The firmware runs a sequence of automated tests on the I2C bus on startup, then enters an infinite loop. All test results are stored in `volatile` global variables that can be read at known RAM addresses via the SWD debug interface.

## What It Tests

The code runs **8 sequential test phases**, with the `test_phase` variable updated after each so you can tell where it stopped if anything hangs:

1. **Capture clock frequencies** — verifies PCLK1 and HCLK are both 64 MHz (proves system clock is correct)
2. **Capture I2C peripheral registers** — reads CR1, CR2, TIMINGR, ISR to verify I2C is configured properly
3. **Manual bus clear** — bit-bangs 9 SCL pulses to wake up any stuck I2C slaves
4. **GPIO pull-up test** — temporarily reconfigures PB8/PB9 as plain GPIO inputs (no internal pull) and reads them. If they read HIGH, external pull-ups are working. Then drives them low to verify the chip can sink current.
5. **WHO_AM_I read at 0x6A** — attempts to read register 0x0F from address 0x6A
6. **WHO_AM_I read at 0x6B** — same attempt at the alternate address (in case SDO/SA0 is high)
7. **Full bus scan** — pings every I2C address from 0x08 to 0x77 looking for any device that ACKs
8. **Final state capture** — reads I2C ISR register again to see post-test status

After tests complete, it enters a `while(1)` loop that increments `loop_count` so you can confirm code is alive.

## How The Code Interacts With The MCU

- **CubeMX setup** handles peripheral init: I2C1 on PB8/PB9 with alternate function, GPIO on other pins, system clock at 64 MHz
- **HAL_Init()** sets up SysTick, NVIC, basic infrastructure
- **MX_I2C1_Init()** configures I2C peripheral with timing register for 100 kHz operation
- **MX_GPIO_Init()** sets PB8/PB9 to alternate function 4 (I2C1) with open-drain mode
- The diagnostic code temporarily reconfigures PB8/PB9 as GPIO during test 4, then re-runs `MX_I2C1_Init` to restore I2C function

## Variables Stored In Global RAM

All declared as `volatile` so the compiler doesn't optimize them away. Read via SWD by typing the address into CubeProgrammer's Memory tab.

### Key Variables

| Variable | Purpose |
|---|---|
| `pb8_idle_high`, `pb9_idle_high` | Pull-up status (1 = pull-up working) |
| `pb8_can_drive_low`, `pb9_can_drive_low` | Chip can sink current (1 = OK) |
| `error_after_6A`, `error_after_6B` | I2C error code after each WHO_AM_I attempt |
| `num_devices`, `devices_found[16]` | Bus scan results |
| `whoami_6A`, `whoami_6B` | Values read (should be 0x6C if IMU works) |
| `pclk1_freq`, `hclk_freq` | Clock verification |
| `test_phase` | Confirms how far code progressed (should be 8) |
| `loop_count` | Confirms main loop is running (increments forever) |
| `i2c_cr1`, `i2c_cr2`, `i2c_timingr` | I2C peripheral register snapshots |
| `i2c_isr_before`, `i2c_isr_after` | I2C ISR before/after tests |

### HAL_StatusTypeDef Values

For interpreting `error_after_*`, `status_*` variables:

| Value | Meaning |
|---|---|
| `0x00` | HAL_OK (success) |
| `0x01` | HAL_I2C_ERROR_BERR (bus error) |
| `0x02` | HAL_I2C_ERROR_ARLO (arbitration lost) |
| `0x04` | HAL_I2C_ERROR_AF (ACK failure - slave didn't respond) |
| `0x08` | HAL_I2C_ERROR_OVR (overrun) |

## Why USB CDC Was Removed

The original code used `CDC_Transmit_FS()` which **blocks** waiting for USB enumeration. Since the board's D- line has hardware issues, USB never enumerates, and the blocking call hangs the firmware before it ever reaches the I2C code. Removing USB CDC entirely lets the diagnostic run reliably.

## Reading Results

After flashing and letting the chip run for ~5 seconds:

1. Find variable addresses in the `.map` file using `grep`
2. Open Memory tab in STM32CubeProgrammer
3. Read at the relevant address (typically `0x20000008` and around `0x20000088`)
4. Decode multi-byte values as **little-endian** (least significant byte first)

## Diagnostic Decision Tree

| Observation | Diagnosis |
|---|---|
| `pb8_idle_high = 0` or `pb9_idle_high = 0` | Pull-up missing or line shorted to GND |
| Both pull-ups OK, all addresses fail with `error = 0x04` | ACK failure - IMU not responding (wiring, CS pin, or damaged IMU) |
| `error = 0x01` (BERR) | Bus error - electrical problem |
| `num_devices > 0` | Some device exists at the listed addresses |
| `whoami_6A = 0x6C` | IMU is alive and working at address 0x6A |
| `test_phase < 8` | Code hung before completing all tests |
| `loop_count` not increasing | Main loop not running |
