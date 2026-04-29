# ST-LINK Setup & Flashing Workflow

Complete workflow for flashing and debugging the STM32WB55 board via ST-LINK SWD.

## Hardware Connection

**5 wires hardwired/soldered to the board:**

| ST-LINK Pin | Board Pin | Purpose |
|---|---|---|
| GND | GND | Ground reference |
| 3.3V | 3.3V net | Powers board (USB#1 unplugged) |
| SWDIO | PA13 (chip pin 34) | SWD data |
| SWCLK | PA14 (chip pin 35) | SWD clock |
| NRST | NRST (chip pin 7) | Reset control |

## Step-By-Step Flash Workflow

### 1. Build Code In STM32CubeIDE

- Project → Build All (hammer icon)
- Verify "0 errors, 0 warnings"
- Output: `Debug/WB55_IMU_USB.hex`

### 2. Connect ST-LINK To Mac

- Plug ST-LINK into Mac via its USB cable
- Board powers up automatically (3.3V wire)
- Verify multimeter reads ~3.22V on board's 3.3V net

### 3. Open STM32CubeProgrammer

### 4. Configure Connection Settings (right panel)

| Setting | Value |
|---|---|
| Port | **SWD** (NOT JTAG — STM32WB55 doesn't support JTAG) |
| Frequency | **1800 kHz** (most stable for this clone ST-LINK) |
| Mode | **Under reset** (for flashing/erasing) |
| Reset mode | **Hardware reset** |
| Speed | Reliable |
| Shared | Disabled |

### 5. Connect To Chip

- Click green **Connect** button (top right)
- Should show "Connected" with green indicator
- Target info populates: STM32WB5x/35xx, Device ID 0x495, Flash 1MB

### 6. Flash The Firmware

**Use the TOP "Download" section, NOT the bottom "Automatic Mode" section!**

- File path: Browse → select `WB55_IMU_USB.hex`
- ☑ Verify programming
- ☑ Run after programming
- ☐ Skip flash erase (UNCHECKED)
- Click **Start Programming**
- Wait for "Download verified successfully"

### 7. Switch To Read Mode (After Flashing)

- Click red **Disconnect** button
- Wait ~5 seconds for code to run through tests
- Change settings:
  - Mode: **Hot plug**
  - Reset mode: **Software reset**
- Click **Connect**

### 8. Find Variable RAM Addresses

The `.map` file lists every global variable's RAM address:

```bash
grep -E "VARIABLE_NAME_PATTERN" /Users/albertwang/Desktop/WB55_IMU_USB/Debug/WB55_IMU_USB.map
```

Replace `VARIABLE_NAME_PATTERN` with the names you care about (e.g., `whoami|loop_count|test_phase`).

### 9. Read Memory

- Click "+" tab → **Open memory tab**
- Address: paste the variable address (e.g., `0x20000008`)
- Size: enough bytes to cover all variables (e.g., `200`)
- Data width: `8-bit`
- Click **Read**
- Hex dump appears
- Decode multi-byte values using **little-endian** (least significant byte first)

## Erasing The Chip

If you need to fully erase:

1. Connect with **Under reset + Hardware reset** (halts CPU)
2. In Erasing & Programming tab, right side: click **Full chip erase**
3. Wait for "Mass erase successfully achieved"

## Common Pitfalls (Hard-Won Lessons)

### Don't Use "Start Automatic Mode"

The button at the **bottom** of the Erasing & Programming panel is for production lines. It will repeatedly mass-erase the chip in a loop, asking you to disconnect/connect new devices. **Always use the top "Start Programming" button.**

### Don't Keep USB#1 Plugged In During SWD Flashing

PA13 is connected to USB#1's D+ on this board (design error). When USB#1 is plugged in, the host's D+ signaling fights against ST-LINK's SWD signaling on PA13. **Power from ST-LINK's 3.3V instead.**

### Default Port Is JTAG

CubeProgrammer defaults to JTAG, but **STM32WB55 only supports SWD**. Must manually change the Port dropdown.

### 9000 kHz Is Too Fast

Clone ST-LINKs are unreliable at high frequencies. **Drop to 1800 kHz** for stable connections.

### "Under reset" Halts The CPU

When you connect with Under reset mode, the CPU is halted and your code doesn't run. To execute code:
- Click **Run** in MCU Core view, OR
- Disconnect and reconnect with **Hot plug + Software reset**

### "Failed To Erase Memory"

If this appears after a successful flash, the chip is running. Disconnect and reconnect with **Under reset + Hardware reset** to halt it before erasing.

### "Unable To Get Core ID" / "DEV_TARGET_CMD_ERR"

Usually means:
1. Port set to JTAG (change to SWD)
2. Frequency too high (drop to 1800 kHz)
3. USB#1 plugged in causing PA13 conflict (unplug it)
4. Wires not making good contact

## Quick Reference: Mode Selection

| Goal | Mode | Reset Mode |
|---|---|---|
| Flash new firmware | Under reset | Hardware reset |
| Erase chip | Under reset | Hardware reset |
| Read live RAM while code runs | Hot plug | Software reset |
| Inspect halted CPU registers | Under reset | Hardware reset |

## Recovery: If Nothing Works

1. **Disconnect ST-LINK from Mac entirely** (unplugs power to board)
2. Wait 10 seconds
3. Plug ST-LINK back in
4. Open CubeProgrammer fresh
5. Set Port=SWD, Frequency=1800, Mode=Under reset, Reset=Hardware reset
6. Click Connect
