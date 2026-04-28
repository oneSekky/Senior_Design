/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Advanced I2C Diagnostic v2
  ******************************************************************************
  */
/* USER CODE END Header */

#include "main.h"
#include "i2c.h"
#include "usb_device.h"
#include "gpio.h"

/* USER CODE BEGIN Includes */
#include <string.h>
/* USER CODE END Includes */

/* USER CODE BEGIN PV */

// ============================================
// DIAGNOSTIC VARIABLES
// ============================================

// GPIO state when pins configured as inputs (more reliable read)
volatile uint8_t pb8_idle_high = 0;       // Should be 1 if pull-up works
volatile uint8_t pb9_idle_high = 0;       // Should be 1 if pull-up works

// Test if we can manually drive the lines
volatile uint8_t pb8_can_drive_low = 0;   // Should be 1
volatile uint8_t pb9_can_drive_low = 0;   // Should be 1

// I2C state at various points
volatile uint32_t i2c_isr_before = 0;
volatile uint32_t i2c_isr_after = 0;
volatile uint32_t i2c_cr1 = 0;
volatile uint32_t i2c_cr2 = 0;
volatile uint32_t i2c_timingr = 0;

// Address scan results
volatile uint8_t devices_found[16] = {0};
volatile uint8_t num_devices = 0;

// WHO_AM_I attempts
volatile uint8_t whoami_6A = 0;
volatile uint8_t whoami_6B = 0;
volatile uint8_t status_6A = 0xFF;
volatile uint8_t status_6B = 0xFF;
volatile uint32_t error_after_6A = 0;
volatile uint32_t error_after_6B = 0;

// Live state
volatile uint32_t loop_count = 0;
volatile uint8_t test_phase = 0;

// Clock info
volatile uint32_t pclk1_freq = 0;
volatile uint32_t hclk_freq = 0;

/* USER CODE END PV */

void SystemClock_Config(void);
void PeriphCommonClock_Config(void);

/* USER CODE BEGIN 0 */

// Configure PB8/PB9 as plain GPIO inputs to test pull-ups
void test_gpio_state(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Configure as input (no pull) - rely on external pull-ups
    GPIO_InitStruct.Pin = GPIO_PIN_8 | GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    HAL_Delay(10);

    // Read state - should be HIGH if pull-ups work
    pb8_idle_high = (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_8) == GPIO_PIN_SET) ? 1 : 0;
    pb9_idle_high = (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9) == GPIO_PIN_SET) ? 1 : 0;

    // Configure as output low to test we can drive
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;  // Open drain, like I2C
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    // Drive PB8 low
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_RESET);
    HAL_Delay(1);
    pb8_can_drive_low = (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_8) == GPIO_PIN_RESET) ? 1 : 0;
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET);
    HAL_Delay(1);

    // Drive PB9 low
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_RESET);
    HAL_Delay(1);
    pb9_can_drive_low = (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9) == GPIO_PIN_RESET) ? 1 : 0;
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_SET);
    HAL_Delay(1);
}

// Generate 9 SCL pulses to clear stuck bus
void clear_i2c_bus(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // PB8 (SCL) as open-drain output
    GPIO_InitStruct.Pin = GPIO_PIN_8;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    // PB9 (SDA) as input to read
    GPIO_InitStruct.Pin = GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    // Toggle SCL 9 times
    for (int i = 0; i < 9; i++) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_RESET);
        HAL_Delay(1);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET);
        HAL_Delay(1);
    }
}

// Reset I2C peripheral
void reset_i2c(void) {
    HAL_I2C_DeInit(&hi2c1);
    HAL_Delay(10);
    MX_I2C1_Init();
    HAL_Delay(10);
}

void run_full_scan(void) {
    num_devices = 0;
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        // Reset peripheral if it errored
        if (hi2c1.ErrorCode != 0) {
            reset_i2c();
        }

        HAL_StatusTypeDef result = HAL_I2C_IsDeviceReady(&hi2c1, addr << 1, 2, 50);
        if (result == HAL_OK) {
            if (num_devices < 16) {
                devices_found[num_devices++] = addr;
            }
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

  HAL_Delay(500);

  // ========== TEST 1: Capture clock frequencies ==========
  test_phase = 1;
  pclk1_freq = HAL_RCC_GetPCLK1Freq();
  hclk_freq = HAL_RCC_GetHCLKFreq();

  // ========== TEST 2: Capture I2C peripheral state ==========
  test_phase = 2;
  i2c_cr1 = hi2c1.Instance->CR1;
  i2c_cr2 = hi2c1.Instance->CR2;
  i2c_timingr = hi2c1.Instance->TIMINGR;
  i2c_isr_before = hi2c1.Instance->ISR;

  // ========== TEST 3: Try to clear bus first ==========
  test_phase = 3;
  clear_i2c_bus();
  HAL_Delay(10);

  // Re-init I2C after manual bit-banging
  MX_I2C1_Init();
  HAL_Delay(10);

  // ========== TEST 4: Test GPIO pull-ups directly ==========
  test_phase = 4;
  test_gpio_state();
  HAL_Delay(10);

  // Re-init I2C peripheral after GPIO test
  MX_I2C1_Init();
  HAL_Delay(10);

  // ========== TEST 5: Try WHO_AM_I at 0x6A ==========
  test_phase = 5;
  reset_i2c();
  status_6A = (uint8_t)HAL_I2C_Mem_Read(&hi2c1, 0x6A << 1, 0x0F,
                                        I2C_MEMADD_SIZE_8BIT,
                                        (uint8_t*)&whoami_6A, 1, 1000);
  error_after_6A = hi2c1.ErrorCode;
  HAL_Delay(50);

  // ========== TEST 6: Try WHO_AM_I at 0x6B ==========
  test_phase = 6;
  reset_i2c();
  status_6B = (uint8_t)HAL_I2C_Mem_Read(&hi2c1, 0x6B << 1, 0x0F,
                                        I2C_MEMADD_SIZE_8BIT,
                                        (uint8_t*)&whoami_6B, 1, 1000);
  error_after_6B = hi2c1.ErrorCode;
  HAL_Delay(50);

  // ========== TEST 7: Full address scan ==========
  test_phase = 7;
  reset_i2c();
  run_full_scan();

  // ========== TEST 8: Final state ==========
  test_phase = 8;
  i2c_isr_after = hi2c1.Instance->ISR;
  /* USER CODE END 2 */

  while (1)
  {
    /* USER CODE BEGIN 3 */
    loop_count++;
    HAL_Delay(100);
    /* USER CODE END 3 */
  }
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  HAL_PWR_EnableBkUpAccess();
  __HAL_RCC_LSEDRIVE_CONFIG(RCC_LSEDRIVE_LOW);
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE|RCC_OSCILLATORTYPE_LSE
                              |RCC_OSCILLATORTYPE_MSI|RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.LSEState = RCC_LSE_ON;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSICalibrationValue = RCC_MSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV1;
  RCC_OscInitStruct.PLL.PLLN = 32;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK4|RCC_CLOCKTYPE_HCLK2
                              |RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.AHBCLK2Divider = RCC_SYSCLK_DIV2;
  RCC_ClkInitStruct.AHBCLK4Divider = RCC_SYSCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_SMPS;
  PeriphClkInitStruct.SmpsClockSelection = RCC_SMPSCLKSOURCE_HSI;
  PeriphClkInitStruct.SmpsDivSelection = RCC_SMPSCLKDIV_RANGE0;

  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

#ifdef  USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif
