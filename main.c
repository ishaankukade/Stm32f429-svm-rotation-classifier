#include "main.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ================================================================
   STM32F429I-DISC1
   AI EMBEDDED SYSTEM LAB
   LINEAR SVM FOR ROTATION DIRECTION CLASSIFICATION
   On-board L3GD20 Gyroscope

   Classes:
   0 = CW   (Clockwise)
   1 = CCW  (Counter-Clockwise)
   2 = STILL

   HOW TO ROTATE THE BOARD
   -----------------------
   Rotate about the X axis like the SPINDLE OF A WALL CLOCK:
   the X axis points straight out of the "wall" toward you,
   and the board turns like clock hands.
   (NOT flat like a turntable.)

   Always view the board from the SAME side when deciding
   what is CW and what is CCW. The SVM learns whatever you
   demonstrate, so just be consistent in training and testing.

   Training : 150 samples = 50 CW + 50 CCW + 50 STILL
   After training: FREE-STYLE live classification
   (rotate CW / CCW / keep still in any order, as long as you like).
   The fixed 15-sample test is optional: set ENABLE_FIXED_TEST to 1.

   Features:
   0 = X (mean rate, signed -> carries the direction)
   1 = Y
   2 = Z
   3 = Magnitude

   Model:
   Linear SVM, one-vs-rest (3 linear classifiers)
       score_k = w_k . x + b_k
       prediction = argmax_k score_k
   Trained on the board with hinge-loss SGD.
   ================================================================ */


/* ================= USER SETTINGS ================= */

#define NUM_FEATURES         4
#define NUM_CLASSES          3

#define TRAIN_PER_CLASS      50
#define TEST_PER_CLASS       5

#define TRAIN_SIZE           (TRAIN_PER_CLASS * NUM_CLASSES)
#define TEST_SIZE            (TEST_PER_CLASS  * NUM_CLASSES)

#define READINGS_PER_SAMPLE  10

/* 0 = go straight to free-style live mode after training
   1 = run the fixed 15-sample test first, then free-style */
#define ENABLE_FIXED_TEST    0

#define CLOCKWISE            0
#define COUNTERCLOCKWISE     1
#define STILL                2

/* SVM training parameters */
#define SVM_EPOCHS           300
#define SVM_LEARNING_RATE    0.01f
#define SVM_LAMBDA           0.001f     /* L2 regularization */
#define SVM_POS_WEIGHT       2.0f       /* balances 1-vs-rest data */


/* ================= GYROSCOPE ================= */

#define CS_GYRO_Pin          GPIO_PIN_1
#define CS_GYRO_GPIO_Port    GPIOC

#define WHO_AM_I             0x0F
#define CTRL_REG1            0x20
#define CTRL_REG4            0x23

#define OUT_X_L              0x28
#define OUT_Y_L              0x2A
#define OUT_Z_L              0x2C


#define GYRO_CS_LOW() \
HAL_GPIO_WritePin(CS_GYRO_GPIO_Port, \
                  CS_GYRO_Pin, \
                  GPIO_PIN_RESET)

#define GYRO_CS_HIGH() \
HAL_GPIO_WritePin(CS_GYRO_GPIO_Port, \
                  CS_GYRO_Pin, \
                  GPIO_PIN_SET)


/* ================= HAL HANDLES ================= */

SPI_HandleTypeDef hspi5;
UART_HandleTypeDef huart1;


/* ================= TRAINING DATA ================= */

typedef struct
{
    float feature[NUM_FEATURES];
    uint8_t label;

} TrainingSample;

TrainingSample trainData[TRAIN_SIZE];


/* ================= LINEAR SVM ================= */

typedef struct
{
    float w[NUM_FEATURES];
    float b;

} LinearSVM;

LinearSVM svm[NUM_CLASSES];

/* Feature standardization (computed from training data) */
float featMean[NUM_FEATURES];
float featStd[NUM_FEATURES];

const char *classNames[NUM_CLASSES] =
{
    "CW",
    "CCW",
    "STILL"
};


/* ================= RANDOM SEED ================= */

uint32_t randomSeed = 12345;


/* ================================================================
   FUNCTION PROTOTYPES
   ================================================================ */

void SystemClock_Config(void);

static void MX_GPIO_Init(void);
static void MX_SPI5_Init(void);
static void MX_USART1_UART_Init(void);

void Error_Handler(void);

/* UART */
void UART_SendString(char *text);

/* Gyroscope */
void Gyro_Init(void);
uint8_t Gyro_ReadRegister(uint8_t reg);
void Gyro_WriteRegister(uint8_t reg, uint8_t data);
int16_t readAxis(uint8_t reg);
void ReadGyroXYZ(int16_t *x, int16_t *y, int16_t *z);

/* Features */
void ExtractFeatures(float feature[]);

/* Training data */
void CollectTrainingData(void);
void CollectClassData(uint8_t label,
                      int startIndex,
                      int count,
                      const char *name,
                      const char *instruction);

/* Normalization */
void ComputeNormalization(void);
void NormalizeFeatures(const float in[], float out[]);

/* SVM */
void InitializeSVM(void);
float SVM_Score(int cls, const float x[]);
void TrainOneSVM(int cls);
void TrainAllSVMs(void);
int PredictSVM(const float rawFeature[], float scores[]);

/* Testing */
void TestClassSamples(uint8_t actual,
                      const char *instruction,
                      int *correct,
                      int confusion[NUM_CLASSES][NUM_CLASSES]);
void TestSVM(void);

/* Live */
void PredictRotation(void);

/* Utility */
uint32_t RandomNumber(void);


/* ================================================================
   UART FUNCTION
   ================================================================ */

void UART_SendString(char *text)
{
    HAL_UART_Transmit(&huart1,
                      (uint8_t *)text,
                      strlen(text),
                      HAL_MAX_DELAY);
}


/* ================================================================
   RANDOM NUMBER (used only to shuffle samples during training)
   ================================================================ */

uint32_t RandomNumber(void)
{
    randomSeed = randomSeed * 1103515245UL + 12345UL;

    return (randomSeed >> 8);
}


/* ================================================================
   GYROSCOPE WRITE
   ================================================================ */

void Gyro_WriteRegister(uint8_t reg, uint8_t data)
{
    uint8_t tx[2];

    tx[0] = reg & 0x7F;
    tx[1] = data;

    GYRO_CS_LOW();
    HAL_SPI_Transmit(&hspi5, tx, 2, HAL_MAX_DELAY);
    GYRO_CS_HIGH();
}


/* ================================================================
   GYROSCOPE READ
   ================================================================ */

uint8_t Gyro_ReadRegister(uint8_t reg)
{
    uint8_t tx;
    uint8_t rx = 0;

    tx = reg | 0x80;

    GYRO_CS_LOW();
    HAL_SPI_Transmit(&hspi5, &tx, 1, HAL_MAX_DELAY);
    HAL_SPI_Receive(&hspi5, &rx, 1, HAL_MAX_DELAY);
    GYRO_CS_HIGH();

    return rx;
}


/* ================================================================
   GYROSCOPE INITIALIZATION
   ================================================================ */

void Gyro_Init(void)
{
    uint8_t whoAmI;
    char msg[60];

    whoAmI = Gyro_ReadRegister(WHO_AM_I);

    sprintf(msg, "WHO_AM_I = 0x%02X\r\n", whoAmI);
    UART_SendString(msg);

    /* Power on, enable X, Y, Z axes */
    Gyro_WriteRegister(CTRL_REG1, 0x0F);

    /* Full scale = 250 dps */
    Gyro_WriteRegister(CTRL_REG4, 0x00);

    UART_SendString("Gyroscope Initialized\r\n");
}


/* ================================================================
   READ ONE AXIS
   ================================================================ */

int16_t readAxis(uint8_t reg)
{
    uint8_t low  = Gyro_ReadRegister(reg);
    uint8_t high = Gyro_ReadRegister(reg + 1);

    return (int16_t)((high << 8) | low);
}


/* ================================================================
   READ X Y Z
   ================================================================ */

void ReadGyroXYZ(int16_t *x, int16_t *y, int16_t *z)
{
    *x = readAxis(OUT_X_L);
    *y = readAxis(OUT_Y_L);
    *z = readAxis(OUT_Z_L);
}


/* ================================================================
   FEATURE EXTRACTION

   10 raw gyro readings are averaged (signed, so the sign of X
   tells the rotation direction).

   Feature 0 = X
   Feature 1 = Y
   Feature 2 = Z
   Feature 3 = Magnitude
   ================================================================ */

void ExtractFeatures(float feature[])
{
    int i;
    int16_t x, y, z;

    int32_t sumX = 0;
    int32_t sumY = 0;
    int32_t sumZ = 0;

    for(i = 0; i < READINGS_PER_SAMPLE; i++)
    {
        ReadGyroXYZ(&x, &y, &z);

        sumX += x;
        sumY += y;
        sumZ += z;

        HAL_Delay(10);
    }

    feature[0] = (float)sumX / READINGS_PER_SAMPLE;
    feature[1] = (float)sumY / READINGS_PER_SAMPLE;
    feature[2] = (float)sumZ / READINGS_PER_SAMPLE;

    feature[3] = sqrtf(feature[0] * feature[0] +
                       feature[1] * feature[1] +
                       feature[2] * feature[2]);
}


/* ================================================================
   COLLECT ONE CLASS OF TRAINING DATA
   ================================================================ */

void CollectClassData(uint8_t label,
                      int startIndex,
                      int count,
                      const char *name,
                      const char *instruction)
{
    int i;
    char msg[160];

    UART_SendString("\r\n");
    UART_SendString((char *)instruction);
    UART_SendString("\r\nStarting in 5 seconds...\r\n");

    HAL_Delay(5000);

    UART_SendString("GO!\r\n");

    for(i = 0; i < count; i++)
    {
        int idx = startIndex + i;

        ExtractFeatures(trainData[idx].feature);
        trainData[idx].label = label;

        sprintf(msg,
                "%s %02d: X=%d, Y=%d, Z=%d, Mag=%d\r\n",
                name,
                i + 1,
                (int)trainData[idx].feature[0],
                (int)trainData[idx].feature[1],
                (int)trainData[idx].feature[2],
                (int)trainData[idx].feature[3]);

        UART_SendString(msg);
    }
}


/* ================================================================
   COLLECT TRAINING DATA
   ================================================================ */

void CollectTrainingData(void)
{
    UART_SendString("\r\n========================================\r\n");
    UART_SendString("STEP 1: TRAINING DATA COLLECTION\r\n");
    UART_SendString("========================================\r\n");
    UART_SendString("50 CW + 50 CCW + 50 STILL samples\r\n");

    CollectClassData(CLOCKWISE,
                     0,
                     TRAIN_PER_CLASS,
                     "CW   ",
                     "Rotate CLOCKWISE like a wall-clock spindle\r\n"
                     "(X axis points at you, keep turning steadily).");

    CollectClassData(COUNTERCLOCKWISE,
                     TRAIN_PER_CLASS,
                     TRAIN_PER_CLASS,
                     "CCW  ",
                     "Rotate COUNTER-CLOCKWISE like a wall-clock spindle\r\n"
                     "(same side view, opposite direction).");

    CollectClassData(STILL,
                     2 * TRAIN_PER_CLASS,
                     TRAIN_PER_CLASS,
                     "STILL",
                     "Keep the board STILL (same orientation).");

    UART_SendString("\r\nTraining data collection completed.\r\n");
}


/* ================================================================
   FEATURE NORMALIZATION (zero mean, unit variance)
   Needed so that SGD on raw gyro counts (~ +/-30000) is stable.
   ================================================================ */

void ComputeNormalization(void)
{
    int i, f;

    for(f = 0; f < NUM_FEATURES; f++)
    {
        float sum = 0.0f;
        float var = 0.0f;

        for(i = 0; i < TRAIN_SIZE; i++)
        {
            sum += trainData[i].feature[f];
        }

        featMean[f] = sum / TRAIN_SIZE;

        for(i = 0; i < TRAIN_SIZE; i++)
        {
            float d = trainData[i].feature[f] - featMean[f];
            var += d * d;
        }

        featStd[f] = sqrtf(var / TRAIN_SIZE);

        if(featStd[f] < 1.0f)
        {
            featStd[f] = 1.0f;
        }
    }
}

void NormalizeFeatures(const float in[], float out[])
{
    int f;

    for(f = 0; f < NUM_FEATURES; f++)
    {
        out[f] = (in[f] - featMean[f]) / featStd[f];
    }
}


/* ================================================================
   INITIALIZE SVM
   ================================================================ */

void InitializeSVM(void)
{
    int c, f;

    UART_SendString("\r\nSTEP 2: INITIALIZE LINEAR SVM\r\n");

    for(c = 0; c < NUM_CLASSES; c++)
    {
        for(f = 0; f < NUM_FEATURES; f++)
        {
            svm[c].w[f] = 0.0f;
        }
        svm[c].b = 0.0f;
    }

    ComputeNormalization();

    UART_SendString("SVM initialized.\r\n");
}


/* ================================================================
   SVM SCORE  :  w . x + b
   ================================================================ */

float SVM_Score(int cls, const float x[])
{
    int f;
    float s = svm[cls].b;

    for(f = 0; f < NUM_FEATURES; f++)
    {
        s += svm[cls].w[f] * x[f];
    }

    return s;
}


/* ================================================================
   TRAIN ONE BINARY SVM  (class cls  vs  rest)
   Hinge loss + L2 regularization, stochastic gradient descent.
   ================================================================ */

void TrainOneSVM(int cls)
{
    int epoch, i, f;
    int order[TRAIN_SIZE];
    float x[NUM_FEATURES];

    for(i = 0; i < TRAIN_SIZE; i++)
    {
        order[i] = i;
    }

    for(epoch = 0; epoch < SVM_EPOCHS; epoch++)
    {
        float lr = SVM_LEARNING_RATE / (1.0f + 0.01f * epoch);

        /* Shuffle (Fisher-Yates) */
        for(i = TRAIN_SIZE - 1; i > 0; i--)
        {
            int j = RandomNumber() % (i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }

        for(i = 0; i < TRAIN_SIZE; i++)
        {
            int idx = order[i];
            float y, weight, margin;

            NormalizeFeatures(trainData[idx].feature, x);

            if(trainData[idx].label == cls)
            {
                y = 1.0f;
                weight = SVM_POS_WEIGHT;
            }
            else
            {
                y = -1.0f;
                weight = 1.0f;
            }

            margin = y * SVM_Score(cls, x);

            for(f = 0; f < NUM_FEATURES; f++)
            {
                svm[cls].w[f] -= lr * SVM_LAMBDA * svm[cls].w[f];
            }

            if(margin < 1.0f)
            {
                for(f = 0; f < NUM_FEATURES; f++)
                {
                    svm[cls].w[f] += lr * weight * y * x[f];
                }
                svm[cls].b += lr * weight * y;
            }
        }
    }
}


/* ================================================================
   TRAIN ALL SVMs
   ================================================================ */

void TrainAllSVMs(void)
{
    int c;
    char msg[160];

    UART_SendString("\r\nSTEP 3: TRAIN LINEAR SVM (one-vs-rest)\r\n");

    for(c = 0; c < NUM_CLASSES; c++)
    {
        TrainOneSVM(c);

        sprintf(msg,
                "SVM %s: w=[%d, %d, %d, %d] b=%d  (x1000)\r\n",
                classNames[c],
                (int)(svm[c].w[0] * 1000),
                (int)(svm[c].w[1] * 1000),
                (int)(svm[c].w[2] * 1000),
                (int)(svm[c].w[3] * 1000),
                (int)(svm[c].b * 1000));

        UART_SendString(msg);
    }

    UART_SendString("Training completed.\r\n");
}


/* ================================================================
   PREDICT  (argmax of the 3 linear scores)
   ================================================================ */

int PredictSVM(const float rawFeature[], float scores[])
{
    int c;
    int best = 0;
    float x[NUM_FEATURES];

    NormalizeFeatures(rawFeature, x);

    for(c = 0; c < NUM_CLASSES; c++)
    {
        scores[c] = SVM_Score(c, x);

        if(scores[c] > scores[best])
        {
            best = c;
        }
    }

    return best;
}


/* ================================================================
   TEST ONE CLASS
   ================================================================ */

void TestClassSamples(uint8_t actual,
                      const char *instruction,
                      int *correct,
                      int confusion[NUM_CLASSES][NUM_CLASSES])
{
    int i, prediction;
    float feature[NUM_FEATURES];
    float scores[NUM_CLASSES];
    char msg[220];

    UART_SendString("\r\nTEST: ");
    UART_SendString((char *)classNames[actual]);
    UART_SendString("\r\n");
    UART_SendString((char *)instruction);
    UART_SendString("\r\nStarting in 5 seconds...\r\n");

    HAL_Delay(5000);

    for(i = 0; i < TEST_PER_CLASS; i++)
    {
        ExtractFeatures(feature);

        prediction = PredictSVM(feature, scores);

        confusion[actual][prediction]++;

        if(prediction == actual)
        {
            (*correct)++;
        }

        sprintf(msg,
                "Test %02d: X=%d, Y=%d, Z=%d, Mag=%d | Actual=%s | Predicted=%s\r\n",
                i + 1,
                (int)feature[0],
                (int)feature[1],
                (int)feature[2],
                (int)feature[3],
                classNames[actual],
                classNames[prediction]);

        UART_SendString(msg);
    }
}


/* ================================================================
   TEST SVM
   Accuracy printed with INTEGER arithmetic (works even if
   float printf is disabled in STM32CubeIDE).
   ================================================================ */

void TestSVM(void)
{
    int correct = 0;
    int a, p;
    int confusion[NUM_CLASSES][NUM_CLASSES];
    int accuracyHundredths;
    char msg[150];

    memset(confusion, 0, sizeof(confusion));

    UART_SendString("\r\n========================================\r\n");
    UART_SendString("OPTIONAL: FIXED LINEAR SVM TEST\r\n");
    UART_SendString("========================================\r\n");

    TestClassSamples(CLOCKWISE,
                     "Rotate CLOCKWISE (wall-clock spindle).",
                     &correct, confusion);

    TestClassSamples(COUNTERCLOCKWISE,
                     "Rotate COUNTER-CLOCKWISE (wall-clock spindle).",
                     &correct, confusion);

    TestClassSamples(STILL,
                     "Keep the board STILL.",
                     &correct, confusion);

    accuracyHundredths = (correct * 10000) / TEST_SIZE;

    sprintf(msg, "\r\nCorrect Samples = %d / %d\r\n", correct, TEST_SIZE);
    UART_SendString(msg);

    sprintf(msg, "Accuracy = %d.%02d %%\r\n",
            accuracyHundredths / 100,
            accuracyHundredths % 100);
    UART_SendString(msg);

    /* Confusion matrix */
    UART_SendString("\r\nConfusion matrix (rows = actual, cols = predicted)\r\n");
    UART_SendString("          CW   CCW  STILL\r\n");

    for(a = 0; a < NUM_CLASSES; a++)
    {
        sprintf(msg, "%-6s", classNames[a]);
        UART_SendString(msg);

        for(p = 0; p < NUM_CLASSES; p++)
        {
            sprintf(msg, "%5d ", confusion[a][p]);
            UART_SendString(msg);
        }
        UART_SendString("\r\n");
    }

    UART_SendString("========================================\r\n");
}


/* ================================================================
   LIVE PREDICTION
   ================================================================ */

void PredictRotation(void)
{
    float feature[NUM_FEATURES];
    float scores[NUM_CLASSES];
    int prediction;
    char msg[200];

    ExtractFeatures(feature);

    prediction = PredictSVM(feature, scores);

    sprintf(msg,
            "LIVE: X=%d, Y=%d, Z=%d, Mag=%d | score CW=%d CCW=%d STILL=%d (x100) -> %s\r\n",
            (int)feature[0],
            (int)feature[1],
            (int)feature[2],
            (int)feature[3],
            (int)(scores[CLOCKWISE] * 100),
            (int)(scores[COUNTERCLOCKWISE] * 100),
            (int)(scores[STILL] * 100),
            classNames[prediction]);

    UART_SendString(msg);
}


/* ================================================================
   MAIN
   ================================================================ */

int main(void)
{
    HAL_Init();

    SystemClock_Config();

    MX_GPIO_Init();
    MX_SPI5_Init();
    MX_USART1_UART_Init();

    Gyro_Init();

    UART_SendString("\r\n========================================\r\n");
    UART_SendString("LINEAR SVM AI EXPERIMENT\r\n");
    UART_SendString("CW / CCW / STILL  (rotation about X axis)\r\n");
    UART_SendString("STM32F429I-DISC1\r\n");
    UART_SendString("========================================\r\n");

    /* STEP 1 */
    CollectTrainingData();

    /* STEP 2 */
    InitializeSVM();

    /* STEP 3 */
    TrainAllSVMs();

#if ENABLE_FIXED_TEST
    /* STEP 4 (optional): fixed 15-sample test */
    TestSVM();
#endif

    /* FREE-STYLE LIVE MODE */
    UART_SendString("\r\n========================================\r\n");
    UART_SendString("FREE-STYLE LIVE CLASSIFICATION\r\n");
    UART_SendString("Rotate CW / CCW or keep still - any order,\r\n");
    UART_SendString("as long as you like. Starting in 3 seconds...\r\n");
    UART_SendString("========================================\r\n");

    HAL_Delay(3000);

    while(1)
    {
        PredictRotation();
    }
}


/* ================================================================
   SYSTEM CLOCK
   ================================================================ */

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    RCC_OscInitStruct.OscillatorType      = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState            = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState        = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource       = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM            = 8;
    RCC_OscInitStruct.PLL.PLLN            = 180;
    RCC_OscInitStruct.PLL.PLLP            = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ            = 4;

    if(HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler();
    }

    if(HAL_PWREx_EnableOverDrive() != HAL_OK)
    {
        Error_Handler();
    }

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK |
                                  RCC_CLOCKTYPE_SYSCLK |
                                  RCC_CLOCKTYPE_PCLK1 |
                                  RCC_CLOCKTYPE_PCLK2;

    RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

    if(HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
    {
        Error_Handler();
    }
}


/* ================================================================
   SPI5 INITIALIZATION
   ================================================================ */

static void MX_SPI5_Init(void)
{
    hspi5.Instance               = SPI5;
    hspi5.Init.Mode              = SPI_MODE_MASTER;
    hspi5.Init.Direction         = SPI_DIRECTION_2LINES;
    hspi5.Init.DataSize          = SPI_DATASIZE_8BIT;
    hspi5.Init.CLKPolarity       = SPI_POLARITY_HIGH;
    hspi5.Init.CLKPhase          = SPI_PHASE_2EDGE;
    hspi5.Init.NSS               = SPI_NSS_SOFT;
    hspi5.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
    hspi5.Init.FirstBit          = SPI_FIRSTBIT_MSB;
    hspi5.Init.TIMode            = SPI_TIMODE_DISABLE;
    hspi5.Init.CRCCalculation    = SPI_CRCCALCULATION_DISABLE;
    hspi5.Init.CRCPolynomial     = 10;

    if(HAL_SPI_Init(&hspi5) != HAL_OK)
    {
        Error_Handler();
    }
}


/* ================================================================
   USART1 INITIALIZATION
   ================================================================ */

static void MX_USART1_UART_Init(void)
{
    huart1.Instance          = USART1;
    huart1.Init.BaudRate     = 9600;
    huart1.Init.WordLength   = UART_WORDLENGTH_8B;
    huart1.Init.StopBits     = UART_STOPBITS_1;
    huart1.Init.Parity       = UART_PARITY_NONE;
    huart1.Init.Mode         = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;

    if(HAL_UART_Init(&huart1) != HAL_OK)
    {
        Error_Handler();
    }
}


/* ================================================================
   GPIO INITIALIZATION
   ================================================================ */

static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOF_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();

    HAL_GPIO_WritePin(CS_GYRO_GPIO_Port, CS_GYRO_Pin, GPIO_PIN_SET);

    GPIO_InitStruct.Pin   = CS_GYRO_Pin;
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull  = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(CS_GYRO_GPIO_Port, &GPIO_InitStruct);

    GPIO_InitStruct.Pin  = GPIO_PIN_1 | GPIO_PIN_2;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}


/* ================================================================
   ERROR HANDLER
   ================================================================ */

void Error_Handler(void)
{
    __disable_irq();

    while(1)
    {
    }
}


#ifdef USE_FULL_ASSERT

void assert_failed(uint8_t *file, uint32_t line)
{
    /* User can add error reporting here. */
}

#endif
