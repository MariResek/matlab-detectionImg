// main.cpp - equivalente a probar.m, para Debian aarch64 en QEMU.
// API generada por MATLAB Coder: buffers fijos (bounded arrays).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// Headers generados por MATLAB Coder
#include "yolov2_detect.h"
#include "yolov2_detect_initialize.h"
#include "yolov2_detect_terminate.h"

// stb_image / stb_image_write
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Deben coincidir con imgType del codegen: [720 1280 3] -> 720*1280*3 = 2764800
static constexpr int H = 720;
static constexpr int W = 1280;
static constexpr int C = 3;
static constexpr int IN_SIZE = H * W * C;

// Máximo de detecciones que la red puede devolver (visto en categorical.h: 845)
static constexpr int MAX_DET = 1024;

// Clases COCO (80), mismo orden que detector.ClassNames del .mat
static const char* CLASS_NAMES[] = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
static constexpr int NUM_CLASSES = sizeof(CLASS_NAMES) / sizeof(CLASS_NAMES[0]);

static void drawRect(uint8_t* img, int Hh, int Ww,
                     int x, int y, int w, int h,
                     uint8_t r, uint8_t g, uint8_t b, int thick = 3) {
    auto px = [&](int xi, int yi) {
        if (xi < 0 || yi < 0 || xi >= Ww || yi >= Hh) return;
        uint8_t* p = img + (yi * Ww + xi) * 3;
        p[0] = r; p[1] = g; p[2] = b;
    };
    for (int t = 0; t < thick; ++t) {
        for (int xi = x; xi < x + w; ++xi) { px(xi, y + t); px(xi, y + h - 1 - t); }
        for (int yi = y; yi < y + h; ++yi) { px(x + t, yi); px(x + w - 1 - t, yi); }
    }
}

int main(int argc, char** argv) {
    const char* inPath  = (argc > 1) ? argv[1] : "gato.png";
    const char* outPath = (argc > 2) ? argv[2] : "out.png";

    // --- 1) Leer imagen ---
    int w, h, ch;
    uint8_t* raw = stbi_load(inPath, &w, &h, &ch, 3);
    if (!raw) {
        std::fprintf(stderr, "No pude abrir %s\n", inPath);
        return 1;
    }
    if (w != W || h != H) {
        std::fprintf(stderr,
            "Tamano de imagen (%dx%d) != esperado (%dx%d). "
            "Redimensiona o regenera con imgType variable.\n",
            w, h, W, H);
        stbi_image_free(raw);
        return 2;
    }

    // --- 2) Convertir a layout MATLAB: column-major, planar HxWx3 uint8 ---
    // stb: row-major, RGB interleaved (idx = (y*W + x)*3 + c)
    // MATLAB: in(y,x,c) -> idx = (c*W + x)*H + y
    static uint8_t in[IN_SIZE];
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                in[(c * W + x) * H + y] = raw[(y * W + x) * 3 + c];
            }
        }
    }

    // --- 3) Buffers de salida (pre-alocados) ---
    static double       bboxes_data[MAX_DET * 4];  // Nx4 column-major, double
    static float        scores_data[MAX_DET];
    static unsigned int labels_data[MAX_DET];      // uint32, 1-based
    int bboxes_size[2] = {0, 0};
    int scores_size[1] = {0};
    int labels_size[1] = {0};

    // --- 4) Inferencia ---
    std::printf("Inicializando red...\n");
    yolov2_detect_initialize();

    std::printf("Corriendo yolov2_detect... (puede tardar mucho en QEMU)\n");
    yolov2_detect(in,
                  bboxes_data, bboxes_size,
                  scores_data, scores_size,
                  labels_data, labels_size);

    const int N = bboxes_size[0];
    std::printf("Detecciones: %d\n", N);

    // --- 5) Dibujar ---
    // bboxes_data es Nx4 column-major -> col c fila i en idx (c*N + i)
    for (int i = 0; i < N; ++i) {
        double bx = bboxes_data[0 * N + i];
        double by = bboxes_data[1 * N + i];
        double bw = bboxes_data[2 * N + i];
        double bh = bboxes_data[3 * N + i];
        float  sc = scores_data[i];
        unsigned int lb = labels_data[i];  // 1-based
        const char* name = (lb >= 1 && (int)lb <= NUM_CLASSES)
                           ? CLASS_NAMES[lb - 1] : "?";
        std::printf("  #%d  lb=%u  %-10s  score=%.3f  bbox=[%.1f %.1f %.1f %.1f]\n",
                    i, lb, name, sc, bx, by, bw, bh);

        drawRect(raw, H, W,
                 (int)bx - 1, (int)by - 1, (int)bw, (int)bh,
                 0, 255, 0);
    }

    // --- 6) Guardar PNG anotado ---
    if (!stbi_write_png(outPath, W, H, 3, raw, W * 3)) {
        std::fprintf(stderr, "No pude escribir %s\n", outPath);
    } else {
        std::printf("Guardado: %s\n", outPath);
    }

    yolov2_detect_terminate();
    stbi_image_free(raw);
    return 0;
}
