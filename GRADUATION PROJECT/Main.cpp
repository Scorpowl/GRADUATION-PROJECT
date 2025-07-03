// ICBYTES ile Lojistik Regresyon Projesi
// Adým 1: Arayüz Kurulumu ve Veri Yükleme
// Yazan: Gemini (Google AI) & Proje Sahibi

#include "ic_media.h"
#include "icbytes.h"

// ICBYTES ile Lojistik Regresyon Projesi
// Final Sürümü
// Yazan: Gemini (Google AI) & Proje Sahibi

#include "icb_gui.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>

// --- Global Deðiþkenler ---
int MLE;
ICBYTES X_train;
ICBYTES y_train;
ICBYTES theta;
double learning_rate = 0.01;
int iterations = 1500; // Veri seti büyüdüðü için iterasyon sayýsýný biraz artýrabiliriz.

// --- Yardýmcý Fonksiyonlar ---

/**
 * @brief Bir CSV dosyasýný okuyup içeriðini bir ICBYTES matrisine aktarýr.
 */
bool ReadCSVtoICBYTES(const std::string& filename, ICBYTES& matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        ICG_printf(MLE, "HATA: %s dosyasi acilamadi!\n", filename.c_str());
        return false;
    }
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            }
            catch (const std::invalid_argument& ia) {
                ICG_printf(MLE, "HATA: Gecersiz veri bulundu: %s\n", value.c_str());
                return false;
            }
        }
        data.push_back(row);
    }
    file.close();
    if (data.empty()) {
        ICG_printf(MLE, "HATA: %s dosyasi bos veya okunamadi.\n", filename.c_str());
        return false;
    }
    long long num_rows = data.size();
    long long num_cols = data[0].size();
    CreateMatrix(matrix, num_cols, num_rows, ICB_DOUBLE);
    for (long long r = 0; r < num_rows; ++r) {
        for (long long c = 0; c < num_cols; ++c) {
            matrix.D(c + 1, r + 1) = data[r][c];
        }
    }
    return true;
}

/**
 * @brief Lojistik regresyon için yapay bir veri seti oluþturur, karýþtýrýr ve
 * eðitim/test olarak ayýrýp CSV dosyalarýna kaydeder.
 */
void GenerateAndSaveData()
{
    ICBYTES class0_features, class1_features;
    int samples_per_class = 100;

    RandomNormal(2.0, 1.5, class0_features, 2, samples_per_class);
    RandomNormal(7.0, 1.5, class1_features, 2, samples_per_class);

    ICBYTES all_features, all_labels;
    CreateMatrix(all_features, 2, samples_per_class * 2, ICB_DOUBLE);
    CreateMatrix(all_labels, 1, samples_per_class * 2, ICB_DOUBLE);

    for (int i = 1; i <= samples_per_class; ++i) {
        all_features.D(1, i) = class0_features.D(1, i);
        all_features.D(2, i) = class0_features.D(2, i);
        all_labels.D(1, i) = 0.0;
    }
    for (int i = 1; i <= samples_per_class; ++i) {
        all_features.D(1, i + samples_per_class) = class1_features.D(1, i);
        all_features.D(2, i + samples_per_class) = class1_features.D(2, i);
        all_labels.D(1, i + samples_per_class) = 1.0;
    }

    std::mt19937 rng(std::random_device{}());
    for (int i = all_features.Y(); i > 1; --i) {
        std::uniform_int_distribution<int> dist(1, i);
        int j = dist(rng);
        for (int c = 1; c <= all_features.X(); ++c) {
            std::swap(all_features.D(c, i), all_features.D(c, j));
        }
        std::swap(all_labels.D(1, i), all_labels.D(1, j));
    }

    int train_size = (samples_per_class * 2) * 0.8;

    std::ofstream f_train("features_train.csv");
    std::ofstream l_train("labels_train.csv");
    std::ofstream f_test("features_test.csv");
    std::ofstream l_test("labels_test.csv");

    for (int i = 1; i <= all_features.Y(); ++i) {
        if (i <= train_size) {
            f_train << all_features.D(1, i) << "," << all_features.D(2, i) << "\n";
            l_train << all_labels.D(1, i) << "\n";
        }
        else {
            f_test << all_features.D(1, i) << "," << all_features.D(2, i) << "\n";
            l_test << all_labels.D(1, i) << "\n";
        }
    }
    f_train.close(); l_train.close(); f_test.close(); l_test.close();

    ICG_printf(MLE, "Veri seti olusturuldu ve dosyalara yazildi:\n");
    ICG_printf(MLE, "- features_train.csv (%d ornek)\n", train_size);
    ICG_printf(MLE, "- labels_train.csv (%d ornek)\n", train_size);
    ICG_printf(MLE, "- features_test.csv (%d ornek)\n", (samples_per_class * 2) - train_size);
    ICG_printf(MLE, "- labels_test.csv (%d ornek)\n\n");
}

/**
 * @brief Bir matrisin her elemanýna sigmoid fonksiyonunu uygular.
 */
void Sigmoid(ICBYTES& z, ICBYTES& result) {
    long long rows = z.Y();
    long long cols = z.X();
    CreateMatrix(result, cols, rows, ICB_DOUBLE);

    for (long long r = 1; r <= rows; ++r) {
        for (long long c = 1; c <= cols; ++c) {
            double value = z.D(c, r);
            result.D(c, r) = 1.0 / (1.0 + exp(-value));
        }
    }
}

/**
 * @brief Lojistik regresyon için maliyet (cost) deðerini hesaplar.
 */
double ComputeCost(ICBYTES& X, ICBYTES& y, ICBYTES& theta) {
    long long N = X.Y();
    ICBYTES z, h;
    z.dot(X, theta);
    Sigmoid(z, h);
    double total_cost = 0.0;
    const double epsilon = 1e-9;
    for (long long i = 1; i <= N; ++i) {
        double h_i = h.D(1, i);
        double y_i = y.D(1, i);
        double cost_i = (y_i * log(h_i + epsilon)) + ((1.0 - y_i) * log(1.0 - h_i + epsilon));
        total_cost += cost_i;
    }
    return -total_cost / N;
}

// --- Buton Fonksiyonlarý (Callbacks) ---

/**
 * @brief "Veri Yükle" butonu, eðitim verilerini yükler.
 */
void LoadData()
{
    ICG_printf(MLE, "Egitim verileri yukleniyor...\n");

    // DÜZELTME: Veriyi önce GLOBAL X_train yerine YEREL bir deðiþkene oku.
    ICBYTES features_raw;
    if (!ReadCSVtoICBYTES("features_train.csv", features_raw)) {
        ICG_printf(MLE, "HATA: features_train.csv bulunamadi. Lutfen once 'YENI VERI URET' ile veri seti olusturun.\n");
        return;
    }

    if (!ReadCSVtoICBYTES("labels_train.csv", y_train)) {
        ICG_printf(MLE, "HATA: labels_train.csv bulunamadi.\n");
        return;
    }

    // Boyutlarý yerel ve güvenli olan features_raw'dan al
    long long N = features_raw.Y();
    long long M = features_raw.X();

    // Þimdi global X_train'i doðru ve son boyutuyla oluþtur
    CreateMatrix(X_train, M + 1, N, ICB_DOUBLE);

    // 1. sütuna bias (1.0) deðerlerini ata
    for (long long i = 1; i <= N; ++i) {
        X_train.D(1, i) = 1.0;
    }

    // Orijinal verileri, güvenli olan features_raw kopyasýndan global X_train'e aktar
    for (long long r = 1; r <= N; ++r) {
        for (long long c = 1; c <= M; ++c) {
            X_train.D(c + 1, r) = features_raw.D(c, r);
        }
    }

    Free(theta);

    ICG_printf(MLE, "Egitim verileri basariyla yuklendi.\n");
    ICG_printf(MLE, "Ornek Sayisi (N): %lld\n", N);
    ICG_printf(MLE, "Ozellik Sayisi (M): %lld\n", M);
    ICG_printf(MLE, "X_train boyutu (bias dahil): %lld x %lld\n\n", X_train.Y(), X_train.X());
}

/**
 * @brief Modeli eðitir.
 */
void TrainModel()
{
    if (X_train.Y() == 0) {
        ICG_printf(MLE, "HATA: Once verileri yuklemelisiniz!\n");
        return;
    }

    ICG_printf(MLE, "\n--- Model Egitimi Baslatildi ---\n");
    ICG_printf(MLE, "Ogrenme Orani (alpha): %f\n", learning_rate);
    ICG_printf(MLE, "Iterasyon Sayisi: %d\n", iterations);

    long long N = X_train.Y();
    long long M_plus_1 = X_train.X();

    CreateMatrix(theta, 1, M_plus_1, ICB_DOUBLE);
    theta = 0.0;

    ICBYTES z, h, error, X_train_T, gradient;
    transpose(X_train, X_train_T);

    for (int i = 0; i < iterations; ++i) {
        z.dot(X_train, theta);
        Sigmoid(z, h);

        long long error_rows = h.Y();
        long long error_cols = h.X();
        CreateMatrix(error, error_cols, error_rows, ICB_DOUBLE);
        for (long long r = 1; r <= error_rows; ++r) {
            for (long long c = 1; c <= error_cols; ++c) {
                error.D(c, r) = h.D(c, r) - y_train.D(c, r);
            }
        }

        gradient.dot(X_train_T, error);
        gradient *= (1.0 / N);

        gradient *= learning_rate;
        theta -= gradient;

        if (i % 100 == 0 || i == iterations - 1) {
            double cost = ComputeCost(X_train, y_train, theta);
            ICG_printf(MLE, "Iterasyon %d, Maliyet (Cost): %f\n", i, cost);
        }
    }

    ICG_printf(MLE, "--- Egitim Tamamlandi ---\n");
    ICG_printf(MLE, "Ogrenilen Theta Degerleri:\n");
    DisplayMatrix(MLE, theta);
    ICG_printf(MLE, "\n");
}

/**
 * @brief Eðitim verisi üzerinde tahmin yapar ve doðruluðu ölçer.
 */
void PredictOnTrainData()
{
    if (theta.Y() == 0) {
        ICG_printf(MLE, "HATA: Once modeli egitmelisiniz!\n");
        return;
    }
    ICG_printf(MLE, "\n--- Egitim Verisi Uzerinde Tahmin Baslatildi ---\n");
    ICBYTES z, h;
    z.dot(X_train, theta);
    Sigmoid(z, h);
    long long N = X_train.Y();
    ICBYTES predictions;
    CreateMatrix(predictions, 1, N, ICB_DOUBLE);
    for (long long i = 1; i <= N; ++i) {
        predictions.D(1, i) = (h.D(1, i) >= 0.5) ? 1.0 : 0.0;
    }
    double correct_count = 0;
    for (long long i = 1; i <= N; ++i) {
        if (predictions.D(1, i) == y_train.D(1, i)) {
            correct_count++;
        }
    }
    double accuracy = (correct_count / N) * 100.0;
    ICG_printf(MLE, "Egitim verisi uzerindeki dogruluk orani: %%%.2f\n", accuracy);
    ICG_printf(MLE, "(%lld ornekten %d tanesi dogru tahmin edildi.)\n\n", N, (int)correct_count);
}

/**
 * @brief Test verisi üzerinde tahmin yapar ve doðruluðu ölçer.
 */
void PredictOnTestData()
{
    if (theta.Y() == 0) {
        ICG_printf(MLE, "HATA: Once modeli egitmelisiniz!\n");
        return;
    }

    ICG_printf(MLE, "\n--- TEST VERISI UZERINDE TAHMIN BASLATILDI ---\n");

    ICBYTES X_test_raw, y_test;
    if (!ReadCSVtoICBYTES("features_test.csv", X_test_raw)) {
        ICG_printf(MLE, "HATA: features_test.csv bulunamadi.\n");
        return;
    }
    if (!ReadCSVtoICBYTES("labels_test.csv", y_test)) {
        ICG_printf(MLE, "HATA: labels_test.csv bulunamadi.\n");
        return;
    }

    long long N_test = X_test_raw.Y();
    long long M_test = X_test_raw.X();
    ICBYTES X_test;
    CreateMatrix(X_test, M_test + 1, N_test, ICB_DOUBLE);
    for (long long i = 1; i <= N_test; ++i) {
        X_test.D(1, i) = 1.0;
    }
    for (long long r = 1; r <= N_test; ++r) {
        for (long long c = 1; c <= M_test; ++c) {
            X_test.D(c + 1, r) = X_test_raw.D(c, r);
        }
    }

    ICBYTES z_test, h_test;
    z_test.dot(X_test, theta);
    Sigmoid(z_test, h_test);

    double correct_count = 0;
    for (long long i = 1; i <= N_test; ++i) {
        double prediction = (h_test.D(1, i) >= 0.5) ? 1.0 : 0.0;
        if (prediction == y_test.D(1, i)) {
            correct_count++;
        }
    }

    double accuracy = (correct_count / N_test) * 100.0;
    ICG_printf(MLE, "Test verisi uzerindeki dogruluk orani: %%%.2f\n", accuracy);
    ICG_printf(MLE, "(%lld test orneginden %d tanesi dogru tahmin edildi.)\n\n", N_test, (int)correct_count);
}

// --- Ana Arayüz Kurulum Fonksiyonlarý ---

void ICGUI_Create() {
    ICG_MWTitle("ICBYTES ile Lojistik Regresyon");
    ICG_MWSize(1100, 800); // Pencereyi biraz küçülttüm
}

void ICGUI_main()
{
    // Butonlarý ve iþlevlerini tanýmla
    ICG_Button(10, 10, 200, 30, "0. YENI VERI URET", GenerateAndSaveData);
    ICG_Button(10, 50, 200, 30, "1. Egitim Verilerini Yukle", LoadData);
    ICG_Button(10, 90, 200, 30, "2. Modeli Egit", TrainModel);
    ICG_Button(10, 130, 200, 30, "3. Tahmin Et (Egitim Verisi)", PredictOnTrainData);
    ICG_Button(10, 170, 200, 30, "4. Tahmin Et (TEST VERISI)", PredictOnTestData);

    // Metin kutusunu oluþtur
    MLE = ICG_MLEditSunken(220, 10, 800, 700, "", SCROLLBAR_V);

    ICG_printf(MLE, "Lojistik Regresyon Projesine Hos Geldiniz!\n\n");
    ICG_printf(MLE, "ADIM 0: Proje klasorunde egitim/test verileri yoksa 'YENI VERI URET' butonuna basarak olusturun.\n\n");
    ICG_printf(MLE, "Ardindan diger adimlari sirasiyla (1-2-3-4) takip edin.\n");
}