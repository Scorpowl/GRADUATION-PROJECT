// ICBYTES ile Lojistik Regresyon Projesi
// Final Sürümü v4.0 - Kullanıcı Tarafından Sağlanan Kodun Temizlenmiş Hali
// Yazan: Gemini (Google AI) & Proje Sahibi

#define NOMINMAX
#include "ic_media.h"
#include "icbytes.h"
#include "icb_gui.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// --- Global Değişkenler ---
int MLE;
ICBYTES X_train;
ICBYTES y_train;
ICBYTES theta;
// Görselleştirme ile ilgili global değişkenler kaldırıldı.
double learning_rate = 0.01;
int iterations = 1500;

// --- Yardımcı Fonksiyonlar ---

/**
 * Bir CSV dosyasını okuyup içeriğini bir ICBYTES matrisine aktarır.
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
 * Lojistik regresyon için yapay bir veri seti oluşturur, karıştırır ve
 * eğitim/test olarak ayırıp CSV dosyalarına kaydeder.
 */
void GenerateAndSaveData()
{
    ICG_ClearText(MLE);
    ICBYTES class0_features, class1_features;
    int samples_per_class = 100;
    double center0 = 3.5, center1 = 6.5, stdev = 2.5;

    RandomNormal(center0, stdev, class0_features, 2, samples_per_class);
    RandomNormal(center1, stdev, class1_features, 2, samples_per_class);

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

    std::vector<int> indices(all_features.Y());
    for (int i = 0; i < indices.size(); ++i) indices[i] = i + 1;
    std::mt19937 rng(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    ICBYTES shuffled_features, shuffled_labels;
    CreateMatrix(shuffled_features, 2, samples_per_class * 2, ICB_DOUBLE);
    CreateMatrix(shuffled_labels, 1, samples_per_class * 2, ICB_DOUBLE);
    for (int i = 0; i < indices.size(); ++i)
    {
        shuffled_features.D(1, i + 1) = all_features.D(1, indices[i]);
        shuffled_features.D(2, i + 1) = all_features.D(2, indices[i]);
        shuffled_labels.D(1, i + 1) = all_labels.D(1, indices[i]);
    }

    int train_size = (samples_per_class * 2) * 0.8;
    std::ofstream f_train("features_train.csv"), l_train("labels_train.csv"), f_test("features_test.csv"), l_test("labels_test.csv");

    for (int i = 1; i <= shuffled_features.Y(); ++i) {
        if (i <= train_size) {
            f_train << shuffled_features.D(1, i) << "," << shuffled_features.D(2, i) << "\n";
            l_train << shuffled_labels.D(1, i) << "\n";
        }
        else {
            f_test << shuffled_features.D(1, i) << "," << shuffled_features.D(2, i) << "\n";
            l_test << shuffled_labels.D(1, i) << "\n";
        }
    }
    f_train.close(); l_train.close(); f_test.close(); l_test.close();
    ICG_printf(MLE, "Zorlu veri seti olusturuldu ve 4 dosyaya ayrildi.\n- Egitim: %d ornek\n- Test: %d ornek\n", train_size, (samples_per_class * 2) - train_size);
}

/**
 * Bir matrisin her elemanına sigmoid fonksiyonunu uygular.
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
 * Lojistik regresyon için maliyet (cost) değerini hesaplar.
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

// --- Buton Fonksiyonları (Callbacks) ---

void LoadData()
{
    ICG_printf(MLE, "\nEgitim verileri yukleniyor...\n");

    ICBYTES features_raw;
    if (!ReadCSVtoICBYTES("features_train.csv", features_raw)) {
        ICG_printf(MLE, "HATA: features_train.csv bulunamadi. Lutfen once '0. YENI VERI URET' ile veri seti olusturun.\n");
        return;
    }

    if (!ReadCSVtoICBYTES("labels_train.csv", y_train)) {
        ICG_printf(MLE, "HATA: labels_train.csv bulunamadi.\n");
        return;
    }

    long long N = features_raw.Y();
    long long M = features_raw.X();

    CreateMatrix(X_train, M + 1, N, ICB_DOUBLE);

    for (long long i = 1; i <= N; ++i) {
        X_train.D(1, i) = 1.0;
    }

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

        if (i % 100 == 99 || i == 0) {
            ICG_printf(MLE, "Iterasyon %d, Maliyet (Cost): %f\n", i + 1, ComputeCost(X_train, y_train, theta));
        }
    }

    ICG_printf(MLE, "--- Egitim Tamamlandi ---\n");
}

void PredictOnTrainData()
{
    if (theta.Y() == 0) {
        ICG_printf(MLE, "\nHATA: Once 2. adimda modeli egitmelisiniz!\n");
        return;
    }
    ICG_printf(MLE, "\n--- Egitim Verisi Uzerinde Tahmin ---\n");
    ICBYTES z, h;
    z.dot(X_train, theta);
    Sigmoid(z, h);
    long long N = X_train.Y();
    double correct_count = 0;
    for (long long i = 1; i <= N; ++i) {
        if (((h.D(1, i) >= 0.5) ? 1.0 : 0.0) == y_train.D(1, i)) {
            correct_count++;
        }
    }
    ICG_printf(MLE, "Dogruluk: %%%.2f (%lld ornekten %d dogru)\n", (correct_count / N) * 100.0, N, (int)correct_count);
}

void PredictOnTestData()
{
    if (theta.Y() == 0) {
        ICG_printf(MLE, "\nHATA: Once 2. adimda modeli egitmelisiniz!\n");
        return;
    }
    ICG_printf(MLE, "\n--- Test Verisi Uzerinde Tahmin ---\n");
    ICBYTES X_test_raw, y_test;
    if (!ReadCSVtoICBYTES("features_test.csv", X_test_raw) || !ReadCSVtoICBYTES("labels_test.csv", y_test)) {
        ICG_printf(MLE, "HATA: Test dosyalari bulunamadi.\n");
        return;
    }
    long long N_test = X_test_raw.Y();
    ICBYTES X_test_with_bias;
    CreateMatrix(X_test_with_bias, X_test_raw.X() + 1, N_test, ICB_DOUBLE);
    for (long long i = 1; i <= N_test; ++i) X_test_with_bias.D(1, i) = 1.0;
    for (long long r = 1; r <= N_test; ++r) { for (long long c = 1; c <= X_test_raw.X(); ++c) X_test_with_bias.D(c + 1, r) = X_test_raw.D(c, r); }
    ICBYTES z_test, h_test;
    z_test.dot(X_test_with_bias, theta);
    Sigmoid(z_test, h_test);
    double correct_count = 0;
    for (long long i = 1; i <= N_test; ++i) {
        if (((h_test.D(1, i) >= 0.5) ? 1.0 : 0.0) == y_test.D(1, i)) {
            correct_count++;
        }
    }
    ICG_printf(MLE, "Dogruluk: %%%.2f (%lld ornekten %d dogru)\n", (correct_count / N_test) * 100.0, N_test, (int)correct_count);
}

void ShowConfusionMatrixAndMetrics()
{
    if (theta.Y() == 0) {
        ICG_printf(MLE, "\nHATA: Once 2. adimda modeli egitmelisiniz!\n");
        return;
    }
    ICG_printf(MLE, "\n--- DETAYLI TEST PERFORMANS ANALIZI ---\n");
    ICBYTES X_test_raw, y_test;
    if (!ReadCSVtoICBYTES("features_test.csv", X_test_raw) || !ReadCSVtoICBYTES("labels_test.csv", y_test)) {
        ICG_printf(MLE, "HATA: Test dosyalari bulunamadi.\n");
        return;
    }
    long long N_test = X_test_raw.Y();
    ICBYTES X_test_with_bias;
    CreateMatrix(X_test_with_bias, X_test_raw.X() + 1, N_test, ICB_DOUBLE);
    for (long long i = 1; i <= N_test; ++i) X_test_with_bias.D(1, i) = 1.0;
    for (long long r = 1; r <= N_test; ++r) { for (long long c = 1; c <= X_test_raw.X(); ++c) X_test_with_bias.D(c + 1, r) = X_test_raw.D(c, r); }
    ICBYTES z_test, h_test;
    z_test.dot(X_test_with_bias, theta);
    Sigmoid(z_test, h_test);
    double TP = 0, TN = 0, FP = 0, FN = 0;
    for (long long i = 1; i <= N_test; ++i) {
        double prediction = (h_test.D(1, i) >= 0.5) ? 1.0 : 0.0;
        double actual = y_test.D(1, i);
        if (prediction == 1.0 && actual == 1.0) TP++;
        else if (prediction == 0.0 && actual == 0.0) TN++;
        else if (prediction == 1.0 && actual == 0.0) FP++;
        else if (prediction == 0.0 && actual == 1.0) FN++;
    }
    double accuracy = (TP + TN) / (TP + TN + FP + FN) * 100.0;
    double precision = (TP + FP == 0) ? 0 : TP / (TP + FP);
    double recall = (TP + FN == 0) ? 0 : TP / (TP + FN);
    double f1_score = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);

    ICG_printf(MLE, "Karısıklık Matrisi (Confusion Matrix):\n");
    ICG_printf(MLE, "       Tahmin:0  Tahmin:1\n");
    ICG_printf(MLE, "Gercek:0 [%.0f]      [%.0f]\n", TN, FP);
    ICG_printf(MLE, "Gercek:1 [%.0f]      [%.0f]\n\n", FN, TP);
    ICG_printf(MLE, "Dogruluk (Accuracy): %%%.2f\n", accuracy);
    ICG_printf(MLE, "Kesinlik (Precision): %.2f\n", precision);
    ICG_printf(MLE, "Duyarlilik (Recall/Sensitivity): %.2f\n", recall);
    ICG_printf(MLE, "F1-Skoru: %.2f\n\n", f1_score);
}

// --- Ana Arayüz Kurulum Fonksiyonları ---

void ICGUI_Create() {
    ICG_MWTitle("ICBYTES ile Lojistik Regresyon");
    ICG_MWSize(850, 450);
}

void ICGUI_main()
{
    int button_x = 10;
    int button_width = 200;

    ICG_Button(button_x, 10, button_width, 30, "0. Sentetik Veri Uret", GenerateAndSaveData);
    ICG_Button(button_x, 50, button_width, 30, "1. Egitim Verilerini Yukle", LoadData);
    ICG_Button(button_x, 90, button_width, 30, "2. Modeli Egit", TrainModel);
    ICG_Button(button_x, 130, button_width, 30, "3. Tahmin (Egitim Verisi)", PredictOnTrainData);
    ICG_Button(button_x, 170, button_width, 30, "4. Tahmin (Test Verisi)", PredictOnTestData);
    ICG_Button(button_x, 210, button_width, 30, "5. Detayli Rapor (Test)", ShowConfusionMatrixAndMetrics);

    // Tek ve büyük bir metin alanı
    MLE = ICG_MLEditSunken(button_x + button_width + 10, 10, 620, 420, "", SCROLLBAR_HV);

    ICG_printf(MLE, "Lojistik Regresyon Projesine Hos Geldiniz!\n\n");
    ICG_printf(MLE, "ADIM 0: Proje klasorunde egitim/test verileri yoksa 'YENI VERI URET' butonuna basarak olusturun.\n\n");
    ICG_printf(MLE, "Ardindan diger adimlari sirasiyla (1-2-3-4) takip edin.\n");
}