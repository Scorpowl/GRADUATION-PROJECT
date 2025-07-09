// ICBYTES ile Lojistik Regresyon Projesi
// Adým 1: Arayüz Kurulumu ve Veri Yükleme
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
// --- Global Deðiþkenler ---
int MLE;
int RICH_REPORT;
ICBYTES X_train;
ICBYTES y_train;
ICBYTES theta;
ICBYTES canvas_gorsel;
int FRM_VISUAL;
double learning_rate = 0.01;
int iterations = 1500; // Veri seti büyüdüðü için iterasyon sayýsýný azaltýp artýrabiliriz.

// --- Yardýmcý Fonksiyonlar ---

/**
    Bir CSV dosyasýný okuyup içeriðini bir ICBYTES matrisine aktarýr.
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
 * Lojistik regresyon için yapay bir veri seti oluþturur, karýþtýrýr ve
 * eðitim/test olarak ayýrýp CSV dosyalarýna kaydeder.
 */
void GenerateAndSaveData()
{
    ICBYTES class0_features, class1_features;
    int samples_per_class = 100;

    // DEÐÝÞÝKLÝK: Veri bulutlarýný birbirine yaklaþtýrýp daðýlýmlarýný artýrýyoruz.
    // Böylece daha zor, iç içe geçmiþ bir problem yaratýyoruz.
    double center0 = 3.5;
    double center1 = 6.5;
    double stdev = 2.5; // Standart sapmayý 1.5'ten 2.5'e çýkardýk.

    // Sýnýf 0: Ortalamasý (3.5, 3.5) olan bir veri bulutu oluþtur
    RandomNormal(center0, stdev, class0_features, 2, samples_per_class);

    // Sýnýf 1: Ortalamasý (6.5, 6.5) olan bir veri bulutu oluþtur
    RandomNormal(center1, stdev, class1_features, 2, samples_per_class);

    // Fonksiyonun geri kalaný (birleþtirme, karýþtýrma, dosyalara yazma) ayný...
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

    ICG_printf(MLE, "DAHA ZOR bir veri seti olusturuldu ve dosyalara yazildi.\n");
}

/**
 * Bir matrisin her elemanýna sigmoid fonksiyonunu uygular.
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
 * Lojistik regresyon için maliyet (cost) deðerini hesaplar.
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
 *  "Veri Yükle" butonu, eðitim verilerini yükler.
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
 *  Modeli eðitir.
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

void ShowConfusionMatrixAndMetrics()
{
    if (theta.Y() == 0) {
        ICG_printf(MLE, "HATA: Once modeli egitmelisiniz!\n");
        return;
    }

    ICG_printf(MLE, "\n--- DETAYLI TEST PERFORMANS ANALIZI ---\n");

    ICBYTES X_test_raw, y_test;
    if (!ReadCSVtoICBYTES("features_test.csv", X_test_raw) || !ReadCSVtoICBYTES("labels_test.csv", y_test)) {
        ICG_printf(MLE, "HATA: Test .csv dosyalari okunamadi.\n");
        return;
    }

    long long N_test = X_test_raw.Y();
    long long M_test = X_test_raw.X();
    ICBYTES X_test;
    CreateMatrix(X_test, M_test + 1, N_test, ICB_DOUBLE);
    for (long long i = 1; i <= N_test; ++i) X_test.D(1, i) = 1.0;
    for (long long r = 1; r <= N_test; ++r) {
        for (long long c = 1; c <= M_test; ++c) {
            X_test.D(c + 1, r) = X_test_raw.D(c, r);
        }
    }

    ICBYTES z_test, h_test;
    z_test.dot(X_test, theta);
    Sigmoid(z_test, h_test);

    // Karýþýklýk matrisi için deðiþkenler
    double TP = 0, TN = 0, FP = 0, FN = 0;

    for (long long i = 1; i <= N_test; ++i) {
        double prediction = (h_test.D(1, i) >= 0.5) ? 1.0 : 0.0;
        double actual = y_test.D(1, i);

        if (prediction == 1.0 && actual == 1.0) {
            TP++; // Doðru Pozitif
        }
        else if (prediction == 0.0 && actual == 0.0) {
            TN++; // Doðru Negatif
        }
        else if (prediction == 1.0 && actual == 0.0) {
            FP++; // Yanlýþ Pozitif
        }
        else if (prediction == 0.0 && actual == 1.0) {
            FN++; // Yanlýþ Negatif
        }
    }

    // Karýþýklýk matrisini ICBYTES ile oluþtur ve ekrana bas
    ICBYTES confusion_matrix;
    CreateMatrix(confusion_matrix, 2, 2, ICB_DOUBLE);
    confusion_matrix.D(1, 1) = TN; confusion_matrix.D(2, 1) = FP;
    confusion_matrix.D(1, 2) = FN; confusion_matrix.D(2, 2) = TP;

    ICG_printf(MLE, "Karýsýklýk Matrisi (Confusion Matrix):\n");
    ICG_printf(MLE, "       Tahmin:0  Tahmin:1\n");
    ICG_printf(MLE, "Gercek:0 [%.0f]      [%.0f]\n", TN, FP);
    ICG_printf(MLE, "Gercek:1 [%.0f]      [%.0f]\n\n", FN, TP);

    // Diðer metrikleri hesapla ve yazdýr
    double accuracy = (TP + TN) / (TP + TN + FP + FN) * 100.0;
    double precision = (TP + FP == 0) ? 0 : TP / (TP + FP);
    double recall = (TP + FN == 0) ? 0 : TP / (TP + FN);
    double f1_score = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);

    ICG_printf(MLE, "Dogruluk (Accuracy): %%%.2f\n", accuracy);
    ICG_printf(MLE, "Kesinlik (Precision): %.2f\n", precision);
    ICG_printf(MLE, "Duyarlilik (Recall/Sensitivity): %.2f\n", recall);
    ICG_printf(MLE, "F1-Skoru: %.2f\n\n", f1_score);
}

/**
 *  Eðitim verisi üzerinde tahmin yapar ve doðruluðu ölçer.
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

void CreateFinalReport()
{
    if (theta.Y() == 0) {
        ICG_printf(MLE, "HATA: Once modeli egitmelisiniz!\n");
        return;
    }

    // Rapor alanýný temizle
    ICG_ClearText(RICH_REPORT);

    ICBYTES X_test_raw, y_test;
    if (!ReadCSVtoICBYTES("features_test.csv", X_test_raw) || !ReadCSVtoICBYTES("labels_test.csv", y_test)) {
        ICG_printf(MLE, "HATA: Test .csv dosyalari okunamadi.\n");
        return;
    }

    // Tahminleri yap
    long long N_test = X_test_raw.Y();
    ICBYTES X_test_with_bias;
    CreateMatrix(X_test_with_bias, X_test_raw.X() + 1, N_test, ICB_DOUBLE);
    for (long long i = 1; i <= N_test; ++i) X_test_with_bias.D(1, i) = 1.0;
    for (long long r = 1; r <= N_test; ++r) { for (long long c = 1; c <= X_test_raw.X(); ++c) { X_test_with_bias.D(c + 1, r) = X_test_raw.D(c, r); } }
    ICBYTES z_test, h_test;
    z_test.dot(X_test_with_bias, theta);
    Sigmoid(z_test, h_test);

    // Karýþýklýk matrisi deðerlerini hesapla
    double TP = 0, TN = 0, FP = 0, FN = 0;
    for (long long i = 1; i <= N_test; ++i) {
        double prediction = (h_test.D(1, i) >= 0.5) ? 1.0 : 0.0;
        double actual = y_test.D(1, i);
        if (prediction == 1.0 && actual == 1.0) TP++;
        else if (prediction == 0.0 && actual == 0.0) TN++;
        else if (prediction == 1.0 && actual == 0.0) FP++;
        else if (prediction == 0.0 && actual == 1.0) FN++;
    }

    // Metrikleri hesapla
    double accuracy = (TP + TN) / (TP + TN + FP + FN) * 100.0;
    double precision = (TP + FP == 0) ? 0 : TP / (TP + FP);
    double recall = (TP + FN == 0) ? 0 : TP / (TP + FN);
    double f1_score = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);

    // --- Raporu Zengin Metin Formatýnda Yazdýr ---
    ICG_SetprintfFont("Verdana", 22);
    ICG_SetprintfColor(0x00D090); // Açýk yeþil renk
    ICG_SetprintfStyle(700, false, true); // Kalýn ve altý çizgili
    ICG_printf(RICH_REPORT, "\tTEST SONUÇ RAPORU\n\n");

    ICG_SetprintfFont("Consolas", 16);
    ICG_SetprintfColor(0xFFFFFF); // Beyaz
    ICG_SetprintfStyle(700, false, false); // Kalýn
    ICG_printf(RICH_REPORT, "Karýsýklýk Matrisi (Confusion Matrix)\n");

    ICG_SetprintfFont("Consolas", 14);
    ICG_SetprintfStyle(400, false, false); // Normal
    ICG_printf(RICH_REPORT, "--------------------------------------\n");
    ICG_printf(RICH_REPORT, "\t\tTahmin: KALDI\tTahmin: GECTI\n");
    ICG_printf(RICH_REPORT, "Gercek: KALDI\t   [ %.0f ]\t\t   [ %.0f ]\n", TN, FP);
    ICG_printf(RICH_REPORT, "Gercek: GECTI\t   [ %.0f ]\t\t   [ %.0f ]\n", FN, TP);
    ICG_printf(RICH_REPORT, "--------------------------------------\n\n");

    ICG_SetprintfFont("Verdana", 16);
    ICG_SetprintfStyle(700, false, false); // Kalýn
    ICG_printf(RICH_REPORT, "Performans Metrikleri\n");

    ICG_SetprintfFont("Verdana", 14);
    ICG_SetprintfStyle(400, false, false); // Normal
    ICG_printf(RICH_REPORT, "---------------------------------\n");
    ICG_printf(RICH_REPORT, "Dogruluk (Accuracy)    : %%%.2f\n", accuracy);
    ICG_printf(RICH_REPORT, "Kesinlik (Precision)   : %.2f\n", precision);
    ICG_printf(RICH_REPORT, "Duyarlilik (Recall)    : %.2f\n", recall);
    ICG_printf(RICH_REPORT, "F1-Skoru               : %.2f\n", f1_score);
    ICG_printf(RICH_REPORT, "---------------------------------\n");
}

/**
 * Test verisi üzerinde tahmin yapar ve doðruluðu ölçer.
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
    ICG_MWSize(1100, 800); 
}

void ICGUI_main()
{
    int button_x = 10;
    int button_width = 200;
    ICG_Button(button_x, 10, button_width, 30, "1. Verileri Yukle & Ayir", GenerateAndSaveData);
    ICG_Button(button_x, 50, button_width, 30, "2. Modeli Egit", TrainModel);

    // YENÝ DÜZEN: Butonlarýn adýný ve iþlevini birleþtirdim
    ICG_Button(button_x, 90, button_width, 30, "3. SONUCLARI RAPORLA", CreateFinalReport);

    int log_x = button_x + button_width + 10;
    // Log ve iþlem adýmlarý için metin kutusu
    MLE = ICG_MLEditSunken(log_x, 10, 600, 200, "", SCROLLBAR_V);

    // Nihai raporun gösterileceði Zengin Metin Kutusu
    RICH_REPORT = ICG_RichEditSunken(log_x, 220, 600, 210, "", SCROLLBAR_V);
    ICG_SetRichColor(RICH_REPORT, 0x1a1a1a); // Koyu arkaplan

    ICG_printf(MLE, "Lojistik Regresyon Projesine Hos Geldiniz!\n\n");
    ICG_printf(MLE, "ADIM 0: Proje klasorunde egitim/test verileri yoksa 'YENI VERI URET' butonuna basarak olusturun.\n\n");
    ICG_printf(MLE, "Ardindan diger adimlari sirasiyla (1-2-3-4) takip edin.\n");
}