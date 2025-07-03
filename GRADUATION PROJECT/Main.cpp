// ICBYTES ile Lojistik Regresyon Projesi
// Adým 1: Arayüz Kurulumu ve Veri Yükleme
// Yazan: Gemini (Google AI) & Proje Sahibi

#include "ic_media.h"
#include "icbytes.h"

// ICBYTES ile Lojistik Regresyon Projesi
// Adým 2.1: Heap Hatasý Düzeltilmiþ ve Verimli Kod
// Yazan: Gemini (Google AI) & Proje Sahibi

#include "icb_gui.h"   
#include <fstream>      
#include <sstream>      
#include <string>       
#include <vector>       
#include <cmath>        

// ... (Global deðiþkenler ve ReadCSVtoICBYTES, Sigmoid, ComputeCost fonksiyonlarý ayný kalacak) ...
// --- Global Deðiþkenler ---
int MLE;
ICBYTES X_train;
ICBYTES y_train;
ICBYTES theta;
double learning_rate = 0.01;
int iterations = 1000;

// --- Yardýmcý Fonksiyonlar ---
bool ReadCSVtoICBYTES(const std::string& filename, ICBYTES& matrix) { /* ... Önceki kodla ayný ... */
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
void Sigmoid(ICBYTES& z, ICBYTES& result) { /* ... Önceki kodla ayný ... */
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
double ComputeCost(ICBYTES& X, ICBYTES& y, ICBYTES& theta) { /* ... Önceki kodla ayný ... */
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
void LoadData() { /* ... Önceki kodla ayný ... */
    ICG_printf(MLE, "Veriler yukleniyor...\n");
    ICBYTES features_raw;
    if (!ReadCSVtoICBYTES("features.csv", features_raw)) {
        return;
    }
    if (!ReadCSVtoICBYTES("labels.csv", y_train)) {
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
    ICG_printf(MLE, "Veriler basariyla yuklendi.\n");
    ICG_printf(MLE, "Ornek Sayisi (N): %lld\n", N);
    ICG_printf(MLE, "Ozellik Sayisi (M): %lld\n", M);
    ICG_printf(MLE, "X_train boyutu (bias dahil): %lld x %lld\n\n", X_train.Y(), X_train.X());
}

/*******************************************************************************************/
//                           DÜZELTÝLMÝÞ EÐÝTÝM FONKSÝYONU
/*******************************************************************************************/
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

    // Gradyan Ýniþi Döngüsü
    for (int i = 0; i < iterations; ++i) {
        // 1. Hipotez
        if (!z.dot(X_train, theta)) { // <-- KONTROL 1
            ICG_printf(MLE, "HATA: z.dot(X_train, theta) isleminde matris boyut hatasi!\n");
            return; // Eðitimi durdur
        }
        Sigmoid(z, h);

        // 2. Hata
        error = h;
        error -= y_train;

        // 3. Gradyan
        // ** EN ÖNEMLÝ KONTROL BURADA **
        if (!gradient.dot(X_train_T, error)) { // <-- KONTROL 2
            ICG_printf(MLE, "HATA: gradient.dot(X_train_T, error) isleminde matris boyut hatasi!\n");
            ICG_printf(MLE, "X_train_T Boyut (sutun, satir): %lld x %lld\n", X_train_T.X(), X_train_T.Y());
            ICG_printf(MLE, "error Boyut (sutun, satir): %lld x %lld\n", error.X(), error.Y());
            return; // Eðitimi durdur
        }

        gradient *= (1.0 / N);

        // 4. Theta Güncelleme
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

void Predict() { /* ... Önceki kodla ayný, þimdilik boþ ... */
    ICG_printf(MLE, "Tahmin Fonksiyonu Henuz Yazilmadi.\n");
    ICG_printf(MLE, "--- Adim 3: Dogruluk Hesaplama ---\n");
}

void TestSumFunction() { /* ... Önceki kodla ayný ... */
    ICG_printf(MLE, "Sum() fonksiyonu test ediliyor...\n");
    ICBYTES A = { {1.5, 2.5, 3.0}, {4.0, 5.0, 6.5} };
    DisplayMatrix(MLE, A);
    double total = Sum(A);
    ICG_printf(MLE, "Matris elemanlarinin toplami: %f\n\n", total);
}


// --- Ana Arayüz Kurulum Fonksiyonlarý ---
void ICGUI_Create() { /* ... Önceki kodla ayný ... */
    ICG_MWTitle("ICBYTES ile Lojistik Regresyon");
    ICG_MWSize(800, 600);
}
void ICGUI_main() { /* ... Önceki kodla ayný ... */
    ICG_Button(10, 10, 150, 30, "Veri Yukle", LoadData);
    ICG_Button(10, 50, 150, 30, "Modeli Egit", TrainModel);
    ICG_Button(10, 90, 150, 30, "Tahmin Et (Egitim Verisi)", Predict);
    ICG_Button(10, 130, 150, 30, "Sum() Fonksiyonunu Test Et", TestSumFunction);
    MLE = ICG_MLEditSunken(170, 10, 610, 570, "", SCROLLBAR_V);
    ICG_printf(MLE, "Hos geldiniz! Lutfen 'features.csv' ve 'labels.csv' dosyalarini proje klasorune yerlestirdikten sonra 'Veri Yukle' butonuna tiklayin.\n\n");
    ICG_printf(MLE, "Ornek features.csv (N ornek, M ozellik):\n");
    ICG_printf(MLE, "ozellik1_1,ozellik1_2,....,ozellik1_M\n");
    ICG_printf(MLE, "ozellik2_1,ozellik2_2,....,ozellik2_M\n...\n\n");
    ICG_printf(MLE, "Ornek labels.csv (N ornek, tek sutunlu etiket):\n");
    ICG_printf(MLE, "etiket_1\netiket_2\n...\n");
}