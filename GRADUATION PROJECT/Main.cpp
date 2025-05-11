#include "icb_gui.h"   // ICGUI_Create, ICG_Button, ICG_MLEditSunken vb.
#include "icbytes.h"  // ICBYTES sýnýfý, CreateMatrix, Sum, transpose vb.
#include "ic_media.h" // Bu projede doðrudan kullanýlmýyor gibi ama kalabilir

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>   // std::exp, std::log için
#include <stdexcept> // std::stod hatalarý için

// --- Global GUI Deðiþkenleri ---
HWND hMLE_HWND_global = 0; // Çýktý için multiline editörün HWND'si

// --- Lojistik Regresyon Modeli ve Veri Deðiþkenleri ---
class LogisticRegressionModel; // Önden bildirim
LogisticRegressionModel* model_global = nullptr; // Dinamik olarak oluþturulacak
ICBYTES X_data_global, y_data_global;

// --- Yardýmcý Fonksiyon: CSV'den ICBYTES matrisine veri yükleme ---
bool load_csv_to_icbytes(const std::string& filename, ICBYTES& matrix, long long& out_rows, long long& out_cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        if (hMLE_HWND_global) ICG_printf("Hata: %s dosyasi acilamadi!\n", filename.c_str());
        return false;
    }

    std::vector<std::vector<double>> data_buffer;
    std::string line;
    out_cols = 0;

    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            out_cols++;
        }
        file.clear();
        file.seekg(0, std::ios::beg);
    }

    if (out_cols == 0) {
        if (hMLE_HWND_global) ICG_printf("Hata: %s dosyasinda sutun bulunamadi.\n", filename.c_str());
        file.close();
        return false;
    }

    out_rows = 0;
    while (std::getline(file, line)) {
        std::vector<double> row_vector;
        std::stringstream ss(line);
        std::string cell;
        long long current_cols_in_row = 0;
        while (std::getline(ss, cell, ',')) {
            try {
                row_vector.push_back(std::stod(cell));
                current_cols_in_row++;
            }
            catch (const std::invalid_argument&) {
                if (hMLE_HWND_global) ICG_printf("Hata: CSV'de gecersiz sayi '%s'. Satir atlandi.\n", cell.c_str());
                row_vector.clear();
                break;
            }
            catch (const std::out_of_range&) {
                if (hMLE_HWND_global) ICG_printf("Hata: CSV'de sayi '%s' aralik disinda. Satir atlandi.\n", cell.c_str());
                row_vector.clear();
                break;
            }
        }
        if (!row_vector.empty() && current_cols_in_row == out_cols) {
            data_buffer.push_back(row_vector);
            out_rows++;
        }
        else if (current_cols_in_row > 0) {
            if (hMLE_HWND_global) ICG_printf("Uyari: %s dosyasinda %lld sutunlu satir atlaniyor (beklenen %lld).\n", filename.c_str(), current_cols_in_row, out_cols);
        }
    }
    file.close();

    if (out_rows == 0) {
        if (hMLE_HWND_global) ICG_printf("Hata: %s dosyasinda gecerli veri satiri bulunamadi.\n", filename.c_str());
        return false;
    }

    if (CreateMatrix(matrix, out_rows, out_cols, ICB_DOUBLE) != 0) {
        if (hMLE_HWND_global) ICG_printf("Hata: CreateMatrix basarisiz oldu.\n");
        return false;
    }

    for (long long i = 0; i < out_rows; ++i) {
        for (long long j = 0; j < out_cols; ++j) {
            matrix.D(i + 1, j + 1) = data_buffer[i][j];
        }
    }
    return true;
}

// --- Yardýmcý Fonksiyonlar: Eleman bazlý iþlemler ---
ICBYTES elementwise_operation(ICBYTES& mat, double (*op_func)(double)) {
    ICBYTES result;
    CreateMatrix(result, mat.Y(), mat.X(), ICB_DOUBLE);
    for (long long i = 1; i <= mat.Y(); ++i) {
        for (long long j = 1; j <= mat.X(); ++j) {
            result.D(i, j) = op_func(mat.D(i, j));
        }
    }
    return result;
}

double safe_log(double val) {
    if (val <= 1e-9) return std::log(1e-9); // Çok küçük bir pozitif sayýya logaritma uygula
    if (val >= (1.0 - 1e-9)) return std::log(1.0 - 1e-9); // 1'e çok yakýnsa
    return std::log(val);
}

ICBYTES elementwise_exp(ICBYTES& mat) { return elementwise_operation(mat, std::exp); }
ICBYTES elementwise_log(ICBYTES& mat) { return elementwise_operation(mat, safe_log); }


// --- Sigmoid Fonksiyonu ---
ICBYTES sigmoid(ICBYTES& z_ref) {
    ICBYTES one_mat;
    CreateMatrix(one_mat, z_ref.Y(), z_ref.X(), ICB_DOUBLE);
    one_mat = 1.0;

    ICBYTES temp_z = z_ref;
    temp_z *= -1.0;
    ICBYTES exp_neg_z = elementwise_exp(temp_z);

    ICBYTES denominator = one_mat;
    denominator += exp_neg_z;

    ICBYTES result;
    CreateMatrix(result, z_ref.Y(), z_ref.X(), ICB_DOUBLE);
    for (long long i = 1; i <= z_ref.Y(); ++i) {
        for (long long j = 1; j <= z_ref.X(); ++j) {
            if (denominator.D(i, j) == 0.0) result.D(i, j) = 0.0;
            else result.D(i, j) = 1.0 / denominator.D(i, j);
        }
    }
    return result;
}

// --- Lojistik Regresyon Modeli Sýnýfý ---
class LogisticRegressionModel {
public:
    ICBYTES weights;
    double bias;

    LogisticRegressionModel() : bias(0.0) {}

    void initialize_weights(long long num_features) {
        CreateMatrix(weights, num_features, 1, ICB_DOUBLE);
        weights = 0.0;
    }

    double compute_cost(ICBYTES& y_true, ICBYTES& h_pred) {
        long long m = y_true.Y();
        if (m == 0) return 0.0;

        ICBYTES temp_h_pred = h_pred; // Log için kopya al
        ICBYTES log_h = elementwise_log(temp_h_pred);

        ICBYTES one_mat; CreateMatrix(one_mat, m, 1, ICB_DOUBLE); one_mat = 1.0;

        ICBYTES one_minus_h = one_mat;
        one_minus_h -= h_pred;
        ICBYTES temp_one_minus_h = one_minus_h; // Log için kopya al
        ICBYTES log_one_minus_h = elementwise_log(temp_one_minus_h);

        ICBYTES one_minus_y = one_mat;
        one_minus_y -= y_true;

        double total_cost = 0;
        for (long long i = 1; i <= m; ++i) {
            total_cost += (y_true.D(i, 1) * log_h.D(i, 1)) + (one_minus_y.D(i, 1) * log_one_minus_h.D(i, 1));
        }
        return (-1.0 / static_cast<double>(m)) * total_cost;
    }

    void train(ICBYTES& X_train, ICBYTES& y_train, double learning_rate, int iterations) {
        long long m = X_train.Y();
        long long n = X_train.X();

        if (weights.Y() == 0 || weights.X() == 0) {
            initialize_weights(n);
        }

        ICBYTES X_T;
        transpose(X_train, X_T);

        for (int iter = 0; iter < iterations; ++iter) {
            ICBYTES z;
            z.dot(X_train, weights);
            z += bias;

            ICBYTES h = sigmoid(z);

            ICBYTES error = h;
            error -= y_train;

            ICBYTES dw;
            dw.dot(X_T, error);
            dw *= (1.0 / static_cast<double>(m));

            double db = Sum(error) / static_cast<double>(m);

            ICBYTES temp_dw_update = dw;
            temp_dw_update *= learning_rate;
            weights -= temp_dw_update;

            bias -= learning_rate * db;

            if (iter % 100 == 0 && hMLE_HWND_global) {
                double current_cost = compute_cost(y_train, h);
                ICG_printf("Iterasyon %d, Maliyet: %f, Bias: %f\n", iter, current_cost, bias);
            }
        }
    }

    ICBYTES predict_proba(ICBYTES& X_test) {
        ICBYTES z;
        z.dot(X_test, weights);
        z += bias;
        return sigmoid(z);
    }

    ICBYTES predict(ICBYTES& X_test, double threshold = 0.5) {
        ICBYTES probabilities = predict_proba(X_test);
        ICBYTES predictions;
        CreateMatrix(predictions, probabilities.Y(), 1, ICB_INT);
        for (long long i = 1; i <= probabilities.Y(); ++i) {
            predictions.I(i, 1) = (probabilities.D(i, 1) >= threshold) ? 1 : 0;
        }
        return predictions;
    }
};

// --- ICGUI Buton Fonksiyonlarý ---
void LoadData_Clicked() {
    long long rows_x, cols_x, rows_y, cols_y;
    bool success_x = load_csv_to_icbytes("features.csv", X_data_global, rows_x, cols_x);
    bool success_y = load_csv_to_icbytes("labels.csv", y_data_global, rows_y, cols_y);

    if (success_x && success_y) {
        if (rows_x != rows_y || cols_y != 1) {
            if (hMLE_HWND_global) ICG_printf("Hata: Ozellik ve etiket boyutlari uyusmuyor! (Etiketler %lldx1 olmali)\n", rows_x);
            return;
        }
        if (hMLE_HWND_global) {
            ICG_printf("Veri yuklendi:\n");
            ICG_printf("Ozellikler (X): %lld ornek, %lld ozellik\n", X_data_global.Y(), X_data_global.X());
            ICG_printf("Etiketler (y): %lld ornek\n", y_data_global.Y());
        }
        if (model_global) delete model_global;
        model_global = new LogisticRegressionModel();
    }
    else {
        if (hMLE_HWND_global) ICG_printf("Veri yukleme basarisiz!\n");
    }
    if (model_global && hMLE_HWND_global) { // model_global ve pencere oluþturulduysa
        ICG_printf("--- X_data_global (ilk 3 satir) ---\n");
        for (long long r = 1; r <= X_data_global.Y() && r <= 3; ++r) {
            for (long long c = 1; c <= X_data_global.X(); ++c) {
                ICG_printf("%f ", X_data_global.D(r, c));
            }
            ICG_printf("\n");
        }
        ICG_printf("Boyutlar: %lld x %lld\n", X_data_global.Y(), X_data_global.X());

        ICG_printf("--- y_data_global (ilk 3 satir) ---\n");
        for (long long r = 1; r <= y_data_global.Y() && r <= 3; ++r) {
            ICG_printf("%f\n", y_data_global.D(r, 1)); // y_data_global tek sütunlu varsayýlýyor
        }
        ICG_printf("Boyutlar: %lld x %lld\n", y_data_global.Y(), y_data_global.X());
    }
}

void TrainModel_Clicked() {
    // TrainModel_Clicked() fonksiyonunun baþýnda
    if (model_global && hMLE_HWND_global) {
        ICG_printf("--- Baslangic Agirliklari (weights) ---\n");
        if (model_global->weights.Y() > 0 && model_global->weights.X() > 0) { // Matris boþ deðilse
            DisplayMatrix(model_global->weights); // veya döngüyle yazdýr
        }
        else {
            ICG_printf("Agirliklar henuz baslatilmadi veya boyutsuz.\n");
        }
        ICG_printf("Baslangic Bias: %f\n", model_global->bias);
    }
    if (!model_global || X_data_global.Y() == 0 || y_data_global.Y() == 0) {
        if (hMLE_HWND_global) ICG_printf("Lutfen once veriyi yukleyin!\n");
        return;
    }
    if (hMLE_HWND_global) ICG_printf("Egitim baslatiliyor...\n");
    model_global->train(X_data_global, y_data_global, 0.01, 1000);
    if (hMLE_HWND_global) {
        ICG_printf("Egitim tamamlandi!\nAgirliklar:\n");
        DisplayMatrix(model_global->weights); // Varsayým: DisplayMatrix varsayýlan pencereye yazar
        ICG_printf("\nBias: %f\n", model_global->bias);
    }
}

void Predict_Clicked() {
    if (!model_global || model_global->weights.Y() == 0) {
        if (hMLE_HWND_global) ICG_printf("Lutfen once modeli egitin!\n");
        return;
    }
    ICBYTES predictions = model_global->predict(X_data_global);
    if (hMLE_HWND_global) {
        ICG_printf("Tahminler (egitim verisi uzerinden):\n");
        DisplayMatrix(predictions); // Varsayým: DisplayMatrix varsayýlan pencereye yazar
    }
}

void OnExitApp(void* param) {
    if (model_global) {
        delete model_global;
        model_global = nullptr;
    }
    // Diðer kaynaklarý serbest býrakma iþlemleri (gerekirse)
}

// --- Ana GUI Kurulumu ---
void ICGUI_Create() {
    ICG_MWSize(850, 650);
    ICG_MWTitle("ICBYTES ile Lojistik Regresyon");
    ICG_SetOnExit(OnExitApp, NULL);
}

void ICGUI_main() {
    int temp_mle_icgui_handle;
    ICG_Button(10, 10, 180, 30, "Veri Yukle (features/labels.csv)", LoadData_Clicked);
    ICG_Button(10, 50, 180, 30, "Modeli Egit", TrainModel_Clicked);
    ICG_Button(10, 90, 180, 30, "Tahmin Et (Egitim Verisi)", Predict_Clicked);

    temp_mle_icgui_handle = ICG_MLEditSunken(200, 10, 630, 600, "", SCROLLBAR_HV);
    hMLE_HWND_global = ICG_GetHWND(temp_mle_icgui_handle); // ICGUI int handle'dan HWND al

    if (hMLE_HWND_global != 0) {
        ICG_SetPrintWindow(hMLE_HWND_global); // Varsayýlan yazdýrma hedefini ayarla

        ICG_printf("Hos geldiniz! Lutfen 'features.csv' ve 'labels.csv' dosyalarini proje\n");
        ICG_printf("klasorune yerlestirdikten sonra 'Veri Yukle' butonuna tiklayin.\n\n");
        ICG_printf("Ornek features.csv:\n1.0,2.5\n0.5,1.2\n...\n\n");
        ICG_printf("Ornek labels.csv (tek sutunlu):\n1\n0\n...\n");
    }
    else {
        // HWND alýnamadýysa bir hata mesajý (örneðin MessageBox veya konsola)
        // MessageBox(NULL, L"Metin kutusu olusturulamadi!", L"Hata", MB_OK | MB_ICONERROR);
    }
}