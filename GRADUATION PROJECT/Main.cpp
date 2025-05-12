#include "icb_gui.h"
#include "icbytes.h"
#include "ic_media.h" // Bu projede doðrudan kullanýlmýyor gibi ama kalabilir

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <stdexcept>

// --- Global GUI Deðiþkenleri ---
HWND hMLE_HWND_global = 0;

// --- Lojistik Regresyon Modeli ve Veri Deðiþkenleri ---
class LogisticRegressionModel;
LogisticRegressionModel* model_global = nullptr;
ICBYTES X_data_global, y_data_global;

// --- Yardýmcý Fonksiyon: CSV'den ICBYTES matrisine veri yükleme (GÜNCELLENMÝÞ) ---
bool load_csv_to_icbytes(const std::string& filename, ICBYTES& matrix, long long& out_rows, long long& out_cols) {
    if (hMLE_HWND_global) ICG_printf("--- load_csv_to_icbytes basladi: %s ---\n", filename.c_str());
    std::ifstream file(filename);
    if (!file.is_open()) {
        if (hMLE_HWND_global) ICG_printf("Hata: %s dosyasi acilamadi!\n", filename.c_str());
        return false;
    }

    std::vector<std::vector<double>> data_buffer;
    std::string line;
    out_cols = 0;

    if (std::getline(file, line)) {
        if (hMLE_HWND_global) ICG_printf("Ilk satir okundu (%s): '%s'\n", filename.c_str(), line.c_str());
        std::stringstream ss_cols(line);
        std::string cell_cols;
        while (std::getline(ss_cols, cell_cols, ',')) {
            // Hücredeki olasý baþýndaki/sonundaki boþluklarý temizle (sütun sayýsýný etkilememeli ama genel temizlik)
            cell_cols.erase(0, cell_cols.find_first_not_of(" \t\n\r\f\v"));
            cell_cols.erase(cell_cols.find_last_not_of(" \t\n\r\f\v") + 1);
            if (!cell_cols.empty()) { // Sadece boþ olmayan hücreler sütun sayýsýný artýrsýn
                out_cols++;
            }
            else if (line.find(',') != std::string::npos) { // Eðer satýrda virgül varsa ama hücre boþsa, yine de bir sütun sayýlýr
                out_cols++;
            }
        }

        // Eðer ilk satýrda hiç virgül yoksa ve satýr boþ deðilse, tek sütunlu olabilir
        if (out_cols == 0 && !line.empty() && line.find_first_not_of(" \t\n\r\f\v") != std::string::npos) {
            out_cols = 1;
            if (hMLE_HWND_global) ICG_printf("Ilk satirda virgul bulunamadi, sutun sayisi 1 olarak ayarlandi (%s).\n", filename.c_str());
        }
        if (hMLE_HWND_global) ICG_printf("Belirlenen sutun sayisi (%s): %lld\n", filename.c_str(), out_cols);

        file.clear();
        file.seekg(0, std::ios::beg);
    }
    else { // Ýlk satýr okunamadýysa (dosya boþ olabilir)
        if (hMLE_HWND_global) ICG_printf("Hata: %s dosyasi bos veya ilk satir okunamadi (sutun sayisi belirlenemedi).\n", filename.c_str());
        file.close();
        return false;
    }


    if (out_cols == 0) {
        if (hMLE_HWND_global) ICG_printf("Hata: %s dosyasi icin sutun sayisi 0 olarak belirlendi.\n", filename.c_str());
        file.close();
        return false;
    }

    out_rows = 0;
    int line_counter = 0;
    while (std::getline(file, line)) {
        line_counter++;
        // Satýrýn baþýndaki ve sonundaki tüm boþluklarý temizle
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (line.empty() && file.eof()) continue; // Dosya sonundaki boþ satýrlarý atla
        if (line.empty() && !file.eof() && line_counter > 1) { // Aradaki boþ satýrlarý logla ve atla
            if (hMLE_HWND_global) ICG_printf("Uyari (%s): Bos satir %d atlaniyor.\n", filename.c_str(), line_counter);
            continue;
        }


        if (hMLE_HWND_global && line_counter <= 7) {
            ICG_printf("Okunan satir %d (%s) (temizlenmis): '%s'\n", line_counter, filename.c_str(), line.c_str());
        }
        std::vector<double> row_vector;
        std::stringstream ss_row(line);
        std::string cell_row;
        long long current_cols_in_row = 0;
        while (std::getline(ss_row, cell_row, ',')) {
            cell_row.erase(0, cell_row.find_first_not_of(" \t\n\r\f\v"));
            cell_row.erase(cell_row.find_last_not_of(" \t\n\r\f\v") + 1);

            if (!cell_row.empty()) {
                try {
                    row_vector.push_back(std::stod(cell_row));
                }
                catch (const std::invalid_argument& ia) {
                    if (hMLE_HWND_global) ICG_printf("Hata (%s): Satir %d, gecersiz sayi '%s'. Satir atlandi.\n", filename.c_str(), line_counter, cell_row.c_str());
                    row_vector.clear();
                    break;
                }
                catch (const std::out_of_range& oor) {
                    if (hMLE_HWND_global) ICG_printf("Hata (%s): Satir %d, sayi '%s' aralik disinda. Satir atlandi.\n", filename.c_str(), line_counter, cell_row.c_str());
                    row_vector.clear();
                    break;
                }
            }
            else if (line.find(',') != std::string::npos) { // Eðer orijinal satýrda virgül varsa boþ hücre de sayýlýr
                // Ýsteðe baðlý: Boþ hücreleri 0 olarak kabul et veya hata ver
                // row_vector.push_back(0.0); // Örneðin 0 olarak kabul et
            }
            current_cols_in_row++; // Her virgülle ayrýlan bölüm bir sütun sayýlýr (boþ olsa bile)
        }

        // Eðer satýr tamamen boþ deðilse ama parse edilen hücre yoksa ve tek sütun bekleniyorsa
        if (out_cols == 1 && current_cols_in_row == 0 && !line.empty() && line.find_first_not_of(" \t\n\r\f\v") != std::string::npos) {
            try {
                row_vector.push_back(std::stod(line)); // Tüm satýrý tek bir sayý olarak parse etmeyi dene
                current_cols_in_row = 1;
            }
            catch (...) { /* Hata oluþursa aþaðýda yakalanacak */ }
        }


        if (!row_vector.empty() && current_cols_in_row == out_cols) {
            data_buffer.push_back(row_vector);
            out_rows++;
        }
        else if (!line.empty() && current_cols_in_row != out_cols) {
            if (hMLE_HWND_global) ICG_printf("Uyari (%s): Satir %d, %lld hucre bulundu, %lld sutun ayristirildi, atlaniyor (beklenen %lld). Satir: '%s'\n", filename.c_str(), line_counter, (long long)row_vector.size(), current_cols_in_row, out_cols, line.c_str());
        }
    }
    file.close();

    if (out_rows == 0) {
        if (hMLE_HWND_global) ICG_printf("Hata: %s dosyasinda gecerli veri satiri bulunamadi (out_cols: %lld).\n", filename.c_str(), out_cols);
        return false;
    }

    if (CreateMatrix(matrix, out_rows, out_cols, ICB_DOUBLE) != 0) {
        if (hMLE_HWND_global) ICG_printf("Hata: CreateMatrix basarisiz oldu (%lldx%lld).\n", out_rows, out_cols);
        return false;
    }

    for (long long i = 0; i < out_rows; ++i) {
        for (long long j = 0; j < out_cols; ++j) {
            matrix.D(i + 1, j + 1) = data_buffer[i][j];
        }
    }
    if (hMLE_HWND_global) ICG_printf("--- load_csv_to_icbytes bitti (%s): %lld satir, %lld sutun yuklendi.---\n", filename.c_str(), out_rows, out_cols);
    return true;
}

// --- Yardýmcý Fonksiyonlar: Eleman bazlý iþlemler ---
ICBYTES elementwise_operation(ICBYTES& mat, double (*op_func)(double)) {
    ICBYTES result;
    if (mat.Y() <= 0 || mat.X() <= 0) { // Geçersiz matris boyutu
        if (hMLE_HWND_global) ICG_printf("Hata: elementwise_operation gecersiz matris boyutu %lldx%lld\n", mat.Y(), mat.X());
        return result; // Boþ result döndür
    }
    CreateMatrix(result, mat.Y(), mat.X(), ICB_DOUBLE);
    for (long long i = 1; i <= mat.Y(); ++i) {
        for (long long j = 1; j <= mat.X(); ++j) {
            result.D(i, j) = op_func(mat.D(i, j));
        }
    }
    return result;
}

double safe_log(double val) {
    if (val <= 1e-9) return std::log(1e-9);
    if (val >= (1.0 - 1e-9)) return std::log(1.0 - 1e-9);
    return std::log(val);
}

ICBYTES elementwise_exp(ICBYTES& mat) { return elementwise_operation(mat, std::exp); }
ICBYTES elementwise_log(ICBYTES& mat) { return elementwise_operation(mat, safe_log); }


// --- Sigmoid Fonksiyonu ---
ICBYTES sigmoid(ICBYTES& z_ref) {
    ICBYTES result; // Boþ baþlat
    if (z_ref.Y() <= 0 || z_ref.X() <= 0) {
        if (hMLE_HWND_global) ICG_printf("Hata: sigmoid fonksiyonuna gecersiz boyutlu matris geldi %lldx%lld\n", z_ref.Y(), z_ref.X());
        return result; // Boþ result döndür
    }

    ICBYTES one_mat;
    CreateMatrix(one_mat, z_ref.Y(), z_ref.X(), ICB_DOUBLE);
    one_mat = 1.0;

    ICBYTES temp_z = z_ref;
    temp_z *= -1.0;
    ICBYTES exp_neg_z = elementwise_exp(temp_z);
    if (exp_neg_z.Y() == 0) return result; // elementwise_exp hata döndürdüyse

    ICBYTES denominator = one_mat;
    denominator += exp_neg_z;

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
        if (num_features <= 0) {
            if (hMLE_HWND_global) ICG_printf("Hata: initialize_weights gecersiz ozellik sayisi %lld\n", num_features);
            return;
        }
        CreateMatrix(weights, num_features, 1, ICB_DOUBLE);
        weights = 0.0;
    }

    double compute_cost(ICBYTES& y_true, ICBYTES& h_pred) {
        long long m = y_true.Y();
        if (m == 0 || h_pred.Y() != m || y_true.X() != 1 || h_pred.X() != 1) {
            if (hMLE_HWND_global) ICG_printf("Hata: compute_cost gecersiz matris boyutlari y_true:%lldx%lld h_pred:%lldx%lld\n", y_true.Y(), y_true.X(), h_pred.Y(), h_pred.X());
            return 0.0; // veya bir hata deðeri
        }

        ICBYTES temp_h_pred = h_pred;
        ICBYTES log_h = elementwise_log(temp_h_pred);
        if (log_h.Y() == 0) return 0.0;


        ICBYTES one_mat; CreateMatrix(one_mat, m, 1, ICB_DOUBLE); one_mat = 1.0;

        ICBYTES one_minus_h = one_mat;
        one_minus_h -= h_pred;
        ICBYTES temp_one_minus_h = one_minus_h;
        ICBYTES log_one_minus_h = elementwise_log(temp_one_minus_h);
        if (log_one_minus_h.Y() == 0) return 0.0;


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

        if (m == 0 || n == 0 || y_train.Y() != m || y_train.X() != 1) {
            if (hMLE_HWND_global) ICG_printf("Hata: train fonksiyonuna gecersiz boyutlu matrisler geldi X:%lldx%lld y:%lldx%lld\n", m, n, y_train.Y(), y_train.X());
            return;
        }


        if (weights.Y() != n || weights.X() != 1) { // Aðýrlýklar doðru boyutta deðilse veya boþsa
            initialize_weights(n);
            if (weights.Y() != n) return; // initialize_weights baþarýsýz olduysa çýk
        }


        ICBYTES X_T;
        transpose(X_train, X_T); // X_T = X_train'in transpozu
        if (X_T.Y() == 0) { if (hMLE_HWND_global) ICG_printf("Hata: Transpoz basarisiz X_T olusturulamadi.\n"); return; }


        for (int iter = 0; iter < iterations; ++iter) {
            ICBYTES z;
            z.dot(X_train, weights); // z = X_train * weights
            if (z.Y() == 0) { if (hMLE_HWND_global) ICG_printf("Hata: z.dot(X_train, weights) basarisiz. Iter: %d\n", iter); return; }

            z += bias; // Her elemana bias ekle

            ICBYTES h = sigmoid(z);
            if (h.Y() == 0) { if (hMLE_HWND_global) ICG_printf("Hata: sigmoid(z) basarisiz. Iter: %d\n", iter); return; }


            ICBYTES error = h;
            error -= y_train; // error = h - y_train

            ICBYTES dw;
            dw.dot(X_T, error); // dw = X_T * error
            if (dw.Y() == 0) { if (hMLE_HWND_global) ICG_printf("Hata: dw.dot(X_T, error) basarisiz. Iter: %d\n", iter); return; }

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
        if (X_test.Y() == 0 || X_test.X() == 0 || weights.Y() == 0 || weights.X() == 0 || X_test.X() != weights.Y()) {
            if (hMLE_HWND_global) ICG_printf("Hata: predict_proba gecersiz matris boyutlari X_test:%lldx%lld weights:%lldx%lld\n", X_test.Y(), X_test.X(), weights.Y(), weights.X());
            return z; // Boþ z döndür
        }
        z.dot(X_test, weights);
        if (z.Y() == 0) return z;
        z += bias;
        return sigmoid(z);
    }

    ICBYTES predict(ICBYTES& X_test, double threshold = 0.5) {
        ICBYTES probabilities = predict_proba(X_test);
        ICBYTES predictions;
        if (probabilities.Y() == 0) return predictions; // predict_proba hata döndürdüyse

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
            if (hMLE_HWND_global) ICG_printf("Hata: Ozellik ve etiket boyutlari uyusmuyor! (labels.csv %lldx1 olmali, X %lld ornek)\n", rows_x, rows_x);
            return;
        }
        if (hMLE_HWND_global) {
            ICG_printf("Veri yuklendi:\n");
            ICG_printf("Ozellikler (X): %lld ornek, %lld ozellik\n", X_data_global.Y(), X_data_global.X());
            ICG_printf("Etiketler (y): %lld ornek, %lld sutun\n", y_data_global.Y(), y_data_global.X());
        }
        if (model_global) delete model_global;
        model_global = new LogisticRegressionModel();
    }
    else {
        if (hMLE_HWND_global) ICG_printf("Veri yukleme basarisiz! Lutfen features.csv ve labels.csv dosyalarini kontrol edin.\n");
    }
}

void TrainModel_Clicked() {
    if (!model_global || X_data_global.Y() == 0 || y_data_global.Y() == 0) {
        if (hMLE_HWND_global) ICG_printf("Lutfen once veriyi dogru sekilde yukleyin!\n");
        return;
    }
    if (hMLE_HWND_global) ICG_printf("Egitim baslatiliyor...\n");
    model_global->train(X_data_global, y_data_global, 0.01, 1000); // Örnek parametreler
    if (hMLE_HWND_global) {
        ICG_printf("Egitim tamamlandi!\nAgirliklar:\n");
        DisplayMatrix(model_global->weights);
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
        if (predictions.Y() > 0) DisplayMatrix(predictions);
        else ICG_printf("Tahminler olusturulamadi.\n");
    }
}

void OnExitApp(void* param) {
    if (model_global) {
        delete model_global;
        model_global = nullptr;
    }
}

// --- Ana GUI Kurulumu ---
void ICGUI_Create() {
    ICG_MWSize(850, 650);
    ICG_MWTitle("ICBYTES ile Lojistik Regresyon");
    ICG_SetOnExit(OnExitApp, NULL);
}

void ICGUI_main() {
    int temp_mle_icgui_handle;
    ICG_Button(10, 10, 250, 30, "Veri Yukle (features/labels.csv)", LoadData_Clicked); // Buton metnini uzattým
    ICG_Button(10, 50, 250, 30, "Modeli Egit", TrainModel_Clicked);
    ICG_Button(10, 90, 250, 30, "Tahmin Et (Egitim Verisi)", Predict_Clicked);

    temp_mle_icgui_handle = ICG_MLEditSunken(270, 10, 560, 600, "", SCROLLBAR_HV);
    hMLE_HWND_global = ICG_GetHWND(temp_mle_icgui_handle);

    if (hMLE_HWND_global != 0) {
        ICG_SetPrintWindow(hMLE_HWND_global);

        ICG_printf("Hos geldiniz! Lutfen 'features.csv' ve 'labels.csv' dosyalarini proje\n");
        ICG_printf("klasorune yerlestirdikten sonra 'Veri Yukle' butonuna tiklayin.\n\n");
        ICG_printf("Ornek features.csv (N ornek, M ozellik):\n");
        ICG_printf("ozellik1_1,ozellik2_1,...,ozellikM_1\n");
        ICG_printf("ozellik1_2,ozellik2_2,...,ozellikM_2\n...\n\n");
        ICG_printf("Ornek labels.csv (N ornek, tek sutunlu etiket):\n");
        ICG_printf("etiket_1\netiket_2\n...\n");
    }
    else {
        // Opsiyonel: Konsola veya MessageBox ile hata
    }
}