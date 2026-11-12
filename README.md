# AI Agent Verifikasi Tanda Tangan Tulisan Tangan

Rancang bangun AI agent untuk membedakan tanda tangan **asli** dan **palsu** dengan pengolahan citra digital dan Convolutional Neural Network (ResNet18).

Agen menerima satu gambar tanda tangan, mengolahnya, menghitung tingkat keyakinan "asli" dan "palsu", mengambil keputusan (**ASLI / PALSU / RAGU**), lalu mencatat keputusannya ke file log.

> **Status:** prototipe untuk tugas kuliah. Akurasi model pada data uji sekitar **65%**, jadi hasilnya belum layak dipakai sebagai penentu keaslian tanda tangan di dunia nyata. Lihat bagian [Keterbatasan](#keterbatasan).

---

## Daftar isi

1. [Gambaran umum](#gambaran-umum)
2. [Cara kerja agen](#cara-kerja-agen)
3. [Struktur proyek](#struktur-proyek)
4. [Instalasi](#instalasi)
5. [Cara menjalankan](#cara-menjalankan)
6. [Penjelasan kode agen](#penjelasan-kode-agen)
7. [Aturan keputusan](#aturan-keputusan)
8. [Data dan pelatihan model](#data-dan-pelatihan-model)
9. [Hasil evaluasi](#hasil-evaluasi)
10. [Dashboard (opsional)](#dashboard-opsional)
11. [Pemecahan masalah](#pemecahan-masalah)
12. [Keterbatasan](#keterbatasan)
13. [Etika dan privasi](#etika-dan-privasi)

---

## Gambaran umum

| Item | Isi |
|---|---|
| Tugas | Klasifikasi biner tanda tangan: asli (0) vs palsu (1) |
| Model | ResNet18 (transfer learning dari ImageNet), 2 kelas |
| Input model | Gambar biner 128x128 hasil pengolahan citra |
| Keluaran agen | Keputusan, persentase keyakinan, pesan tindakan, dan catatan log |
| Bahasa dan pustaka | Python, PyTorch, OpenCV, NumPy |
| Pelatihan | Google Colab (GPU). Penggunaan agen cukup di CPU |

Proyek ini dikembangkan bersama untuk dua mata kuliah, dengan penekanan berbeda:

- **Pengolahan Citra Digital:** tahap preprocessing, projection profile sebagai baseline klasik, dan perbandingan model.
- **Kecerdasan Tiruan:** perancangan agen (persepsi, penalaran, keputusan, aksi).

---

## Cara kerja agen

Agen dirancang dengan kerangka **PEAS**:

| Komponen | Isi |
|---|---|
| **P**erformance | Akurasi, presisi, FAR, FRR |
| **E**nvironment | Gambar tanda tangan tulisan tangan (foto atau scan) |
| **S**ensors | Gambar yang diberikan pengguna |
| **A**ctuators | Keputusan, persentase, pesan, dan baris log |

Alur kerjanya:

```
Gambar tanda tangan
        |
        v
[1. PERSEPSI]  grayscale -> Gaussian blur -> Otsu -> crop -> padding persegi -> resize 128x128
        |
        v
[2. PENALARAN] ResNet18 menghitung probabilitas (asli, palsu) dengan softmax
        |
        v
[3. KEPUTUSAN] aturan tiga arah: ASLI / RAGU / PALSU
        |
        v
[4. AKSI]      pesan tindakan + catatan ke log_keputusan.csv
```

Tahap pengolahan citra pada langkah 1 **harus sama persis** dengan yang dipakai saat training. Jika diubah, hasil prediksi tidak dapat dipercaya.

---

## Struktur proyek

```
ttd_agent/
├── models/
│   └── ResNet18.pth          <- bobot model (sekitar 45 MB), unduh dari Google Drive
├── venv/                     <- virtual environment (dibuat saat instalasi)
├── agent.py                  <- kelas SignatureAgent
├── coba.py                   <- skrip untuk mencoba agen
├── contoh_tandatangan.jpg    <- satu foto tanda tangan untuk pengujian
├── log_keputusan.csv         <- dibuat otomatis setelah agen dijalankan
└── README.md                 <- dokumen ini
```

Folder dan file yang dibuat sendiri: `ttd_agent/`, `models/`, `agent.py`, `coba.py`, `README.md`. Sedangkan `venv/` dan `log_keputusan.csv` muncul otomatis.

---

## Instalasi

### 1. Prasyarat

- Python 3.9 atau lebih baru
- Visual Studio Code (disarankan, dengan ekstensi **Python**)
- File model `ResNet18.pth`

### 2. Siapkan folder

1. Buat folder `ttd_agent` di laptopmu, lalu buka di VS Code lewat **File > Open Folder**.
2. Buat subfolder `models`.
3. Unduh `ResNet18.pth` dari Google Drive (`Drive Saya > Signature_Project > models > ResNet18.pth`), lalu taruh di folder `models`.
4. Taruh satu foto tanda tangan bernama `contoh_tandatangan.jpg` di folder `ttd_agent`.

> Jangan membuka file `.pth` di editor VS Code. File itu biner, jadi pesan *"The file is not displayed in the text editor"* adalah hal yang normal.

### 3. Buat virtual environment dan pasang pustaka

Buka terminal di VS Code (**Terminal > New Terminal**), lalu jalankan.

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pandas
```

Jika PowerShell menolak mengaktifkan venv dengan pesan *"running scripts is disabled"*:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pandas
```

Tanda venv aktif: awal baris terminal menampilkan `(venv)`.

Jika perintah PyTorch di atas bermasalah di laptopmu, ambil perintah yang sesuai di <https://pytorch.org> (pilih Pip, dan CPU bila tidak ada GPU NVIDIA).

### 4. Pilih interpreter di VS Code (opsional)

Tekan `Ctrl+Shift+P`, ketik **Python: Select Interpreter**, lalu pilih yang berada di `venv`.

---

## Cara menjalankan

### 1. Isi `coba.py`

```python
from agent import SignatureAgent

agen = SignatureAgent("models/ResNet18.pth")
hasil = agen.jalankan("contoh_tandatangan.jpg")

print("Keputusan:", hasil["keputusan"])
print(f"Asli {hasil['p_asli']*100:.1f}% | Palsu {hasil['p_palsu']*100:.1f}%")
print(hasil["pesan"])
```

### 2. Simpan semua file lalu jalankan

Tekan `Ctrl+K` lalu `S` untuk menyimpan semua file. Titik bulat di tab file berarti belum tersimpan.

```powershell
python coba.py
```

### 3. Contoh keluaran

```
Keputusan: ASLI
Asli 62.3% | Palsu 37.7%
Tanda tangan diterima.
```

Setiap keputusan otomatis tercatat di `log_keputusan.csv`:

```
waktu,file,p_asli,p_palsu,keputusan
2026-11-12T10:21:05,contoh_tandatangan.jpg,0.6231,0.3769,ASLI
```

### Syarat gambar masukan

Model dilatih dari potongan (crop) tanda tangan, jadi gambar uji harus:

- berisi **satu** tanda tangan
- dipotong **rapat** di sekitar tanda tangan
- berada di **kertas polos** dengan pencahayaan merata, tanpa bayangan
- berformat JPG atau PNG yang valid

Foto seluruh lembar kertas akan menghasilkan prediksi yang tidak berarti.

---

## Penjelasan kode agen

`agent.py` berisi kelas `SignatureAgent`:

| Metode | Peran | Isi |
|---|---|---|
| `__init__(model_path, batas_ragu, log_path)` | Inisialisasi | Memuat ResNet18 dan bobot dari `model_path`, menyimpan batas keputusan dan lokasi log |
| `persepsi(path)` | Persepsi | Membaca gambar lalu menjalankan grayscale, blur, Otsu, crop, padding, dan resize |
| `hitung_probabilitas(akhir)` | Penalaran | Menjalankan ResNet18 dan mengembalikan `(p_asli, p_palsu)` lewat softmax |
| `keputusan(p_palsu)` | Keputusan | Menerapkan aturan tiga arah |
| `aksi(hasil, path, p_asli, p_palsu)` | Aksi | Membuat pesan tindakan dan menulis baris ke log CSV |
| `jalankan(path)` | Orkestrasi | Menjalankan keempat tahap berurutan dan mengembalikan dictionary hasil |

Hasil `jalankan()` berupa dictionary:

| Kunci | Isi |
|---|---|
| `keputusan` | `"ASLI"`, `"PALSU"`, atau `"RAGU"` |
| `p_asli`, `p_palsu` | Probabilitas 0 sampai 1 |
| `pesan` | Pesan tindakan untuk pengguna |
| `tahap` | Dictionary gambar tiap tahap: `asli`, `grayscale`, `otsu`, `akhir` |

---

## Aturan keputusan

Agen memakai aturan tiga arah berdasarkan `p_palsu`:

| Kondisi | Keputusan | Aksi |
|---|---|---|
| `p_palsu >= 0.60` | **PALSU** | Ditolak dan ditandai mencurigakan |
| `p_palsu <= 0.40` | **ASLI** | Diterima |
| di antaranya | **RAGU** | Minta foto ulang atau verifikasi manual |

Zona **RAGU** sengaja ada karena akurasi model hanya sekitar 65%. Lebih jujur bagi agen untuk menyatakan tidak yakin daripada memaksa tebakan.

Nilai 0,40 dan 0,60 adalah **nilai awal**. Ubah lewat parameter:

```python
agen = SignatureAgent("models/ResNet18.pth", batas_ragu=(0.35, 0.65))
```

Memperlebar zona RAGU membuat agen memutuskan lebih sedikit gambar, tetapi akurasi pada gambar yang diputuskan cenderung naik. Pilih nilai batas dari data **validasi**, bukan dari data uji.

---

## Data dan pelatihan model

Pelatihan dilakukan terpisah di Google Colab (notebook `Signature.ipynb`). Folder ini hanya berisi hasil akhirnya, yaitu `ResNet18.pth`.

### Dataset

- Sumber: Kaggle, **tienen/handwritten-signature-verification**
- Isi yang dipakai: potongan (crop) tanda tangan di folder `real` dan `forged`
- Jumlah crop setelah preprocessing: sekitar 5.650 (real 2.913, forged sekitar 2.735)
- Label diambil dari nama folder `real` dan `forged`. File CSV/TSV pada dataset hanyalah catatan penugasan dan bukan label.
- Ukuran crop kecil (sekitar 100x70 piksel)

Periksa lisensi dan asal data pada halaman dataset di Kaggle sebelum dicantumkan di laporan.

### Pengolahan citra

| Langkah | Detail | Alasan |
|---|---|---|
| 1. Grayscale | Baca gambar sebagai skala abu-abu | Warna tidak relevan, data lebih sederhana |
| 2. Gaussian blur | Kernel 3x3 | Mengurangi noise |
| 3. Binarisasi Otsu (inverse) | Tinta putih, latar hitam | Memisahkan tinta dari kertas secara otomatis |
| 4. Crop bounding box | Potong pada area berisi tinta | Membuang ruang kosong |
| 5. Padding persegi | Letakkan di tengah kanvas persegi | Proporsi goresan tidak berubah |
| 6. Resize | 128x128, interpolasi `INTER_AREA` | Ukuran masukan seragam |

Otsu dipilih daripada Adaptive Threshold karena pada contoh data, Adaptive ikut menangkap garis buku dan kotak kertas sebagai noise.

### Pembagian data

Data dibagi **per foto asal** (`GroupShuffleSplit`, `random_state=42`) agar crop dari foto yang sama tidak muncul di data latih dan data uji sekaligus. Data uji berisi 1.134 gambar (530 asli, 604 palsu). Bagian latih dibagi lagi menjadi latih dan validasi (validasi 15% dari bagian latih).

### Pengaturan training

| Parameter | Nilai |
|---|---|
| Model | ResNet18 pretrained ImageNet, lapisan akhir diganti jadi 2 kelas |
| Optimizer | Adam, learning rate 1e-4 |
| Epoch | 8, bobot dari epoch dengan akurasi validasi terbaik yang disimpan |
| Batch size | 64 |
| Augmentasi (hanya data latih) | `RandomAffine`: rotasi 5 derajat, geser 5%, skala 0,95 sampai 1,05 |
| Normalisasi | Mean dan std ImageNet, gambar biner diduplikasi menjadi 3 channel |
| Seed | 42 |

---

## Hasil evaluasi

Evaluasi pada 1.134 data uji. Seluruh model dilatih satu kali dengan pengaturan yang sama.

| Metode | Akurasi | Presisi | Recall | F1 | FAR | FRR |
|---|---|---|---|---|---|---|
| **ResNet18** (dipakai agen) | **0,6534** | 0,6522 | 0,6525 | 0,6523 | 0,3328 | 0,3623 |
| VGG16 | 0,6429 | 0,6459 | 0,6458 | 0,6429 | 0,3990 | 0,3094 |
| ResNet18 + CBAM | 0,5697 | 0,5705 | 0,5707 | 0,5695 | 0,4454 | 0,4132 |
| Projection profile + SVM | 0,5688 | 0,5703 | 0,5705 | 0,5687 | 0,4553 | 0,4038 |
| EfficientNet-B0 | 0,5556 | 0,5557 | 0,5560 | 0,5552 | 0,4503 | 0,4377 |
| *Tebakan kelas mayoritas* | *0,533* | | | | | |

- **FAR** (False Acceptance Rate): tanda tangan palsu yang diterima sebagai asli.
- **FRR** (False Rejection Rate): tanda tangan asli yang ditolak sebagai palsu.
- Presisi, recall, dan F1 adalah rata-rata makro dua kelas.

### Cara membaca hasil

- Ketidakpastian akurasi pada 1.134 data uji sekitar **±2,8 poin**. Selisih di bawah itu tidak dianggap perbedaan nyata.
- ResNet18 dan VGG16 **setara**. ResNet18 dipilih untuk agen karena jauh lebih ringan (sekitar 45 MB dibanding sekitar 528 MB).
- Model deep learning unggul sekitar 7 sampai 9 poin dari projection profile + SVM.
- Penambahan modul CBAM tidak meningkatkan hasil pada data ini. Dugaannya, modul atensi yang diinisialisasi acak mengganggu fitur pretrained. Dugaan ini belum dibuktikan.
- Semua model mengalami overfitting: loss latih terus turun, sedangkan akurasi validasi stagnan di sekitar 60 sampai 66%.

---

## Dashboard (opsional)

Untuk tampilan web sederhana yang memperlihatkan tahap pengolahan citra dan persentase hasil, pasang Gradio:

```powershell
pip install gradio
```

Buat file `app.py`:

```python
import cv2
import gradio as gr
from agent import SignatureAgent

agen = SignatureAgent("models/ResNet18.pth")

def analisis(path):
    if path is None:
        return None, None, None, None, None, "Upload gambar terlebih dahulu."
    h = agen.jalankan(path)
    t = h["tahap"]
    rgb = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    ringkas = f"### Keputusan: **{h['keputusan']}**\n{h['pesan']}"
    return (cv2.cvtColor(t["asli"], cv2.COLOR_BGR2RGB), rgb(t["grayscale"]),
            rgb(t["otsu"]), rgb(t["akhir"]),
            {"Asli": h["p_asli"], "Palsu": h["p_palsu"]}, ringkas)

with gr.Blocks(title="Agen Verifikasi Tanda Tangan") as demo:
    gr.Markdown("# Agen Verifikasi Tanda Tangan")
    inp = gr.Image(type="filepath", label="Upload satu tanda tangan (crop rapat, kertas polos)")
    btn = gr.Button("Analisis", variant="primary")
    ringkasan = gr.Markdown()
    label = gr.Label(label="Tingkat keyakinan model")
    with gr.Row():
        g1 = gr.Image(label="1. Foto asli")
        g2 = gr.Image(label="2. Grayscale")
        g3 = gr.Image(label="3. Binarisasi Otsu")
        g4 = gr.Image(label="4. Crop + resize 128x128")
    btn.click(analisis, inp, [g1, g2, g3, g4, label, ringkasan])

demo.launch()
```

Jalankan dengan `python app.py`, lalu buka alamat lokal yang muncul di terminal (biasanya `http://127.0.0.1:7860`).

---

## Pemecahan masalah

| Gejala | Penyebab dan solusi |
|---|---|
| `python coba.py` tidak mengeluarkan apa-apa | `coba.py` belum disimpan (ada titik bulat di tab). Simpan dengan `Ctrl+S` |
| `FileNotFoundError: Gambar tidak terbaca` | Nama atau lokasi gambar salah, atau file bukan gambar valid. Samakan nama di `coba.py` dengan nama file di Explorer |
| VS Code: *An error occurred while loading the image* | File kemungkinan bukan JPG asli (WebP, HEIC, atau rusak). Buka di Paint, lalu **Save As > JPEG** |
| `ValueError: Tidak ada goresan yang terdeteksi` | Gambar kosong atau terlalu terang. Gunakan foto tanda tangan yang jelas |
| `ModuleNotFoundError: No module named 'torch'` (atau `cv2`) | Pustaka belum terpasang atau venv belum aktif. Aktifkan venv, lalu jalankan ulang `pip install` |
| Awal baris terminal tidak ada `(venv)` | Jalankan `venv\Scripts\activate` |
| PowerShell: *running scripts is disabled* | Jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, lalu aktifkan venv |
| Error saat memuat model (`FileNotFoundError` untuk `.pth`) | Pastikan file ada di `models/ResNet18.pth` dan jalankan perintah dari folder `ttd_agent` |
| Ukuran `ResNet18.pth` hanya beberapa KB | Unduhan tidak lengkap. Unduh ulang dari Drive (ukuran normal sekitar 45 MB) |
| `RuntimeError` saat `load_state_dict` | File bobot bukan dari ResNet18 dua kelas. Pastikan memakai `ResNet18.pth` |
| Hasil prediksi terasa acak | Gambar bukan crop satu tanda tangan, atau pencahayaan buruk. Lihat [syarat gambar masukan](#syarat-gambar-masukan) |

---

## Keterbatasan

- **Akurasi sekitar 65%.** Hanya sekitar 12 poin di atas tebakan kelas mayoritas (53,3%). Jangan dipakai sebagai penentu keaslian.
- **Persentase adalah tingkat keyakinan model, bukan peluang yang pasti.** Keluaran softmax tidak terkalibrasi, dan model yang overfitting cenderung terlalu yakin.
- **Tidak ada tanda tangan referensi.** Dataset tidak menyediakan identitas penulis, sehingga model hanya menilai satu gambar tanpa membandingkannya dengan tanda tangan asli milik orang yang sama. Verifikasi yang sesungguhnya memerlukan pembandingan semacam itu.
- **Binarisasi menghilangkan informasi** seperti tekanan dan ketebalan tinta, yang bisa menjadi petunjuk pemalsuan.
- **Citra kecil** (sekitar 100x70 piksel) dan datanya terbatas.
- **Domain sempit.** Model dilatih pada foto kertas dengan kondisi tertentu, sehingga performa pada jenis kertas, pulpen, atau kamera lain belum teruji.
- Setiap model dilatih satu kali, sehingga hasil bisa bervariasi sekitar 3 poin antarpercobaan.

Pengembangan yang mungkin: input grayscale tanpa binarisasi, pembandingan dengan tanda tangan referensi per orang, dataset lebih besar, dan evaluasi pada tanda tangan dengan izin dari sumber lain.

---

## Etika dan privasi

- Tanda tangan adalah data pribadi. Jangan mengunggah tanda tangan orang lain ke layanan publik tanpa izin.
- Jika mengumpulkan tanda tangan teman untuk pengujian: minta persetujuan, samarkan identitas dengan kode, jangan dipublikasikan, dan hapus setelah tugas selesai bila memungkinkan.
- Tanda tangan palsu untuk eksperimen hanya dibuat atas izin pemilik tanda tangan.
- Jangan menggunakan agen ini untuk memutuskan keaslian dokumen yang berdampak hukum atau finansial.

---

*Dibuat sebagai bagian dari tugas mata kuliah Kecerdasan Tiruan dan Pengolahan Citra Digital.*
