import os, csv, datetime
import cv2, numpy as np, torch, torch.nn as nn
from torchvision import models, transforms

SIZE = 128

class SignatureAgent:
    def __init__(self, model_path, batas_ragu=(0.40, 0.60), log_path="log_keputusan.csv"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.batas_ragu = batas_ragu
        self.log_path = log_path

    # 1. PERSEPSI: baca gambar dan olah jadi masukan model (sama dengan saat training)
    def persepsi(self, path):
        color = cv2.imread(path)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if color is None or gray is None:
            raise FileNotFoundError(f"Gambar tidak terbaca: {path}")
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(otsu > 0)
        if len(ys) == 0:
            raise ValueError("Tidak ada goresan yang terdeteksi.")
        crop = otsu[ys.min():ys.max()+1, xs.min():xs.max()+1]
        h, w = crop.shape
        s = max(h, w)
        canvas = np.zeros((s, s), np.uint8)
        canvas[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = crop
        final = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        return {"asli": color, "grayscale": gray, "otsu": otsu, "akhir": final}

    # 2. PENALARAN: model menghitung probabilitas
    def hitung_probabilitas(self, akhir):
        x = torch.from_numpy(akhir).float().div(255).unsqueeze(0)
        x = self.norm(x.repeat(3, 1, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return float(p[0]), float(p[1])          # (asli, palsu)

    # 3. KEPUTUSAN: aturan tiga arah
    def keputusan(self, p_palsu):
        rendah, tinggi = self.batas_ragu
        if p_palsu >= tinggi:
            return "PALSU"
        if p_palsu <= rendah:
            return "ASLI"
        return "RAGU"

    # 4. AKSI: pesan dan catatan log
    def aksi(self, hasil, path, p_asli, p_palsu):
        pesan = {"ASLI": "Tanda tangan diterima.",
                 "PALSU": "Tanda tangan ditolak dan ditandai mencurigakan.",
                 "RAGU": "Keyakinan rendah: minta foto ulang atau verifikasi manual."}[hasil]
        baru = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            tulis = csv.writer(f)
            if baru:
                tulis.writerow(["waktu", "file", "p_asli", "p_palsu", "keputusan"])
            tulis.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                            os.path.basename(path), round(p_asli, 4), round(p_palsu, 4), hasil])
        return pesan

    def jalankan(self, path):
        tahap = self.persepsi(path)
        p_asli, p_palsu = self.hitung_probabilitas(tahap["akhir"])
        hasil = self.keputusan(p_palsu)
        pesan = self.aksi(hasil, path, p_asli, p_palsu)
        return {"keputusan": hasil, "p_asli": p_asli, "p_palsu": p_palsu,
                "pesan": pesan, "tahap": tahap}