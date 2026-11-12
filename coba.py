from agent import SignatureAgent

agen = SignatureAgent("models/ResNet18.pth")
hasil = agen.jalankan("contoh_tandatangan.jpg")
print("Keputusan:", hasil["keputusan"])
print(f"Asli {hasil['p_asli']*100:.1f}% | Palsu {hasil['p_palsu']*100:.1f}%")
print(hasil["pesan"])