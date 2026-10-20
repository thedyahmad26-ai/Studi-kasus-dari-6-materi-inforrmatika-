# Studi Kasus 6: Atribusi Penyebab Deforestasi dari Citra Satelit dan AI
**Materi:** Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI

## Latar Belakang
Sistem seperti GLAD/RADD mendeteksi *di mana* hutan hilang, tetapi tidak menjelaskan *mengapa*. Penegakan hukum dan kebijakan membutuhkan penyebab langsung: sawit, tambang, jalan, ladang kecil, atau kebakaran.

## Kaitan dengan Materi
1. **Input:** peringatan deforestasi + citra Sentinel-2 pasca-kejadian.
2. **Fitur:** 6 band spektral, indeks NDVI/NBR/BSI, serta fitur geometri (luas, elongasi poligon).
3. **Model:** Random Forest multikelas.
4. **Output:** penyebab prediksi + tingkat keyakinan untuk tiap alert (`outputs/atribusi_alert.csv`).

Intuisi fitur: jalan itu panjang dan sempit (elongasi tinggi), bekas bakar punya NBR rendah dan SWIR2 tinggi, tambang punya BSI tinggi.

## Menjalankan
```bash
python main.py
```
Output: `outputs/deforestation_drivers.png`, `outputs/atribusi_alert.csv`.

## Batasan & Pengembangan
- Data sintetis, jadi akurasi tinggi tidak mencerminkan kondisi nyata.
- Langkah nyata: ekstrak piksel dari Sentinel-2 via Google Earth Engine/`rasterio` untuk poligon alert, lalu latih dengan label lapangan.
- Pertimbangkan CNN atau *foundation model* citra satelit untuk menangkap konteks spasial.
