"""Studi Kasus 6 - Attributing Direct Drivers to Deforestation Alerts (Satellite + AI).
Untuk tiap peringatan deforestasi, klasifikasikan penyebab langsungnya dari
fitur spektral (Sentinel-2 sintetis) + geometri."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

OUT = os.path.join(os.path.dirname(__file__), "outputs"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)
KELAS = ["Perkebunan sawit", "Pertanian kecil", "Tambang", "Jalan/infrastruktur", "Kebakaran"]
# rata-rata reflektansi (B2 blue, B3 green, B4 red, B8 NIR, B11 SWIR1, B12 SWIR2) PASCA-deforestasi
SPEKTRAL = np.array([
    [.04, .07, .05, .30, .18, .09],   # sawit muda: vegetasi muncul cepat
    [.05, .09, .08, .26, .22, .13],   # ladang campuran
    [.12, .15, .20, .24, .38, .33],   # tanah terbuka terang
    [.10, .12, .15, .20, .30, .26],   # linear, tanah terbuka
    [.03, .04, .06, .10, .20, .24],   # bekas bakar: gelap, SWIR2 tinggi
])
GEOM = np.array([  # (luas ha, rasio memanjang)
    [25, 1.5], [3, 1.8], [12, 1.4], [4, 9.0], [40, 1.6]])

n = 600
rows = []
for k in range(5):
    refl = np.clip(rng.normal(SPEKTRAL[k], .06, (n, 6)), .005, 1)
    luas = np.clip(rng.lognormal(np.log(GEOM[k, 0]), .5, n), .2, None)
    elong = np.clip(rng.normal(GEOM[k, 1], .4 + .1*GEOM[k, 1], n), 1, None)
    rows.append(np.c_[refl, luas, elong, np.full(n, k)])
d = pd.DataFrame(np.vstack(rows), columns=["B2", "B3", "B4", "B8", "B11", "B12", "luas_ha", "elongasi", "kelas"])

# Indeks spektral
d["NDVI"] = (d.B8 - d.B4) / (d.B8 + d.B4)
d["NBR"] = (d.B8 - d.B12) / (d.B8 + d.B12)      # sensitif terhadap bekas bakar
d["BSI"] = ((d.B11 + d.B4) - (d.B8 + d.B2)) / ((d.B11 + d.B4) + (d.B8 + d.B2))  # tanah terbuka

X, y = d.drop(columns="kelas"), d.kelas.astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=0)
rf = RandomForestClassifier(300, random_state=0, class_weight="balanced").fit(Xtr, ytr)
pred = rf.predict(Xte)
print(classification_report(yte, pred, target_names=KELAS))

# Atribusi peringatan baru
alert = pd.DataFrame({"alert_id": range(1, 6)})
baru = X.sample(5, random_state=1).reset_index(drop=True)
alert["penyebab_prediksi"] = [KELAS[i] for i in rf.predict(baru)]
alert["keyakinan"] = rf.predict_proba(baru).max(axis=1).round(2)
print(alert); alert.to_csv(f"{OUT}/atribusi_alert.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for k in range(5):
    s = d[d.kelas == k]; ax[0].scatter(s.NDVI, s.NBR, s=6, alpha=.4, label=KELAS[k])
ax[0].set(xlabel="NDVI", ylabel="NBR", title="Ruang fitur spektral"); ax[0].legend(fontsize=7, markerscale=2)
ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=KELAS, ax=ax[1], colorbar=False, xticks_rotation=40)
ax[1].set_title("Confusion matrix")
pd.Series(rf.feature_importances_, X.columns).sort_values().plot.barh(ax=ax[2], color="#1b9e77")
ax[2].set_title("Feature importance")
plt.tight_layout(); plt.savefig(f"{OUT}/deforestation_drivers.png", dpi=120)
