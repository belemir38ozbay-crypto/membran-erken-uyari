"""
╔══════════════════════════════════════════════════════════════╗
║   MEMBRAN SU ARITIM SİSTEMİ — ERKEN UYARI DASHBOARD         ║
║   Çalıştırmak için: streamlit run dashboard.py               ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, ast, re
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.impute            import SimpleImputer
from sklearn.metrics           import r2_score, mean_absolute_error
from sklearn.ensemble          import IsolationForest
from xgboost                   import XGBRegressor

# ─────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Membran Erken Uyarı Sistemi",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# RENK & STIL
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f1923; }

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0f1923 0%, #162534 50%, #0d2137 100%);
    border: 1px solid #1e3a52;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
}
.dash-title {
    font-size: 28px; font-weight: 700;
    color: #e8f4fd; letter-spacing: -0.5px; margin: 0;
}
.dash-sub {
    font-size: 13px; color: #6b9ab8; margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* Metrik kartlar */
.metric-card {
    background: #131f2e;
    border: 1px solid #1e3a52;
    border-radius: 10px;
    padding: 20px 22px;
    text-align: center;
    height: 100%;
}
.metric-value {
    font-size: 32px; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-label {
    font-size: 11px; color: #6b9ab8;
    text-transform: uppercase; letter-spacing: 1px;
    margin-top: 6px;
}
.metric-delta {
    font-size: 12px; margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* Uyarı kartları */
.alert-critical {
    background: linear-gradient(135deg, #2d0f0f, #1a0808);
    border: 1px solid #8b1a1a; border-left: 4px solid #ef4444;
    border-radius: 10px; padding: 16px 20px; margin: 8px 0;
}
.alert-warning {
    background: linear-gradient(135deg, #2d1f0f, #1a1008);
    border: 1px solid #8b4a1a; border-left: 4px solid #f97316;
    border-radius: 10px; padding: 16px 20px; margin: 8px 0;
}
.alert-caution {
    background: linear-gradient(135deg, #2d2a0f, #1a1908);
    border: 1px solid #8b7a1a; border-left: 4px solid #eab308;
    border-radius: 10px; padding: 16px 20px; margin: 8px 0;
}
.alert-ok {
    background: linear-gradient(135deg, #0f2d18, #081a0d);
    border: 1px solid #1a5c2a; border-left: 4px solid #22c55e;
    border-radius: 10px; padding: 16px 20px; margin: 8px 0;
}
.alert-title { font-size: 14px; font-weight: 600; margin: 0 0 4px; }
.alert-desc  { font-size: 12px; color: #9ca3af; margin: 0; }

/* Bölüm başlığı */
.section-title {
    font-size: 13px; font-weight: 600;
    color: #4a9eca; text-transform: uppercase;
    letter-spacing: 1.5px; margin: 20px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e3a52;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1b27 !important;
    border-right: 1px solid #1e3a52;
}
[data-testid="stSidebar"] label { color: #8ab4cc !important; font-size: 12px; }

/* Streamlit varsayılan elemanlarını karartma */
.stSlider > div > div { background: #1e3a52 !important; }
.stSelectbox > div > div { background: #131f2e !important; color: #e8f4fd !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VERİ & MODEL (cache ile hızlandır)
# ─────────────────────────────────────────────
@st.cache_data
def veri_yukle_ve_egit(dosya):
    df = pd.read_csv(dosya)

    def temizle(v):
        if pd.isna(v): return 'Unknown'
        try:
            lst = ast.literal_eval(str(v))
            if isinstance(lst, list): return '|'.join(lst)
        except: pass
        return re.sub(r'[\[\]"]','',str(v)).strip()

    le = LabelEncoder()
    for col in ['chemistry','modification','structure','filtrationMode','cpMode']:
        df[col+'_enc'] = le.fit_transform(df[col].apply(temizle).fillna('Unknown').astype(str))

    FEATURES = ['hydraulicP','CP','B','NaCl conc (M)','osmoticP','A',
                'chemistry_enc','modification_enc','structure_enc',
                'filtrationMode_enc','reportYear','CA','roughness',
                'thickness','pH','couponSize']
    TARGET = 'Rreal'

    df_ml = df[FEATURES + [TARGET]].dropna(subset=[TARGET]).copy()
    X = df_ml[FEATURES]; y = df_ml[TARGET]

    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=FEATURES)

    X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    iso = IsolationForest(contamination=0.08, random_state=42)
    anomali = iso.fit_predict(X_imp)
    df_ml = df_ml.copy()
    df_ml['anomali'] = anomali
    df_ml['tahmin'] = model.predict(X_imp)

    return df, df_ml, model, imp, X_te, y_te, y_pred, FEATURES

@st.cache_data
def simulasyon_verisi_uret(gun, fouling_tipi, gurultu_seviyesi):
    """Gerçek zamanlı sistem davranışını simüle eder."""
    t = np.arange(0, gun, 1)
    np.random.seed(42)
    g = gurultu_seviyesi / 100

    if fouling_tipi == 'Partikül Fouling':
        NPF     = 100 * np.exp(-0.018 * t) + np.random.normal(0, g*2, len(t))
        delta_P = 1.0 * (1 + 0.022 * t)   + np.random.normal(0, g*0.05, len(t))
        Rreal   = 99  - (100 - NPF.clip(0,100)) * 0.28
    elif fouling_tipi == 'Scaling':
        NPF     = 100 / (1 + 0.0008 * t**1.6) + np.random.normal(0, g*1.5, len(t))
        delta_P = 1.0 * (1 + 0.007 * t)       + np.random.normal(0, g*0.04, len(t))
        Rreal   = 99.5 - (100 - NPF.clip(0,100)) * 0.25
    elif fouling_tipi == 'Biyofouling':
        NPF     = 100 / (1 + np.exp(0.12*(t - gun*0.4))) + np.random.normal(0, g*2, len(t))
        delta_P = 1.0 + 0.6 / (1 + np.exp(-0.12*(t - gun*0.4))) + np.random.normal(0, g*0.05, len(t))
        Rreal   = 99 - (100 - NPF.clip(0,100)) * 0.32
    else:  # Organik
        NPF     = 100 * (1 - 0.0003 * t**1.3) + np.random.normal(0, g*1, len(t))
        delta_P = 1.0 + 0.0015 * t**1.1       + np.random.normal(0, g*0.03, len(t))
        Rreal   = 99.8 - (100 - NPF.clip(0,100)) * 0.2

    enerji  = 3.5 + (100 - NPF.clip(60,100)) * 0.04 + np.random.normal(0, g*0.1, len(t))
    CP_val  = 0.8 + (100 - NPF.clip(60,100)) * 0.003 + np.random.normal(0, g*0.01, len(t))

    return pd.DataFrame({
        'Gün': t,
        'NPF (%)': NPF.clip(0, 100),
        'ΔP (bar)': delta_P.clip(0.5, 8),
        'Rreal (%)': Rreal.clip(70, 100),
        'Enerji (kWh/m³)': enerji.clip(2, 10),
        'CP': CP_val.clip(0.5, 1.5),
    })


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Sistem Ayarları")
    st.markdown("---")

    uploaded = st.file_uploader("📂 CSV Dosyası Yükle", type=['csv'])

    st.markdown('<div class="section-title">Simülasyon</div>', unsafe_allow_html=True)
    fouling_tipi    = st.selectbox("Tıkanma Türü",
                        ['Partikül Fouling','Scaling','Biyofouling','Organik Fouling'])
    sim_gun         = st.slider("Simülasyon Süresi (gün)", 30, 365, 120, step=10)
    gurultu         = st.slider("Gürültü Seviyesi", 0, 50, 10,
                        help="Sensör ölçüm hatasını simüle eder")

    st.markdown('<div class="section-title">Erken Uyarı Eşikleri</div>', unsafe_allow_html=True)
    esik_npf    = st.slider("NPF Uyarı Eşiği (%)", 70, 98, 90)
    esik_rreal  = st.slider("Rreal Uyarı Eşiği (%)", 85, 99, 97)
    esik_dp     = st.slider("ΔP Uyarı Eşiği (bar)", 1.0, 6.0, 2.5, step=0.1)

    st.markdown('<div class="section-title">Maliyet (TL)</div>', unsafe_allow_html=True)
    mal_cip     = st.number_input("Kimyasal Yıkama (CIP)", value=5000,  step=500)
    mal_membran = st.number_input("Membran Değişimi",      value=120000, step=5000)
    mal_duruş   = st.number_input("Duruş Maliyeti/Gün",   value=25000,  step=1000)


# ─────────────────────────────────────────────
# VERİ YÜKLE
# ─────────────────────────────────────────────
veri_kaynagi = uploaded if uploaded else "OMD_RO_2026-06-21.csv"

try:
    df, df_ml, model, imp, X_te, y_te, y_pred, FEATURES = veri_yukle_ve_egit(veri_kaynagi)
    veri_ok = True
except:
    veri_ok = False

sim_df = simulasyon_verisi_uret(sim_gun, fouling_tipi, gurultu)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <p class="dash-title">🌊 Membran Su Arıtım — Erken Uyarı Sistemi</p>
    <p class="dash-sub">ML Destekli Fouling Tespiti &nbsp;|&nbsp; XGBoost + Isolation Forest &nbsp;|&nbsp; Maliyet Optimizasyonu</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Simülasyon İzleme",
    "🤖 ML Model",
    "💰 Maliyet Analizi",
    "🔮 Tahmin Aracı",
])


# ══════════════════════════════════════════════
# TAB 1: SİMÜLASYON
# ══════════════════════════════════════════════
with tab1:
    son = sim_df.iloc[-1]
    mevcut_gun = int(son['Gün'])

    # Durum tespiti
    uyari_npf   = son['NPF (%)']   < esik_npf
    uyari_rreal = son['Rreal (%)'] < esik_rreal
    uyari_dp    = son['ΔP (bar)']  > esik_dp

    n_uyari = sum([uyari_npf, uyari_rreal, uyari_dp])
    genel_durum = "KRİTİK" if n_uyari >= 2 else ("UYARI" if n_uyari == 1 else "NORMAL")
    durum_renk  = "#ef4444" if genel_durum == "KRİTİK" else ("#f97316" if genel_durum == "UYARI" else "#22c55e")

    # ── Üst metrik kartları ──
    c0, c1, c2, c3, c4, c5 = st.columns(6)
    kartlar = [
        (c0, f"{son['NPF (%)']:.1f}%",       "Normalized Flow",   "#22c55e" if not uyari_npf else "#ef4444",   f"Eşik: %{esik_npf}"),
        (c1, f"{son['Rreal (%)']:.2f}%",      "Tutuklama (Rreal)", "#22c55e" if not uyari_rreal else "#ef4444", f"Eşik: %{esik_rreal}"),
        (c2, f"{son['ΔP (bar)']:.2f} bar",    "Basınç Farkı ΔP",  "#22c55e" if not uyari_dp else "#ef4444",    f"Eşik: {esik_dp} bar"),
        (c3, f"{son['Enerji (kWh/m³)']:.2f}", "Enerji (kWh/m³)",  "#4a9eca", "—"),
        (c4, f"{son['CP']:.3f}",              "Konsentrasyon Pol.","#4a9eca", "—"),
        (c5, f"{mevcut_gun} gün",             "Simülasyon Günü",   durum_renk, genel_durum),
    ]
    for col, val, lbl, renk, delta in kartlar:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{renk}">{val}</div>
                <div class="metric-label">{lbl}</div>
                <div class="metric-delta" style="color:{renk}88">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Uyarı paneli ──
    st.markdown('<div class="section-title">Erken Uyarı Durumu</div>', unsafe_allow_html=True)
    uw1, uw2, uw3 = st.columns(3)

    def uyari_karti(col, aktif, baslik, aciklama, mudahale):
        cls = "alert-critical" if aktif else "alert-ok"
        ikon = "🔴" if aktif else "🟢"
        with col:
            st.markdown(f"""
            <div class="{cls}">
                <p class="alert-title">{ikon} {baslik}</p>
                <p class="alert-desc">{aciklama}</p>
                <p class="alert-desc" style="color:#6b9ab8;margin-top:6px">
                    {"⚡ " + mudahale if aktif else "✓ Normal aralıkta"}
                </p>
            </div>""", unsafe_allow_html=True)

    uyari_karti(uw1, uyari_npf,
        "NPF Düşüşü",
        f"Mevcut: %{son['NPF (%)']:.1f} | Eşik: %{esik_npf}",
        "Kimyasal yıkama (CIP) önerilir")
    uyari_karti(uw2, uyari_rreal,
        "Tutuklama Kaybı",
        f"Mevcut: %{son['Rreal (%)']:.2f} | Eşik: %{esik_rreal}",
        "Membran bütünlüğü kontrol edilmeli")
    uyari_karti(uw3, uyari_dp,
        "Basınç Artışı",
        f"Mevcut: {son['ΔP (bar)']:.2f} bar | Eşik: {esik_dp} bar",
        "Mekanik kontrol & temizlik")

    st.markdown("")

    # ── Zaman serisi grafikleri ──
    st.markdown('<div class="section-title">Sistem Performansı Zaman Serisi</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#0f1923')
    plt.rcParams.update({'font.family':'DejaVu Sans'})

    parametreler = [
        ('NPF (%)',       '#4a9eca', esik_npf,  'alt', "Normalized Permeate Flow"),
        ('Rreal (%)',     '#22c55e', esik_rreal,'alt', "Tutuklama Oranı"),
        ('ΔP (bar)',      '#f97316', esik_dp,   'üst', "Basınç Farkı"),
        ('Enerji (kWh/m³)','#a78bfa',None,      None,  "Enerji Tüketimi"),
    ]

    for ax, (col, renk, esik, yon, baslik) in zip(axes.flat, parametreler):
        ax.set_facecolor('#131f2e')
        ax.plot(sim_df['Gün'], sim_df[col], color=renk, linewidth=1.8, alpha=0.9)
        ax.fill_between(sim_df['Gün'], sim_df[col], sim_df[col].min(),
                        alpha=0.12, color=renk)
        if esik:
            ax.axhline(esik, color='#ef4444', linestyle='--', linewidth=1.2, alpha=0.7,
                       label=f'Eşik: {esik}')
            ax.legend(fontsize=8, facecolor='#131f2e', labelcolor='white')
            # Eşiği aştığı bölgeleri renklendir
            if yon == 'alt':
                mask = sim_df[col] < esik
            else:
                mask = sim_df[col] > esik
            if mask.any():
                ax.fill_between(sim_df['Gün'], sim_df[col],
                                esik if yon=='alt' else sim_df[col].min(),
                                where=mask, alpha=0.25, color='#ef4444')
        ax.set_title(baslik, color='#8ab4cc', fontsize=10, fontweight='600', pad=8)
        ax.set_xlabel('Gün', color='#4a6b7a', fontsize=8)
        ax.set_ylabel(col, color='#4a6b7a', fontsize=8)
        ax.tick_params(colors='#4a6b7a', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a52')

    plt.tight_layout(pad=2.0)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Fouling türü bilgi kutusu ──
    bilgi = {
        'Partikül Fouling': ('Kil, kum, kolloidal maddeler membrana yapışır.',
                              'NPF hızlı düşer, ΔP ani artar.', 'Fiziksel geri yıkama veya CIP'),
        'Scaling':          ('Kireç, sülfat, silis kristalleri oluşur.',
                              'Yavaş NPF düşüşü, ani bozulma.',  'Antiscalant dozajı, asit yıkama'),
        'Biyofouling':      ('Bakteri ve biyofilm tabakası gelişir.',
                              'Gece/gündüz ΔP farkı, koku.',     'Kloraminle dezenfeksiyon'),
        'Organik Fouling':  ('Humus, yağ, organik moleküller birikim.',
                              'Kademeli NPF düşüşü.',             'Alkali CIP (pH 11-12)'),
    }
    b = bilgi[fouling_tipi]
    st.markdown(f"""
    <div style="background:#131f2e;border:1px solid #1e3a52;border-radius:10px;padding:18px 22px;margin-top:8px">
        <b style="color:#4a9eca">{fouling_tipi} Hakkında</b><br>
        <span style="color:#9ca3af;font-size:13px">⚙️ {b[0]}</span><br>
        <span style="color:#9ca3af;font-size:13px">📉 {b[1]}</span><br>
        <span style="color:#22c55e;font-size:13px">✅ Önerilen müdahale: {b[2]}</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2: ML MODEL
# ══════════════════════════════════════════════
with tab2:
    if not veri_ok:
        st.warning("⚠️ Veri seti yüklenemedi. Lütfen CSV dosyasını sidebar'dan yükleyin.")
    else:
        r2   = r2_score(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)

        # Metrikler
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl, renk in [
            (m1, f"{r2:.4f}",         "R² Skoru",        "#22c55e"),
            (m2, f"{mae:.4f}%",       "Ortalama Hata",    "#4a9eca"),
            (m3, f"{len(df_ml)}",     "Membran Sayısı",   "#a78bfa"),
            (m4, f"{(df_ml['anomali']==-1).sum()}", "Anomali Tespit", "#f97316"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{renk}">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-title">Model Analizi</div>', unsafe_allow_html=True)

        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
        fig2.patch.set_facecolor('#0f1923')

        # Gerçek vs Tahmin
        ax = axes2[0]
        ax.set_facecolor('#131f2e')
        ax.scatter(y_te, y_pred, alpha=0.4, s=18, color='#4a9eca')
        mn, mx = float(y_te.min()), float(y_te.max())
        ax.plot([mn,mx],[mn,mx],'--', color='#ef4444', linewidth=1.5)
        ax.set_xlabel('Gerçek Rreal (%)', color='#6b9ab8', fontsize=9)
        ax.set_ylabel('Tahmin Rreal (%)', color='#6b9ab8', fontsize=9)
        ax.set_title(f'Gerçek vs Tahmin  R²={r2:.4f}', color='#8ab4cc', fontsize=10, fontweight='600')
        ax.tick_params(colors='#4a6b7a', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e3a52')

        # Özellik önemi
        ax = axes2[1]
        ax.set_facecolor('#131f2e')
        onem = pd.Series(model.feature_importances_, index=FEATURES).nlargest(10)
        isimler = {
            'hydraulicP':'Hidrolik Basınç','CP':'Konsentrasyon Pol.','B':'Tuz Geçirgenliği',
            'NaCl conc (M)':'NaCl Kons.','osmoticP':'Osmotik Basınç','A':'Su Geçirgenliği',
            'chemistry_enc':'Kimya','modification_enc':'Modifikasyon','structure_enc':'Yapı',
            'filtrationMode_enc':'Filtrasyon','reportYear':'Yıl','CA':'Temas Açısı',
            'roughness':'Pürüzlülük','thickness':'Kalınlık','pH':'pH','couponSize':'Numune'
        }
        onem.index = [isimler.get(i,i) for i in onem.index]
        renkler_bar = ['#4a9eca','#22c55e','#a78bfa','#f97316','#eab308',
                       '#6b9ab8','#4a9eca','#22c55e','#a78bfa','#f97316']
        ax.barh(range(len(onem)), onem.values, color=renkler_bar, edgecolor='none')
        ax.set_yticks(range(len(onem))); ax.set_yticklabels(onem.index, fontsize=8, color='#8ab4cc')
        ax.set_title('Özellik Önemi (Top 10)', color='#8ab4cc', fontsize=10, fontweight='600')
        ax.set_xlabel('Önem Skoru', color='#6b9ab8', fontsize=8)
        ax.tick_params(colors='#4a6b7a', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e3a52')

        # Anomali dağılımı
        ax = axes2[2]
        ax.set_facecolor('#131f2e')
        normal = df_ml[df_ml['anomali']==1]
        anom   = df_ml[df_ml['anomali']==-1]
        mask_v = (df_ml['A'] < 15) & df_ml['Rreal'].notna()
        ax.scatter(normal.loc[mask_v[normal.index],'A'],
                   normal.loc[mask_v[normal.index],'Rreal'],
                   alpha=0.3, s=12, color='#4a9eca', label='Normal')
        ax.scatter(anom.loc[mask_v[anom.index],'A'],
                   anom.loc[mask_v[anom.index],'Rreal'],
                   alpha=0.8, s=30, color='#ef4444', marker='x',
                   linewidths=1.5, label=f'Anomali (n={len(anom)})')
        ax.set_xlabel('A (L/m²/h/bar)', color='#6b9ab8', fontsize=9)
        ax.set_ylabel('Rreal (%)', color='#6b9ab8', fontsize=9)
        ax.set_title('Anomali Tespiti', color='#8ab4cc', fontsize=10, fontweight='600')
        ax.legend(fontsize=8, facecolor='#131f2e', labelcolor='white')
        ax.tick_params(colors='#4a6b7a', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e3a52')

        plt.tight_layout(pad=2)
        st.pyplot(fig2, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════
# TAB 3: MALİYET ANALİZİ
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Senaryo Bazlı Maliyet Optimizasyonu</div>', unsafe_allow_html=True)

    if not veri_ok:
        st.warning("Veri seti gerekli.")
    else:
        SENARYOLAR = {
            'Fouling':  {'esik': 97.0, 'mal_fp': mal_cip,    'mal_fn': mal_membran + mal_duruş*3},
            'Scaling':  {'esik': 95.0, 'mal_fp': mal_cip//2, 'mal_fn': mal_membran + 45000},
            'Kritik':   {'esik': 90.0, 'mal_fp': mal_duruş,  'mal_fn': mal_membran*1.5 + 30000 + mal_duruş*7},
        }

        esik_aralik = np.linspace(float(y_pred.min()), float(y_pred.max()), 200)

        fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
        fig3.patch.set_facecolor('#0f1923')
        renkler_scen = ['#4a9eca','#f97316','#ef4444']

        cols_mal = st.columns(3)
        for idx, (sadi, scen) in enumerate(SENARYOLAR.items()):
            y_bin = (y_te.values < scen['esik']).astype(int)
            maliyetler = []
            for t in esik_aralik:
                yp = (y_pred < t).astype(int)
                TP = np.sum((yp==1)&(y_bin==1))
                FP = np.sum((yp==1)&(y_bin==0))
                FN = np.sum((yp==0)&(y_bin==1))
                maliyetler.append(FP*scen['mal_fp'] + FN*scen['mal_fn'] + TP*scen['mal_fp'])

            maliyetler = np.array(maliyetler)
            opt_idx  = np.argmin(maliyetler)
            opt_esik = esik_aralik[opt_idx]
            opt_mal  = maliyetler[opt_idx]
            uyarisiz = np.sum(y_bin) * scen['mal_fn']
            tasarruf = max(0, uyarisiz - opt_mal)

            renk = renkler_scen[idx]
            ax = axes3[idx]
            ax.set_facecolor('#131f2e')
            ax.plot(esik_aralik, maliyetler/1000, color=renk, linewidth=2)
            ax.axvline(opt_esik, color='#ef4444', linestyle='--', linewidth=1.2)
            ax.scatter([opt_esik],[opt_mal/1000], color='white', s=60, zorder=5)
            ax.fill_between(esik_aralik, maliyetler/1000, maliyetler.min()/1000,
                            alpha=0.1, color=renk)
            ax.set_title(f'{sadi} Senaryosu', color='#8ab4cc', fontsize=10, fontweight='600')
            ax.set_xlabel('Tahmin Eşiği', color='#6b9ab8', fontsize=8)
            ax.set_ylabel('Maliyet (TL × 1000)', color='#6b9ab8', fontsize=8)
            ax.tick_params(colors='#4a6b7a', labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#1e3a52')

            with cols_mal[idx]:
                st.markdown(f"""
                <div class="metric-card" style="text-align:left">
                    <div style="color:{renk};font-weight:700;font-size:14px;margin-bottom:10px">
                        {sadi} Senaryosu
                    </div>
                    <div style="color:#8ab4cc;font-size:12px;margin:4px 0">
                        📍 Opt. eşik: <b style="color:white">%{opt_esik:.2f}</b>
                    </div>
                    <div style="color:#8ab4cc;font-size:12px;margin:4px 0">
                        💸 Sistem yok: <b style="color:#ef4444">{uyarisiz:,.0f} TL</b>
                    </div>
                    <div style="color:#8ab4cc;font-size:12px;margin:4px 0">
                        ✅ Optimize: <b style="color:#22c55e">{opt_mal:,.0f} TL</b>
                    </div>
                    <div style="color:#8ab4cc;font-size:12px;margin:4px 0">
                        🎯 Tasarruf: <b style="color:#22c55e">%{tasarruf/uyarisiz*100:.1f}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        plt.tight_layout(pad=2)
        st.pyplot(fig3, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════
# TAB 4: TAHMİN ARACI
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Yeni Membran Performans Tahmini</div>', unsafe_allow_html=True)

    if not veri_ok:
        st.warning("Veri seti gerekli.")
    else:
        st.markdown("""
        <div style="background:#131f2e;border:1px solid #1e3a52;border-radius:10px;
                    padding:14px 20px;margin-bottom:16px;font-size:13px;color:#8ab4cc">
            💡 Membranın operasyonel parametrelerini girerek tahmin edilen Rreal ve risk seviyesini öğrenin.
        </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Operasyonel Parametreler**")
            hp   = st.number_input("Hidrolik Basınç (bar)",      0.0, 100.0, 15.5, 0.1)
            cp   = st.number_input("Konsantrasyon Pol. (CP)",    0.1,  2.0,  0.85, 0.01)
            b    = st.number_input("Tuz Geçirgenliği B (L/m²/h)",0.0, 10.0, 0.12, 0.01)
            nacl = st.number_input("NaCl Konsantrasyonu (M)",    0.0,  2.0,  0.034,0.001)
            osm  = st.number_input("Osmotik Basınç (bar)",       0.0, 30.0,  2.7,  0.1)
            a    = st.number_input("Su Geçirgenliği A (L/m²/h/bar)", 0.0, 20.0, 3.2, 0.1)

        with col_b:
            st.markdown("**Fiziksel & Kimyasal Özellikler**")
            yil  = st.number_input("Yayın Yılı",   2000, 2026, 2023, 1)
            ca   = st.number_input("Temas Açısı (°)", 0.0, 180.0, 65.0, 0.5)
            roug = st.number_input("Pürüzlülük (nm)", 0.0, 500.0, 55.0, 1.0)
            thk  = st.number_input("Kalınlık (nm)",   0.0,2000.0,150.0, 5.0)
            ph   = st.number_input("pH",              0.0,  14.0,  7.2, 0.1)
            cs   = st.number_input("Numune Alanı (cm²)", 1.0, 200.0, 14.4, 0.1)

        if st.button("🔮 Tahmin Yap", use_container_width=True):
            yeni = pd.DataFrame([{
                'hydraulicP': hp, 'CP': cp, 'B': b, 'NaCl conc (M)': nacl,
                'osmoticP': osm, 'A': a, 'chemistry_enc': 5,
                'modification_enc': 2, 'structure_enc': 1,
                'filtrationMode_enc': 0, 'reportYear': yil,
                'CA': ca, 'roughness': roug, 'thickness': thk,
                'pH': ph, 'couponSize': cs,
            }])
            yeni_imp = pd.DataFrame(imp.transform(yeni), columns=FEATURES)
            tahmin   = float(model.predict(yeni_imp)[0])

            renk_t = '#22c55e' if tahmin >= 98 else ('#eab308' if tahmin >= 95 else ('#f97316' if tahmin >= 90 else '#ef4444'))
            risk_t = 'İyi' if tahmin >= 98 else ('Dikkat' if tahmin >= 95 else ('Riskli' if tahmin >= 90 else 'Kritik'))

            st.markdown(f"""
            <div style="background:#131f2e;border:2px solid {renk_t};border-radius:12px;
                        padding:28px;text-align:center;margin-top:16px">
                <div style="font-size:14px;color:#6b9ab8;margin-bottom:8px;
                            font-family:'JetBrains Mono'">TAHMİN EDİLEN RREAL</div>
                <div style="font-size:56px;font-weight:700;color:{renk_t};
                            font-family:'JetBrains Mono';line-height:1">%{tahmin:.2f}</div>
                <div style="font-size:18px;color:{renk_t};margin-top:8px;font-weight:600">
                    {risk_t}
                </div>
                <div style="font-size:12px;color:#6b9ab8;margin-top:12px">
                    Fouling eşiği: %{esik_rreal} | 
                    {'⚠️ UYARI GEREKLİ' if tahmin < esik_rreal else '✅ Normal aralıkta'}
                </div>
            </div>""", unsafe_allow_html=True)

