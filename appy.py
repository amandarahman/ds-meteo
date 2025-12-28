import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date

# ======================================================
# 1. KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(page_title="MeteoForecaster Palembang — LSTM", layout="wide")

st.markdown("""
    <style>
    .header-style { text-align: center; color: #0B3C5D; font-size: 38px; font-weight: bold; font-family: 'Times New Roman', serif; }
    .subheader-style { text-align: center; color: #555; font-size: 18px; margin-bottom: 30px; }
    .result-card { background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 10px solid #0B3C5D; color: #0B3C5D; font-size: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ======================================================
# 2. FUNGSI LOAD DATA (DENGAN PENGURUTAN KRONOLOGIS)
# ======================================================
@st.cache_data
def load_all_files():
    try:
        hist = pd.read_csv("data processed_data monthly.csv", index_col=0)
        fore = pd.read_csv("data forecast_peramalan 20 tahun semua parameter.csv", index_col=0)
        
        # Konversi ke Datetime dan SORTING agar tidak NGACAK
        hist.index = pd.to_datetime(hist.index)
        fore.index = pd.to_datetime(fore.index)
        hist = hist.sort_index()
        fore = fore.sort_index()
        
        metr = pd.read_csv("evaluation model_metrics.csv", index_col=0)
        meta = pd.read_csv("metadata_model metadata.csv", header=None, index_col=0)
        act_t = pd.read_csv("data dashboard_data aktual test.csv", index_col=0, parse_dates=True).sort_index()
        pre_t = pd.read_csv("data dashboard_data prediksi test.csv", index_col=0, parse_dates=True).sort_index()
        
        return hist, fore, metr, meta, act_t, pre_t
    except Exception as e:
        return None, None, None, None, None, None

df, future_df, metrics_df, metadata_df, actual_test, pred_test = load_all_files()

if df is None:
    st.error("⚠️ File CSV tidak ditemukan. Pastikan 6 file data berada di folder yang sama.")
    st.stop()

# ======================================================
# 3. HEADER & SIDEBAR
# ======================================================
st.markdown('<div class="header-style">DASHBOARD PERAMALAN IKLIM KOTA PALEMBANG</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-style">Analisis Temporal Jangka Panjang Berbasis LSTM - 2025</div>', unsafe_allow_html=True)

label_map = {
    "TN": "Temperatur Minimum (TN)", "TX": "Temperatur Maksimum (TX)",
    "RH_AVG": "Kelembapan Relatif Rata-rata (RH_AVG)", "RR": "Curah Hujan (RR)",
    "SS": "Lama Penyinaran Matahari (SS)", "FF_X": "Kecepatan Angin Maksimum (FF_X)",
    "FF_AVG": "Kecepatan Angin Rata-rata (FF_AVG)", "DDD_X_sin": "Komponen Arah Angin Maksimum (DDD_X_sin)"
}

st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Visualisasi & Prediksi", "Uji Validitas", "Profil Peneliti"])
var_name = st.sidebar.selectbox("Pilih Parameter:", list(label_map.keys()), format_func=lambda x: label_map[x])

# ======================================================
# 4. HALAMAN 1: VISUALISASI & PREDISI (PAKAI KALENDER)
# ======================================================
if menu == "Visualisasi & Prediksi":
    # METRIK AKURASI
    st.subheader(f"📊 Metrik Akurasi Model: {label_map[var_name]}")
    c1, c2, c3 = st.columns(3)
    if var_name in metrics_df.index:
        c1.metric("RMSE", f"{metrics_df.loc[var_name, 'RMSE']:.4f}")
        c2.metric("MAE", f"{metrics_df.loc[var_name, 'MAE']:.4f}")
        # Menggunakan LaTeX untuk R-Squared
        c3.metric("R-Squared ($R^2$)", f"{metrics_df.loc[var_name, 'R2']:.4f}")
    
    st.divider()

    # FITUR PENCARIAN TANGGAL (KALENDER)
    st.subheader("📅 Cari Hasil Prediksi Berdasarkan Tanggal")
    
    # Input Kalender
    selected_date = st.date_input("Pilih Tanggal:", 
                                   value=date(2025, 1, 1),
                                   min_value=date(2025, 1, 1), 
                                   max_value=date(2044, 12, 31))
    
    # LOGIKA PENGUNCI DATA: 
    # Karena data Anda bulanan, kita paksa mencari tanggal 1 di bulan yang dipilih user
    lookup_date = selected_date.replace(day=1)
    
    try:
        # Mencari nilai pada index yang sudah di-konversi ke datetime
        # Penggunaan .loc dengan format YYYY-MM-DD sangat stabil
        val = future_df.loc[lookup_date.strftime('%Y-%m-%d'), var_name]
        
        st.markdown(f"""
            <div class="result-card">
                Hasil Prediksi {label_map[var_name]}<br>
                Periode: {selected_date.strftime("%B %Y")}<br>
                Nilai: {val:.2f}
            </div>
            """, unsafe_allow_html=True)
    except:
        st.warning(f"Data untuk bulan {selected_date.strftime('%B %Y')} tidak ditemukan. Periksa kembali rentang tahun di file CSV Anda.")

    st.divider()

    # GRAFIK (BIRU & KUNING)
    st.subheader("📈 Tren Historis & Proyeksi 20 Tahun")
    combined_plot = pd.concat([
        df[[var_name]].assign(Kategori="Historis"),
        future_df[[var_name]].assign(Kategori="Prediksi")
    ]).sort_index()
    
    fig = px.line(combined_plot, x=combined_plot.index, y=var_name, color="Kategori",
                  color_discrete_map={"Historis": "#0B3C5D", "Prediksi": "#F2C94C"},
                  template="plotly_white")
    
    # Hover detail agar tanggal terlihat rapi (Hari Bulan Tahun)
    fig.update_traces(hovertemplate="<b>Tanggal:</b> %{x|%d %B %Y}<br><b>Nilai:</b> %{y}")
    fig.update_layout(hovermode="x unified", xaxis_title="Waktu (Tahun)", yaxis_title=label_map[var_name])
    
    
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 5. HALAMAN 2: UJI VALIDITAS (RESIDUAL)
# ======================================================
elif menu == "Uji Validitas":
    st.header(f"🎯 Analisis Residual: {label_map[var_name]}")
    residual_error = actual_test[var_name] - pred_test[var_name]
    
    col_a, col_b = st.columns(2)
    with col_a:
        f1 = px.scatter(x=pred_test[var_name], y=residual_error, 
                         labels={'x':'Nilai Prediksi', 'y':'Error (Residual)'},
                         title="Residual Plot (Scatter)", color_discrete_sequence=['#0B3C5D'])
        f1.add_hline(y=0, line_dash="dash", line_color="#F2C94C")
        st.plotly_chart(f1, use_container_width=True)
    with col_b:
        f2 = px.histogram(residual_error, nbins=20, title="Distribusi Error", 
                          color_discrete_sequence=['#F2C94C'])
        st.plotly_chart(f2, use_container_width=True)

# ======================================================
# 6. HALAMAN 3: PROFIL PENELITI
# ======================================================
else:
    st.header("👤 Profil Peneliti & Akademik")
    st.info(f"**Nama Peneliti:** Amanda Rahmannisa\n\n**NIM:** 06111282227058")
    st.warning(f"**Dosen Pembimbing:**\n\nDr. Melly Ariska, S.Pd., M.Sc.")
    st.success(f"**Informasi Akademik:**\n* **Program Studi:** Pendidikan Fisika\n* **Fakultas:** Keguruan dan Ilmu Pendidikan\n* **Universitas:** Universitas Sriwijaya\n* **Tahun:** 2025")
    st.divider()
    st.subheader("🛠️ Metadata Konfigurasi Model")
    st.table(metadata_df)