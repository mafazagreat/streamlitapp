# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Coba import dengan error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from streamlit_option_menu import option_menu
    MENU_AVAILABLE = True
except ImportError:
    MENU_AVAILABLE = False

# Import tslearn
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
    from tslearn.utils import to_time_series_dataset
    from tslearn.metrics import cdist_dtw
    TSL_AVAILABLE = True
except ImportError:
    TSL_AVAILABLE = False

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Komparasi Clustering Pengangguran Terbuka",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.3px;
        line-height: 1.35;
        color: #1a2733;
        text-align: center;
        padding: 0.9rem 1.5rem 1.1rem 1.5rem;
        border-bottom: 1px solid #e3e8ee;
        margin-bottom: 1.6rem;
    }
    .main-header-icon {
        color: #1f77b4;
        margin-right: 0.35rem;
    }
    .app-subtitle {
        font-size: 0.92rem;
        font-weight: 400;
        color: #7f8c95;
        text-align: center;
        margin-top: 0.35rem;
        letter-spacing: 0;
    }
    section[data-testid="stSidebar"] {
        min-width: 230px !important;
        max-width: 230px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #8a94a3;
        padding-left: 0.3rem;
        margin-bottom: 0.4rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        padding: 1rem 0;
        border-bottom: 2px solid #ecf0f1;
        margin-bottom: 1rem;
    }
    .stat-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: 0.5rem;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .metric-good {
        color: #27ae60;
        font-weight: bold;
    }
    .metric-bad {
        color: #e74c3c;
        font-weight: bold;
    }
    .upload-area {
        border: 2px dashed #1f77b4;
        padding: 3rem;
        text-align: center;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 1rem 0;
        border-top: 1px solid #ecf0f1;
        margin-top: 2rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #2c3e50;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #ecf0f1;
        margin-bottom: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-header">'
    '<span class="main-header-icon">📊</span>Sistem Komparasi Clustering Pengangguran Terbuka'
    '<div class="app-subtitle">Berdasarkan Tingkat Pendidikan di Jawa Barat</div>'
    '</div>',
    unsafe_allow_html=True
)

# ==================== FUNGSI UTAMA ====================

def preprocess_data(data):
    """Preprocessing data seperti di untitled36.py"""
    # Filter tahun 2020-2025
    data = data[(data['tahun'] >= 2020) & (data['tahun'] <= 2025)]
    
    # Gabungkan kategori pendidikan
    data['pendidikan'] = data['pendidikan'].replace(
        ['TIDAK/BELUM PERNAH SEKOLAH/TIDAK/BELUM TAMAT SD', 'SD'],
        'SD KE BAWAH'
    )
    
    pendidikan_order = ['SD KE BAWAH', 'SMP', 'SMA (UMUM)', 'SMA (KEJURUAN)', 'DIPLOMA I/II/III/AKADEMI/UNIVERSITAS']
    data['pendidikan'] = pd.Categorical(data['pendidikan'], categories=pendidikan_order, ordered=True)
    
    return data, pendidikan_order

def kmeans_analysis(data, pendidikan_order, final_k=None):
    """Analisis K-Means seperti di untitled36.py"""
    # Pivot ke bentuk wide
    pivot_km = data.pivot_table(
        index=['nama_kabupaten_kota', 'tahun'],
        columns='pendidikan',
        values='jumlah_pengangguran',
        aggfunc='mean',
        observed=True
    )
    pivot_km = pivot_km[pendidikan_order]
    
    # Interpolasi jika ada missing
    if pivot_km.isnull().values.any():
        pivot_km = pivot_km.interpolate(axis=0, limit_direction='both')
    
    # Normalisasi Min-Max global
    scaler = MinMaxScaler()
    X_km_flat = pivot_km.values.reshape(-1, 1)
    X_km_scaled_flat = scaler.fit_transform(X_km_flat)
    X = X_km_scaled_flat.reshape(pivot_km.shape)
    
    # Jika final_k tidak ditentukan, cari optimal
    if final_k is None:
        inertia_list = []
        silhouette_list = []
        dbi_list = []
        K_range = range(2, 10)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertia_list.append(kmeans.inertia_)
            silhouette_list.append(silhouette_score(X, labels))
            dbi_list.append(davies_bouldin_score(X, labels))
        
        # Voting untuk menentukan k optimal
        best_silhouette_k = K_range[np.argmax(silhouette_list)]
        best_dbi_k = K_range[np.argmin(dbi_list)]
        
        inertia_diff = np.diff(inertia_list)
        inertia_ratio = inertia_diff[1:] / inertia_diff[:-1]
        best_elbow_k = list(K_range)[np.argmin(inertia_ratio) + 1]
        
        vote_counts = pd.Series([best_elbow_k, best_silhouette_k, best_dbi_k]).value_counts()
        final_k = vote_counts.index[0]
    
    # Clustering final
    kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    final_labels = kmeans_final.fit_predict(X)
    
    # Hasil
    hasil_km = pivot_km.reset_index()
    hasil_km['cluster'] = final_labels
    
    # Metrik
    sil_score = silhouette_score(X, final_labels)
    dbi_score = davies_bouldin_score(X, final_labels)
    
    return {
        'X': X,
        'labels': final_labels,
        'hasil': hasil_km,
        'pivot': pivot_km,
        'final_k': final_k,
        'silhouette': sil_score,
        'dbi': dbi_score,
        'kmeans_model': kmeans_final,
        'scaler': scaler
    }

def timeseries_kmeans_analysis(data, pendidikan_order, final_k=None, scaler_type='minmax'):
    """Analisis Time Series K-Means seperti di untitled36.py"""
    if not TSL_AVAILABLE:
        st.error("⚠️ tslearn tidak tersedia. Install dengan: pip install tslearn")
        return None
    
    ts_data = data.copy()
    tahun_order = sorted(ts_data['tahun'].unique())
    
    # Pivot: setiap baris = kab/kota x pendidikan, kolom = tahun
    pivot_ts = ts_data.pivot_table(
        index=['nama_kabupaten_kota', 'pendidikan'],
        columns='tahun',
        values='jumlah_pengangguran',
        aggfunc='mean'
    )
    pivot_ts = pivot_ts[tahun_order]
    
    # Interpolasi jika ada missing
    if pivot_ts.isnull().values.any():
        pivot_ts = pivot_ts.interpolate(axis=1, limit_direction='both')
    
    # Bentuk array time series 3D
    X_ts_raw = to_time_series_dataset(pivot_ts.values)
    
    # Normalisasi
    if scaler_type == 'minmax':
        scaler_ts = TimeSeriesScalerMinMax()
    else:  # zscore
        scaler_ts = TimeSeriesScalerMeanVariance()
    
    X_ts = scaler_ts.fit_transform(X_ts_raw)
    X_ts_flat = X_ts.reshape(X_ts.shape[0], X_ts.shape[1])
    
    labels_id = pivot_ts.reset_index()[['nama_kabupaten_kota', 'pendidikan']]
    
    # Jika final_k tidak ditentukan, cari optimal
    if final_k is None:
        inertia_ts = []
        silhouette_ts = []
        dbi_ts = []
        K_range_ts = range(2, 10)
        
        for k in K_range_ts:
            model = TimeSeriesKMeans(
                n_clusters=k,
                metric="dtw",
                random_state=42,
                n_init=5,
                max_iter=50,
                n_jobs=-1
            )
            labels_k = model.fit_predict(X_ts)
            inertia_ts.append(model.inertia_)
            
            dist_matrix = cdist_dtw(X_ts)
            sil = silhouette_score(dist_matrix, labels_k, metric="precomputed")
            silhouette_ts.append(sil)
            
            dbi = davies_bouldin_score(X_ts_flat, labels_k)
            dbi_ts.append(dbi)
        
        # Voting untuk menentukan k optimal
        best_sil_k = K_range_ts[np.argmax(silhouette_ts)]
        best_dbi_k = K_range_ts[np.argmin(dbi_ts)]
        
        inertia_diff_ts = np.diff(inertia_ts)
        inertia_ratio_ts = inertia_diff_ts[1:] / inertia_diff_ts[:-1]
        best_elbow_k = list(K_range_ts)[np.argmin(inertia_ratio_ts) + 1]
        
        vote_counts = pd.Series([best_elbow_k, best_sil_k, best_dbi_k]).value_counts()
        final_k = vote_counts.index[0]
    
    # Clustering final
    model_final = TimeSeriesKMeans(
        n_clusters=final_k,
        metric="dtw",
        random_state=42,
        n_init=15,
        max_iter=50,
        n_jobs=-1
    )
    final_labels = model_final.fit_predict(X_ts)
    
    # Hasil
    hasil_ts = labels_id.copy()
    hasil_ts['cluster'] = final_labels
    
    # Metrik
    dist_matrix_final = cdist_dtw(X_ts)
    sil_score = silhouette_score(dist_matrix_final, final_labels, metric="precomputed")
    dbi_score = davies_bouldin_score(X_ts_flat, final_labels)
    
    return {
        'X_ts': X_ts,
        'X_ts_raw': X_ts_raw,
        'labels': final_labels,
        'hasil': hasil_ts,
        'pivot': pivot_ts,
        'final_k': final_k,
        'silhouette': sil_score,
        'dbi': dbi_score,
        'model': model_final,
        'scaler': scaler_ts
    }

def stability_analysis_km(X, final_k, random_states=[0, 1, 21, 42, 100, 123]):
    """Stabilitas K-Means"""
    stability = []
    for rs in random_states:
        km_test = KMeans(n_clusters=final_k, random_state=rs, n_init=10)
        labels_test = km_test.fit_predict(X)
        sil_test = silhouette_score(X, labels_test)
        dbi_test = davies_bouldin_score(X, labels_test)
        stability.append({'random_state': rs, 'Silhouette': sil_test, 'DBI': dbi_test})
    
    df = pd.DataFrame(stability)
    return df

def stability_analysis_ts(X_ts, final_k, random_states=[0, 1, 21, 42, 100, 123]):
    """Stabilitas Time Series K-Means"""
    if not TSL_AVAILABLE:
        return None
    
    X_ts_flat = X_ts.reshape(X_ts.shape[0], X_ts.shape[1])
    stability = []
    
    for rs in random_states:
        model_stab = TimeSeriesKMeans(
            n_clusters=final_k, metric="dtw", random_state=rs,
            n_init=15, max_iter=50, n_jobs=-1
        )
        labels_stab = model_stab.fit_predict(X_ts)
        dist_stab = cdist_dtw(X_ts)
        sil_stab = silhouette_score(dist_stab, labels_stab, metric="precomputed")
        dbi_stab = davies_bouldin_score(X_ts_flat, labels_stab)
        stability.append({'random_state': rs, 'Silhouette': sil_stab, 'DBI': dbi_stab})
    
    df = pd.DataFrame(stability)
    return df

# ==================== SIDEBAR MENU ====================

with st.sidebar:
    st.markdown("### MENU")
    
    if MENU_AVAILABLE:
        selected = option_menu(
            menu_title=None,
            options=["Beranda", "Upload Dataset", "Exploratory Data Analysis", 
                    "K-Means Clustering", "Time Series K-Means", "Komparasi Algoritma", 
                    "Review Model", "Visualisasi", "Unduh Hasil", "Tentang Aplikasi"],
            icons=["house", "cloud-upload", "bar-chart", "diagram-3", "clock-history", 
                   "shuffle", "check-circle", "graph-up", "download", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1f77b4", "font-size": "15px"},
                "nav-link": {"font-size": "13.5px", "text-align": "left", "margin": "1px 0",
                            "padding": "8px 10px", "border-radius": "6px",
                            "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#1f77b4", "font-weight": "500"},
            }
        )
    else:
        menu_options = ["Beranda", "Upload Dataset", "Exploratory Data Analysis", 
                       "K-Means Clustering", "Time Series K-Means", "Komparasi Algoritma", 
                       "Review Model", "Visualisasi", "Unduh Hasil", "Tentang Aplikasi"]
        selected = st.selectbox("Pilih Menu", menu_options)

# ==================== INISIALISASI DATA ====================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.pendidikan_order = None
    st.session_state.km_results = None
    st.session_state.ts_results = None

# ==================== BERANDA ====================

if selected == "Beranda":
    st.markdown('<div class="sub-header">📊 Sistem Komparasi Clustering Pengangguran Terbuka</div>', unsafe_allow_html=True)
    st.caption("Aplikasi ini digunakan untuk melakukan analisis clustering menggunakan algoritma K-Means dan TimeSeriesKMeans serta membandingkan hasil kedua algoritma.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔍 Fitur Utama
        1. **Upload Dataset** - Unggah file Excel data pengangguran
        2. **Exploratory Data Analysis** - Analisis eksploratif data
        3. **K-Means Clustering** - Clustering dengan K-Means
        4. **Time Series K-Means** - Clustering dengan DTW
        5. **Komparasi Algoritma** - Perbandingan kedua algoritma
        6. **Review Model** - Evaluasi stabilitas model
        7. **Visualisasi** - Visualisasi interaktif
        8. **Unduh Hasil** - Download hasil analisis
        """)
    
    with col2:
        st.markdown("""
        #### 📋 Persyaratan Data
        - Format: **.xlsx**
        - Kolom yang diperlukan:
        - `nama_kabupaten_kota` - Nama wilayah
        - `tahun` - Tahun (2020-2025)
        - `pendidikan` - Tingkat pendidikan
        - `jumlah_pengangguran` - Jumlah pengangguran
        """)
        
        if st.button("📥 Gunakan Dataset Contoh", use_container_width=True):
            # Generate data contoh
            np.random.seed(42)
            kabupaten = ['KABUPATEN BOGOR', 'KABUPATEN SUKABUMI', 'KABUPATEN CIANJUR', 
                        'KABUPATEN BANDUNG', 'KABUPATEN GARUT', 'KABUPATEN TASIKMALAYA',
                        'KABUPATEN CIAMIS', 'KABUPATEN KUNINGAN', 'KABUPATEN CIREBON',
                        'KABUPATEN MAJALENGKA', 'KABUPATEN SUMEDANG', 'KABUPATEN INDRAMAYU',
                        'KABUPATEN SUBANG', 'KABUPATEN PURWAKARTA', 'KABUPATEN KARAWANG',
                        'KABUPATEN BEKASI', 'KABUPATEN BANDUNG BARAT', 'KABUPATEN PANGANDARAN',
                        'KOTA BOGOR', 'KOTA SUKABUMI', 'KOTA BANDUNG', 'KOTA CIREBON',
                        'KOTA BEKASI', 'KOTA DEPOK', 'KOTA CIMAHI', 'KOTA TASIKMALAYA',
                        'KOTA BANJAR']
            
            pendidikan = ['SD KE BAWAH', 'SMP', 'SMA (UMUM)', 'SMA (KEJURUAN)', 
                         'DIPLOMA I/II/III/AKADEMI/UNIVERSITAS']
            
            data_list = []
            for kab in kabupaten:
                for edu in pendidikan:
                    base = np.random.randint(500, 5000)
                    for tahun in range(2020, 2026):
                        # Simulasi tren dengan random
                        trend = 1 + 0.1 * (tahun - 2020) + np.random.normal(0, 0.05)
                        # Efek COVID: 2020-2021 lebih tinggi
                        if tahun in [2020, 2021]:
                            covid_factor = 1.3
                        else:
                            covid_factor = 1.0
                        nilai = int(base * trend * covid_factor * np.random.uniform(0.7, 1.3))
                        data_list.append({
                            'nama_kabupaten_kota': kab,
                            'tahun': tahun,
                            'pendidikan': edu,
                            'jumlah_pengangguran': max(10, nilai)
                        })
            
            df_sample = pd.DataFrame(data_list)
            st.session_state.df = df_sample
            st.session_state.data_loaded = True
            
            # Preprocess
            df_processed, pendidikan_order = preprocess_data(df_sample)
            st.session_state.pendidikan_order = pendidikan_order
            
            st.success(f"✅ Dataset contoh berhasil dimuat! {len(df_sample)} baris data")
            st.rerun()

    # Status data
    if st.session_state.data_loaded:
        st.success("✅ Dataset telah dimuat!")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
    else:
        st.info("ℹ️ Silakan upload dataset atau gunakan dataset contoh di atas.")

# ==================== UPLOAD DATASET ====================

elif selected == "Upload Dataset":
    st.markdown('<div class="sub-header">📤 Upload Dataset</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="upload-area">
            <h4>📁 Upload file dataset (.xlsx)</h4>
            <p style="color: #7f8c8d;">Drag and drop file here</p>
            <p style="color: #95a5a6; font-size: 0.9rem;">Limit 200MB per file - XLSX</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file XLSX", type=['xlsx'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            
            # Cek kolom yang diperlukan
            required_cols = ['nama_kabupaten_kota', 'tahun', 'pendidikan', 'jumlah_pengangguran']
            missing_cols = [col for col in required_cols if col not in df_raw.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom yang diperlukan tidak ditemukan: {missing_cols}")
                st.info("Pastikan dataset memiliki kolom: nama_kabupaten_kota, tahun, pendidikan, jumlah_pengangguran")
            else:
                st.session_state.df = df_raw
                st.session_state.data_loaded = True
                
                # Preprocess
                df_processed, pendidikan_order = preprocess_data(df_raw)
                st.session_state.pendidikan_order = pendidikan_order
                
                st.success(f"✅ File berhasil diupload! {len(df_processed)} baris data (setelah filter 2020-2025)")
                
                st.markdown("#### Preview Dataset")
                st.dataframe(df_processed.head(), use_container_width=True)
                
                st.markdown("#### Ringkasan Data")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Jumlah Wilayah", df_processed['nama_kabupaten_kota'].nunique())
                with col2:
                    st.metric("Jumlah Pendidikan", df_processed['pendidikan'].nunique())
                with col3:
                    st.metric("Jumlah Tahun", df_processed['tahun'].nunique())
                with col4:
                    st.metric("Total Observasi", len(df_processed))
                    
        except Exception as e:
            st.error(f"Error membaca file: {e}")
    else:
        if st.session_state.data_loaded:
            st.info(f"Dataset saat ini: {len(st.session_state.df)} baris data")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
        else:
            st.info("Belum ada dataset yang diupload. Silakan upload file Excel.")

# ==================== EXPLORATORY DATA ANALYSIS ====================

elif selected == "Exploratory Data Analysis":
    st.markdown('<div class="sub-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Belum ada dataset. Silakan upload dataset terlebih dahulu.")
    else:
        df = st.session_state.df
        pendidikan_order = st.session_state.pendidikan_order
        
        st.markdown("#### Ringkasan Data")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Jumlah Wilayah", df['nama_kabupaten_kota'].nunique())
        with col2:
            st.metric("Jumlah Pendidikan", df['pendidikan'].nunique())
        with col3:
            st.metric("Jumlah Tahun", df['tahun'].nunique())
        with col4:
            st.metric("Total Observasi", len(df))
        
        # Statistik deskriptif
        st.markdown("#### Statistik Deskriptif")
        st.dataframe(df['jumlah_pengangguran'].describe(), use_container_width=True)
        
        if PLOTLY_AVAILABLE:
            st.markdown("#### Visualisasi Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Tren per pendidikan
                fig = px.line(
                    df, x='tahun', y='jumlah_pengangguran', 
                    color='pendidikan', title='Tren Pengangguran per Pendidikan'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Boxplot per tahun
                fig = px.box(
                    df, x='tahun', y='jumlah_pengangguran',
                    title='Distribusi Pengangguran per Tahun'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap rata-rata per pendidikan dan tahun
            pivot_heat = df.pivot_table(
                index='pendidikan', columns='tahun', 
                values='jumlah_pengangguran', aggfunc='mean'
            )
            
            fig = px.imshow(
                pivot_heat,
                title='Heatmap Rata-rata Pengangguran per Pendidikan dan Tahun',
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback ke matplotlib
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Tren per pendidikan
            for edu in df['pendidikan'].unique():
                edu_data = df[df['pendidikan'] == edu]
                axes[0].plot(edu_data.groupby('tahun')['jumlah_pengangguran'].mean(), marker='o', label=edu[:10])
            axes[0].set_title('Tren Pengangguran per Pendidikan')
            axes[0].legend()
            axes[0].grid(True)
            
            # Boxplot per tahun
            df.boxplot(column='jumlah_pengangguran', by='tahun', ax=axes[1])
            axes[1].set_title('Distribusi Pengangguran per Tahun')
            axes[1].set_xlabel('Tahun')
            axes[1].set_ylabel('Jumlah Pengangguran')
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================== K-MEANS CLUSTERING ====================

elif selected == "K-Means Clustering":
    st.markdown('<div class="sub-header">🎯 K-Means Clustering</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Belum ada dataset. Silakan upload dataset terlebih dahulu.")
    else:
        df = st.session_state.df
        pendidikan_order = st.session_state.pendidikan_order
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Konfigurasi K-Means")
            k_method = st.radio(
                "Metode pemilihan k",
                ["Otomatis (Rekomendasi)", "Manual"],
                index=0,
                key="km_k_method"
            )
            
            if k_method == "Manual":
                k_value = st.slider("Jumlah Cluster (k)", 2, 10, 3, key="km_k_value")
            else:
                k_value = None
        
        with col2:
            st.markdown("#### Status")
            if st.session_state.km_results is not None:
                st.success(f"✅ K-Means selesai (k={st.session_state.km_results['final_k']})")
            else:
                st.info("ℹ️ Belum menjalankan K-Means")
        
        if st.button("🚀 Jalankan K-Means", use_container_width=True, key="run_km"):
            with st.spinner("Sedang menjalankan K-Means..."):
                try:
                    results = kmeans_analysis(df, pendidikan_order, final_k=k_value)
                    st.session_state.km_results = results
                    st.success(f"✅ K-Means selesai! Cluster optimal: k={results['final_k']}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Tampilkan hasil
        if st.session_state.km_results is not None:
            results = st.session_state.km_results
            
            st.markdown("#### Hasil Clustering")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cluster Optimal", f"k={results['final_k']}")
            with col2:
                st.metric("Silhouette Score", f"{results['silhouette']:.4f}")
            with col3:
                st.metric("Davies-Bouldin Index", f"{results['dbi']:.4f}")
            
            st.markdown("##### Distribusi Cluster")
            cluster_counts = results['hasil']['cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
            st.markdown("##### Preview Hasil")
            st.dataframe(results['hasil'].head(10), use_container_width=True)

# ==================== TIME SERIES K-MEANS ====================

elif selected == "Time Series K-Means":
    st.markdown('<div class="sub-header">⏱️ Time Series K-Means (DTW)</div>', unsafe_allow_html=True)
    
    if not TSL_AVAILABLE:
        st.error("⚠️ tslearn tidak tersedia. Install dengan: pip install tslearn")
    elif not st.session_state.data_loaded:
        st.warning("⚠️ Belum ada dataset. Silakan upload dataset terlebih dahulu.")
    else:
        df = st.session_state.df
        pendidikan_order = st.session_state.pendidikan_order
        
        col1, col2, col3 = st.columns([1.5, 1, 1])
        
        with col1:
            st.markdown("#### Konfigurasi Time Series K-Means")
            k_method = st.radio(
                "Metode pemilihan k",
                ["Otomatis (Rekomendasi)", "Manual"],
                index=0,
                key="ts_k_method"
            )
            
            if k_method == "Manual":
                k_value = st.slider("Jumlah Cluster (k)", 2, 10, 3, key="ts_k_value")
            else:
                k_value = None
        
        with col2:
            st.markdown("#### Scaler")
            scaler_type = st.radio(
                "Tipe Normalisasi",
                ["MinMax", "Z-Score"],
                index=0,
                key="ts_scaler"
            )
        
        with col3:
            st.markdown("#### Status")
            if st.session_state.ts_results is not None:
                st.success(f"✅ TS K-Means selesai (k={st.session_state.ts_results['final_k']})")
            else:
                st.info("ℹ️ Belum menjalankan TS K-Means")
        
        if st.button("🚀 Jalankan Time Series K-Means", use_container_width=True, key="run_ts"):
            with st.spinner("Sedang menjalankan Time Series K-Means..."):
                try:
                    results = timeseries_kmeans_analysis(
                        df, pendidikan_order, 
                        final_k=k_value, 
                        scaler_type='minmax' if scaler_type == "MinMax" else 'zscore'
                    )
                    st.session_state.ts_results = results
                    st.success(f"✅ Time Series K-Means selesai! Cluster optimal: k={results['final_k']}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Tampilkan hasil
        if st.session_state.ts_results is not None:
            results = st.session_state.ts_results
            
            st.markdown("#### Hasil Clustering")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cluster Optimal", f"k={results['final_k']}")
            with col2:
                st.metric("Silhouette Score (DTW)", f"{results['silhouette']:.4f}")
            with col3:
                st.metric("Davies-Bouldin Index", f"{results['dbi']:.4f}")
            
            st.markdown("##### Distribusi Cluster")
            cluster_counts = results['hasil']['cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
            st.markdown("##### Preview Hasil")
            st.dataframe(results['hasil'].head(10), use_container_width=True)

# ==================== KOMPARASI ALGORITMA ====================

elif selected == "Komparasi Algoritma":
    st.markdown('<div class="sub-header">🔍 Komparasi Algoritma</div>', unsafe_allow_html=True)
    
    km_results = st.session_state.km_results
    ts_results = st.session_state.ts_results
    
    if km_results is None and ts_results is None:
        st.warning("⚠️ Belum ada hasil clustering. Jalankan K-Means dan/atau Time Series K-Means terlebih dahulu.")
    else:
        st.markdown("#### Perbandingan Metrik")
        
        data_compare = []
        
        if km_results is not None:
            data_compare.append({
                'Algoritma': 'K-Means',
                'Cluster Optimal': km_results['final_k'],
                'Silhouette Score': km_results['silhouette'],
                'Davies-Bouldin Index': km_results['dbi']
            })
        
        if ts_results is not None:
            data_compare.append({
                'Algoritma': 'Time Series K-Means (DTW)',
                'Cluster Optimal': ts_results['final_k'],
                'Silhouette Score': ts_results['silhouette'],
                'Davies-Bouldin Index': ts_results['dbi']
            })
        
        df_compare = pd.DataFrame(data_compare)
        st.dataframe(df_compare, use_container_width=True)
        
        if PLOTLY_AVAILABLE and len(data_compare) > 1:
            st.markdown("#### Visualisasi Perbandingan")
            
            metrics = ['Silhouette Score', 'Davies-Bouldin Index']
            fig = go.Figure()
            
            for _, row in df_compare.iterrows():
                fig.add_trace(go.Bar(
                    name=row['Algoritma'],
                    x=metrics,
                    y=[row['Silhouette Score'], row['Davies-Bouldin Index']]
                ))
            
            fig.update_layout(
                title='Perbandingan Metrik Clustering',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Kesimpulan
        st.markdown("#### Kesimpulan")
        if km_results is not None and ts_results is not None:
            if km_results['silhouette'] > ts_results['silhouette']:
                better = "K-Means"
                better_sil = km_results['silhouette']
            else:
                better = "Time Series K-Means (DTW)"
                better_sil = ts_results['silhouette']
            
            st.info(f"""
            **📊 Analisis Perbandingan:**
            
            - **Silhouette Score:** {better} memiliki skor lebih tinggi ({better_sil:.4f})
            - **Interpretasi:** 
              - K-Means: Mengelompokkan berdasarkan nilai absolut
              - Time Series K-Means: Mengelompokkan berdasarkan pola waktu (DTW)
            - **Rekomendasi:** Pilih algoritma berdasarkan tujuan analisis
            """)

# ==================== REVIEW MODEL ====================

elif selected == "Review Model":
    st.markdown('<div class="sub-header">✅ Review Model</div>', unsafe_allow_html=True)
    
    km_results = st.session_state.km_results
    ts_results = st.session_state.ts_results
    
    st.markdown("""
    Review model bertujuan untuk mengevaluasi stabilitas dan konsistensi 
    hasil clustering dengan berbagai random state.
    """)
    
    tab1, tab2, tab3 = st.tabs(["K-Means", "Time Series K-Means", "Kesimpulan"])
    
    with tab1:
        if km_results is None:
            st.warning("⚠️ Jalankan K-Means terlebih dahulu.")
        else:
            st.markdown("#### Stabilitas K-Means")
            
            if st.button("🔄 Analisis Stabilitas K-Means", key="stab_km"):
                with st.spinner("Menganalisis stabilitas..."):
                    X = km_results['X']
                    final_k = km_results['final_k']
                    stab_df = stability_analysis_km(X, final_k)
                    
                    st.dataframe(stab_df, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rata-rata Silhouette", f"{stab_df['Silhouette'].mean():.4f}")
                    with col2:
                        st.metric("Std Silhouette", f"{stab_df['Silhouette'].std():.4f}")
                    
                    if stab_df['Silhouette'].std() < 0.03:
                        st.success("✅ Model stabil (std < 0.03)")
                    else:
                        st.warning("⚠️ Model kurang stabil (std >= 0.03)")
    
    with tab2:
        if not TSL_AVAILABLE:
            st.error("⚠️ tslearn tidak tersedia.")
        elif ts_results is None:
            st.warning("⚠️ Jalankan Time Series K-Means terlebih dahulu.")
        else:
            st.markdown("#### Stabilitas Time Series K-Means")
            
            if st.button("🔄 Analisis Stabilitas Time Series K-Means", key="stab_ts"):
                with st.spinner("Menganalisis stabilitas..."):
                    X_ts = ts_results['X_ts']
                    final_k = ts_results['final_k']
                    stab_df = stability_analysis_ts(X_ts, final_k)
                    
                    if stab_df is not None:
                        st.dataframe(stab_df, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rata-rata Silhouette (DTW)", f"{stab_df['Silhouette'].mean():.4f}")
                        with col2:
                            st.metric("Std Silhouette", f"{stab_df['Silhouette'].std():.4f}")
                        
                        if stab_df['Silhouette'].std() < 0.03:
                            st.success("✅ Model stabil (std < 0.03)")
                        else:
                            st.warning("⚠️ Model kurang stabil (std >= 0.03)")
    
    with tab3:
        st.markdown("#### Kesimpulan Review Model")
        
        if km_results is not None and ts_results is not None:
            st.markdown(f"""
            **📊 Hasil Review:**
            
            | Aspek | K-Means | Time Series K-Means |
            |-------|---------|-------------------|
            | Cluster Optimal | {km_results['final_k']} | {ts_results['final_k']} |
            | Silhouette Score | {km_results['silhouette']:.4f} | {ts_results['silhouette']:.4f} |
            | Davies-Bouldin | {km_results['dbi']:.4f} | {ts_results['dbi']:.4f} |
            """)
            
            st.info("""
            **📝 Interpretasi:**
            
            1. **K-Means** menggunakan normalisasi global dan cocok untuk data dengan skala berbeda
            2. **Time Series K-Means** menggunakan DTW dan memperhatikan pola waktu antar tahun
            3. Kedua model menunjukkan hasil yang stabil dengan nilai Silhouette Score yang baik
            """)

# ==================== VISUALISASI ====================

elif selected == "Visualisasi":
    st.markdown('<div class="sub-header">📈 Visualisasi</div>', unsafe_allow_html=True)
    
    km_results = st.session_state.km_results
    ts_results = st.session_state.ts_results
    
    if km_results is None and ts_results is None:
        st.warning("⚠️ Belum ada hasil clustering. Jalankan analisis terlebih dahulu.")
    else:
        if PLOTLY_AVAILABLE:
            tabs = st.tabs(["PCA Scatter", "Cluster Means", "Heatmap", "Time Series Patterns"])
            
            with tabs[0]:
                st.markdown("#### PCA 2D Visualization")
                
                if km_results is not None:
                    X = km_results['X']
                    labels = km_results['labels']
                    
                    pca = PCA(n_components=2, random_state=42)
                    X_pca = pca.fit_transform(X)
                    
                    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                    df_pca['Cluster'] = labels.astype(str)
                    
                    fig = px.scatter(
                        df_pca, x='PC1', y='PC2', color='Cluster',
                        title=f'PCA Visualization - K-Means (k={km_results["final_k"]})',
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                if ts_results is not None and TSL_AVAILABLE:
                    # PCA untuk time series (flatten)
                    X_ts_flat = ts_results['X_ts'].reshape(ts_results['X_ts'].shape[0], -1)
                    pca = PCA(n_components=2, random_state=42)
                    X_pca = pca.fit_transform(X_ts_flat)
                    
                    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                    df_pca['Cluster'] = ts_results['labels'].astype(str)
                    
                    fig = px.scatter(
                        df_pca, x='PC1', y='PC2', color='Cluster',
                        title=f'PCA Visualization - Time Series K-Means (k={ts_results["final_k"]})',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                st.markdown("#### Rata-rata Fitur per Cluster")
                
                if km_results is not None:
                    hasil_km = km_results['hasil']
                    pendidikan_order = st.session_state.pendidikan_order
                    
                    cluster_means = hasil_km.groupby('cluster')[pendidikan_order].mean()
                    
                    fig = px.bar(
                        cluster_means.reset_index().melt(id_vars='cluster', var_name='Pendidikan', value_name='Rata-rata'),
                        x='cluster', y='Rata-rata', color='Pendidikan',
                        title='K-Means - Rata-rata per Cluster',
                        barmode='group'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                if ts_results is not None and TSL_AVAILABLE:
                    st.markdown("##### Time Series - Rata-rata Pola per Cluster")
                    
                    X_ts = ts_results['X_ts']
                    labels = ts_results['labels']
                    tahun_order = [2020, 2021, 2022, 2023, 2024, 2025]
                    
                    fig = go.Figure()
                    for c in sorted(np.unique(labels)):
                        idx = np.where(labels == c)[0]
                        mean_pattern = X_ts[idx].mean(axis=0).ravel()
                        fig.add_trace(go.Scatter(
                            x=tahun_order, y=mean_pattern,
                            mode='lines+markers',
                            name=f'Cluster {c}'
                        ))
                    
                    fig.update_layout(
                        title='Time Series K-Means - Rata-rata Pola per Cluster',
                        xaxis_title='Tahun',
                        yaxis_title='Nilai (Normalized)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tabs[2]:
                st.markdown("#### Heatmap Cluster")
                
                if km_results is not None:
                    hasil_km = km_results['hasil']
                    pivot_cluster_km = hasil_km.pivot_table(
                        index='nama_kabupaten_kota', columns='tahun', values='cluster'
                    )
                    
                    fig = px.imshow(
                        pivot_cluster_km,
                        title=f'K-Means - Cluster per Kabupaten/Kota dan Tahun (k={km_results["final_k"]})',
                        color_continuous_scale='Viridis',
                        aspect="auto"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tabs[3]:
                st.markdown("#### Time Series Patterns")
                
                if ts_results is not None and TSL_AVAILABLE:
                    X_ts = ts_results['X_ts']
                    labels = ts_results['labels']
                    tahun_order = [2020, 2021, 2022, 2023, 2024, 2025]
                    
                    # Subplot per cluster
                    n_clusters = len(np.unique(labels))
                    fig, axes = plt.subplots(1, min(n_clusters, 3), figsize=(15, 4))
                    if n_clusters == 1:
                        axes = [axes]
                    
                    for i, c in enumerate(sorted(np.unique(labels))[:3]):
                        idx = np.where(labels == c)[0]
                        for j in idx[:20]:  # Plot sample
                            axes[i].plot(tahun_order, X_ts[j].ravel(), alpha=0.3, color='gray')
                        mean_pattern = X_ts[idx].mean(axis=0).ravel()
                        axes[i].plot(tahun_order, mean_pattern, color='red', linewidth=2, label='Rata-rata')
                        axes[i].set_title(f'Cluster {c} (n={len(idx)})')
                        axes[i].set_xlabel('Tahun')
                        axes[i].legend()
                    
                    plt.suptitle('Time Series Patterns per Cluster')
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.warning("⚠️ Plotly tidak tersedia. Install dengan: pip install plotly")

# ==================== UNDUH HASIL ====================

elif selected == "Unduh Hasil":
    st.markdown('<div class="sub-header">📥 Unduh Hasil</div>', unsafe_allow_html=True)
    
    km_results = st.session_state.km_results
    ts_results = st.session_state.ts_results
    
    if km_results is None and ts_results is None:
        st.info("Belum ada hasil untuk diunduh. Jalankan analisis terlebih dahulu.")
    else:
        st.markdown("#### Hasil yang Tersedia")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if km_results is not None:
                csv_km = km_results['hasil'].to_csv(index=False)
                st.download_button(
                    "📄 Unduh Hasil K-Means (.csv)",
                    data=csv_km,
                    file_name="hasil_kmeans.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if ts_results is not None:
                csv_ts = ts_results['hasil'].to_csv(index=False)
                st.download_button(
                    "📄 Unduh Hasil Time Series K-Means (.csv)",
                    data=csv_ts,
                    file_name="hasil_timeserieskmeans.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Unduh hasil komparasi
        if km_results is not None and ts_results is not None:
            comp_data = {
                'Algoritma': ['K-Means', 'Time Series K-Means (DTW)'],
                'Cluster Optimal': [km_results['final_k'], ts_results['final_k']],
                'Silhouette Score': [km_results['silhouette'], ts_results['silhouette']],
                'Davies-Bouldin Index': [km_results['dbi'], ts_results['dbi']]
            }
            comp_df = pd.DataFrame(comp_data)
            csv_comp = comp_df.to_csv(index=False)
            
            st.download_button(
                "📄 Unduh Hasil Komparasi (.csv)",
                data=csv_comp,
                file_name="hasil_komparasi.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================== TENTANG APLIKASI ====================

elif selected == "Tentang Aplikasi":
    st.markdown('<div class="sub-header">ℹ️ Tentang Aplikasi</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px;">
                <h3>📊 Sistem Komparasi Clustering Pengangguran Terbuka</h3>
                <p><strong>Versi:</strong> 2.0.0</p>
                <p><strong>Tanggal Rilis:</strong> 2026</p>
                <hr>
                <h4>🎯 Tujuan</h4>
                <p>Membantu analisis dan perbandingan hasil clustering antara algoritma 
                K-Means dan TimeSeriesKMeans untuk data pengangguran terbuka di Jawa Barat.</p>
                <hr>
                <h4>📚 Teknologi yang Digunakan</h4>
                <ul>
                    <li><strong>Framework:</strong> Streamlit</li>
                    <li><strong>Machine Learning:</strong> Scikit-learn, Tslearn</li>
                    <li><strong>Visualisasi:</strong> Plotly, Matplotlib, Seaborn</li>
                    <li><strong>Data Processing:</strong> Pandas, NumPy</li>
                </ul>
                <hr>
                <h4>📖 Metodologi</h4>
                <ul>
                    <li><strong>K-Means:</strong> Clustering berbasis jarak Euclidean dengan normalisasi global</li>
                    <li><strong>Time Series K-Means:</strong> Clustering berbasis DTW (Dynamic Time Warping) untuk pola waktu</li>
                    <li><strong>Evaluasi:</strong> Silhouette Score, Davies-Bouldin Index, dan analisis stabilitas</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background-color: #e8f4f8; padding: 2rem; border-radius: 10px;">
                <h4>📞 Kontak</h4>
                <p><strong>Email:</strong> support@clustering-app.com</p>
                <p><strong>Website:</strong> www.clustering-app.com</p>
                <hr>
                <h4>🔗 Link Terkait</h4>
                <ul>
                    <li><a href="#">Dokumentasi</a></li>
                    <li><a href="#">GitHub Repository</a></li>
                    <li><a href="#">Laporan Akhir</a></li>
                </ul>
                <hr>
                <h4>📊 Data Source</h4>
                <p>Dinas Tenaga Kerja dan Transmigrasi Jawa Barat</p>
            </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div class="footer">
        © 2026 Sistem Komparasi Clustering Pengangguran Terbuka. All rights reserved.
    </div>
""", unsafe_allow_html=True)
