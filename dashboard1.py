import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="MindBalance Dashboard", layout="wide")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv('Mental_Health_Cleaned.csv')
    return df

# HALAMAN UTAMA
def halaman_utama():
    st.title("MindBalance Dashboard")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=1350&q=80", use_container_width=True)
    
    st.markdown("""
    ### **Ringkasan Proyek**
    MindBalance menganalisis hubungan antara gaya hidup digital dan kesehatan mental.
    Dataset ini berisi 3.000 responden dengan berbagai faktor seperti jam tidur, screen time, pola makan, dan sebagainya.
    """)
    st.info("Pilih menu 'Analisis Data' di samping untuk melihat hasil EDA.")

# HALAMAN EDA
def halaman_eda(df):
    st.title("Exploratory Data Analysis")

    # Filter Sidebar
    st.sidebar.subheader("Pengaturan Grafik")
    st.sidebar.write("Filter berdasarkan usia:")
    age_range = st.sidebar.slider("Rentang Usia", int(df['Age'].min()), int(df['Age'].max()), (20, 50))
    
    # Apply Filter
    filtered_df = df[df['Age'].between(age_range[0], age_range[1])]

    # Tab Menu
    tab1, tab2, tab3 = st.tabs(["Demografi & Target", "Gaya Hidup", "Kesehatan Mental"])

    with tab1:
        st.subheader("Distribusi Usia & Stress Level")
        col1, col2 = st.columns(2)
        with col1:
            fig_age = px.histogram(filtered_df, x="Age", title="Distribusi Usia", color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            # Karena data aslinya tulisan (Low, Moderate, High), kita urutkan manual
            fig_stress = px.pie(filtered_df, names='Stress Level', title="Proporsi Stress Level",
                                category_orders={"Stress Level": ["Low", "Moderate", "High"]},
                                color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_stress, use_container_width=True)

    with tab2:
        st.subheader("Analisis Gaya Hidup")
        col3, col4 = st.columns(2)
        with col3:
            # BOXPLOT SLEEP HOURS VS STRESS LEVEL
            fig_sleep = px.box(
                filtered_df, x="Stress Level", y="Sleep Hours", color="Stress Level",
                category_orders={"Stress Level": ["Low", "Moderate", "High"]},
                title="Jam Tidur Berdasarkan Tingkat Stres",
                color_discrete_map={"Low": "#1f77b4", "Moderate": "#ff7f0e", "High": "#2ca02c"},
            )
            st.plotly_chart(fig_sleep, use_container_width=True)
        with col4:
            # SCATTER PLOT SCREEN TIME VS HAPPINESS SCORE
            fig_screen = px.scatter(
                filtered_df, x="Screen Time per Day (Hours)", y="Happiness Score", color="Stress Level",
                category_orders={"Stress Level": ["Low", "Moderate", "High"]},
                title="Tingkat Kebahagiaan Berdasarkan Screen Time",
                color_discrete_map={"Low": "#1f77b4", "Moderate": "#ff7f0e", "High": "#2ca02c"},
            )
            st.plotly_chart(fig_screen, use_container_width=True)

    with tab3:
        st.subheader("Faktor Kesehatan & Kondisi Mental")
        col5, col6 = st.columns(2)
        with col5:
            # DIET TYPE VS STRESS LEVEL
            fig_diet = px.histogram(
                filtered_df, x="Diet Type", color="Stress Level", barmode="group",
                category_orders={"Stress Level": ["Low", "Moderate", "High"]},
                title="Tingkat Stres Berdasarkan Tipe Diet"
            )
            st.plotly_chart(fig_diet, use_container_width=True)
        with col6:
            # MENTAL HEALTH CONDITION
            fig_mh_pie = px.pie(
                filtered_df, names='Mental Health Condition', 
                title="Distribusi Mental Health Condition",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_mh_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_mh_pie, use_container_width=True)

# HALAMAN PREDIKSI
def halaman_prediksi():
    st.title("Simulasi Prediksi Stres")
    st.write("Masukkan data untuk mengecek estimasi tingkat stres.")
    
    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Usia", 15, 80, 22)
            sleep = st.slider("Jam Tidur", 0.0, 12.0, 7.0)
            diet = st.selectbox("Tipe Diet", ["Balanced", "Vegan", "Vegetarian", "Keto", "Junk Food"])
        with c2:
            screen = st.slider("Screen Time (Jam)", 0.0, 24.0, 5.0)
            work = st.number_input("Jam Kerja per Minggu", 0, 100, 40)
            exercise = st.selectbox("Level Olahraga", ["Low", "Moderate", "High"])
            
        if st.form_submit_button("Analisis"):
            st.success("Analisis berhasil! Hasil prediksi akan aktif di V2 dengan Model ML.")

# HALAMAN SEARCH
def halaman_pencarian(df):
    st.title("Pencarian Data")
    
    st.markdown("""
        <style>
            .stDataFrame {
                font-size: 18px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("Cari Kondisi Mental:", placeholder="Contoh: Anxiety")
        with col2:
            stress_choice = st.multiselect("Tingkat Stres:", options=df['Stress Level'].unique(), default=df['Stress Level'].unique())
        with col3:
            diet_choice = st.multiselect("Tipe Diet:", options=df['Diet Type'].unique(), default=df['Diet Type'].unique())

    # Filtering
    filtered_df = df[(df['Stress Level'].isin(stress_choice)) & (df['Diet Type'].isin(diet_choice))]
    if search_query:
        filtered_df = filtered_df[filtered_df['Mental Health Condition'].str.contains(search_query, case=False, na=False)]

    st.write(f"Menampilkan **{len(filtered_df)}** responden.")

    st.dataframe(filtered_df, use_container_width=True, height=800) 

    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name='data_mindbalance.csv', mime='text/csv')

# MAIN
def main():
    df = load_data()
    st.sidebar.title("MindBalance Menu")
    pilihan = st.sidebar.selectbox("Navigasi", ["Beranda", "Analisis Data", "Pencarian Data", "Simulasi Prediksi"])
    
    if pilihan == "Beranda":
        halaman_utama()
    elif pilihan == "Analisis Data":
        halaman_eda(df)
    elif pilihan == "Pencarian Data":
        halaman_pencarian(df)
    elif pilihan == "Simulasi Prediksi":
        halaman_prediksi()

if __name__ == "__main__":
    main()