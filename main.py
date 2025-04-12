import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import time
import random
import streamlit as st
import joblib
#from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities
import ast
import gdown

st.set_page_config(layout="wide")
# Sidebar Navigation
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio("Điều hướng", ["Giới thiệu", "Phân tích dữ liệu","Model Building", "Gợi ý sản phẩm"])
st.sidebar.image("image/sub_banner.png")
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📝 Ghi chú đồ án")
st.sidebar.markdown("""
**Đồ án tốt nghiệp DL07-K302**  
**GVHD**: cô Khuất Thùy Phương

**Nhóm thực hiện (Ngáo Ngơ Team)**  
- Hoàng Thị Thanh Huyền  
- Huỳnh Tấn Minh Triết
""")

# 1. Giới thiệu về project
if menu == "Giới thiệu":
    st.image("image/banner.jpg")
    st.title("Project 2: Recommender System")

    st.subheader("Nhóm thực hiện (Ngáo Ngơ Team)")
    st.markdown("""
    - Hoàng Thị Thanh Huyền  
    - Huỳnh Tấn Minh Triết
    """)

    st.subheader("Business Understanding")
    st.markdown("""
    Trong bối cảnh thương mại điện tử phát triển mạnh mẽ, việc hỗ trợ người dùng tìm kiếm và lựa chọn sản phẩm phù hợp đóng vai trò quan trọng trong việc nâng cao trải nghiệm mua sắm và thúc đẩy doanh số bán hàng.  
    Tuy nhiên, nhiều nền tảng vẫn chưa triển khai hoặc tối ưu hóa các hệ thống gợi ý sản phẩm một cách hiệu quả cho từng phân khúc khách hàng. Điều này dẫn đến trải nghiệm người dùng còn hạn chế, đặc biệt là khi họ phải tìm kiếm sản phẩm giữa hàng triệu lựa chọn khác nhau.  

    **Mục tiêu chính của đồ án:**  
    Xây dựng một hệ thống gợi ý sản phẩm (Recommender System) có khả năng:
    - Phân tích hành vi và nhu cầu của người dùng.
    - Đưa ra các gợi ý sản phẩm cá nhân hóa theo từng khách hàng.
    - Ứng dụng trong một hoặc một số nhóm hàng hóa tiêu biểu trên nền tảng web bán hàng.

    **Dự án triển khai hai mô hình đề xuất phổ biến:**
    - **Content-Based Filtering**: Gợi ý sản phẩm dựa trên đặc điểm nội dung và sở thích riêng của người dùng.
    - **Collaborative Filtering**: Gợi ý sản phẩm dựa trên hành vi và lịch sử tương tác của cộng đồng người dùng có đặc điểm tương đồng.
    """)

# 2. Phân tích và thống kê dữ liệu
elif menu == "Phân tích dữ liệu":
    st.title("📊 Phân tích Dữ liệu")
    @st.cache_data
    def load_csv_from_drive():
        os.makedirs("data", exist_ok=True)

        drive_files = {
            "Products_ThoiTrangNam_rating_raw.csv": "1D9fjsXCsuny7buOo-pbvCQt48oHuCc5b",  # Thay ID thực tế vào
            "Products_ThoiTrangNam_raw.csv": "1qAZFhPv_rdme6Cdt19agBkh4HNElF_Sp"        # Thay ID thực tế vào
        }

        for filename, file_id in drive_files.items():
            file_path = os.path.join("data", filename)
            if not os.path.exists(file_path):
                st.info(f"🔽 Đang tải {filename} từ Google Drive...")
                gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    

        return ratings_df, products_df



    data_option = st.radio("Chọn nguồn dữ liệu:", ("Sử dụng file mặc định", "Tải lên file riêng"))

    if data_option == "Sử dụng file mặc định":
        if os.path.exists("data/Products_ThoiTrangNam_rating_raw.csv") and os.path.exists("data/Products_ThoiTrangNam_raw.csv"):
            ratings_df = pd.read_csv("data/Products_ThoiTrangNam_rating_raw.csv", sep='\t')
            products_df = pd.read_csv("data/Products_ThoiTrangNam_raw.csv")
            st.success("Đã tải 2 file mặc định thành công.")
        else:
            st.error("Không tìm thấy một trong hai file mặc định. Vui lòng kiểm tra lại đường dẫn hoặc tên file.")
            st.stop()

    else:
        uploaded_rating_file = st.file_uploader("Tải lên file đánh giá (Collaborative Filtering)", type="csv")
        uploaded_product_file = st.file_uploader("Tải lên file sản phẩm (Content-Based Filtering)", type="csv")

        if uploaded_rating_file is not None and uploaded_product_file is not None:
            ratings_df = pd.read_csv(uploaded_rating_file, sep='\t')
            products_df = pd.read_csv(uploaded_product_file)
            st.success("Đã tải dữ liệu thành công.")
        else:
            st.warning("Vui lòng tải cả hai file để tiếp tục.")
            st.stop()

    # Nút nhấn để thực hiện phân tích
    if st.button("Phân tích dữ liệu"):
        st.subheader("📄 Dữ liệu đánh giá (ratings_df)")
        st.dataframe(ratings_df.head(3))
        st.subheader("📊 Tổng quan dữ liệu ratings")

        # Tính toán chỉ số
        num_users = ratings_df['user_id'].nunique()
        num_products = ratings_df['product_id'].nunique()
        num_ratings = ratings_df['rating'].nunique()

        avg_price = products_df['price'].mean()
        max_price = products_df['price'].max()
        min_price = products_df['price'].min()

        # Thiết kế các ô thống kê theo dạng 3 + 3 hoặc 2 + 2
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("###### 👤 Số lượng người đánh giá")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#2C6EF2'>{num_users}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("###### 📦 Số lượng sản phẩm")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#2C6EF2'>{num_products}</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("###### ⭐ Phân loại đánh giá ")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#2C6EF2'>{num_ratings}</div>", unsafe_allow_html=True)

        st.markdown("---")

        col4, col5, col6 = st.columns(3)
        with col5:
            st.markdown("###### 💰 Giá trung bình")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#27AE60'>{avg_price:,.0f} đ</div>", unsafe_allow_html=True)

        with col4:
            st.markdown("###### 💸 Giá cao nhất")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#E74C3C'>{max_price:,.0f} đ</div>", unsafe_allow_html=True)

        with col6:
            st.markdown("###### 🏷️ Giá thấp nhất")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#F39C12'>{min_price:,.0f} đ</div>", unsafe_allow_html=True)
        st.markdown("---")

        # 1. Bar Chart: Top 10 sản phẩm được đánh giá nhiều nhất
        st.markdown("### Top 10 sản phẩm được đánh giá nhiều nhất")
        # Tính số lượng đánh giá mỗi sản phẩm
        rating_counts = ratings_df['product_id'].value_counts().reset_index()
        rating_counts.columns = ['product_id', 'num_ratings']

        # Tính rating trung bình mỗi sản phẩm
        avg_rating = ratings_df.groupby('product_id')['rating'].mean().reset_index()
        avg_rating.columns = ['product_id', 'avg_rating']

        # Gộp dữ liệu
        product_stats = rating_counts.merge(avg_rating, on='product_id')

        # Gộp thêm thông tin giá và tên sản phẩm từ products_df
        product_stats = product_stats.merge(
            products_df[['product_id', 'product_name', 'price']],
            on='product_id',
            how='left'
        )

        # Xử lý tên sản phẩm (giới hạn độ dài)
        product_stats['product_name'] = product_stats['product_name'].fillna('Unknown').str.slice(0, 50) + '...'

        # Chọn Top 10 sản phẩm có rating trung bình cao nhất
        top_rated = product_stats.sort_values(by='avg_rating', ascending=False).head(10)

        # Sắp xếp để biểu đồ hiển thị đẹp
        top_rated = top_rated.sort_values(by='avg_rating', ascending=True)

        # Vẽ biểu đồ 3 trục
        fig, ax1 = plt.subplots(figsize=(12, 6))

        product_names = top_rated['product_name']
        x = range(len(product_names))

        # Trục 1: Bar chart cho số lượng đánh giá
        bars = ax1.bar(x, top_rated['num_ratings'], color='lightgray', label='Số lượng đánh giá')
        ax1.set_ylabel('Số lượng đánh giá', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        # Trục 2: Đường biểu diễn rating trung bình
        ax2 = ax1.twinx()
        ax2.plot(x, top_rated['avg_rating'], color='blue', marker='o', label='Rating trung bình')
        ax2.set_ylabel('Rating trung bình', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Trục 3: Đường biểu diễn giá
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))  # đẩy trục giá sang phải 1 chút
        ax3.plot(x, top_rated['price'], color='green', marker='s', label='Giá', linestyle='--')
        ax3.set_ylabel('Giá sản phẩm', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # Cài đặt trục x
        ax1.set_xticks(x)
        ax1.set_xticklabels(product_names, rotation=45, ha='right')
        ax1.set_title('🔝 Top 10 sản phẩm có rating trung bình cao nhất')

        # Thêm chú thích (legend)
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))

        st.pyplot(fig)
 
        # 2. Histogram: Phân phối các mức rating
        st.markdown("### 🔻 Top 10 sản phẩm bị đánh giá thấp nhất")
        # Tính số lượng đánh giá mỗi sản phẩm
        rating_counts = ratings_df['product_id'].value_counts().reset_index()
        rating_counts.columns = ['product_id', 'num_ratings']

        # Tính rating trung bình mỗi sản phẩm
        avg_rating = ratings_df.groupby('product_id')['rating'].mean().reset_index()
        avg_rating.columns = ['product_id', 'avg_rating']

        # Gộp dữ liệu
        product_stats = rating_counts.merge(avg_rating, on='product_id')

        # Gộp thêm thông tin giá và tên sản phẩm từ products_df
        product_stats = product_stats.merge(
            products_df[['product_id', 'product_name', 'price']],
            on='product_id',
            how='left'
        )

        # Xử lý tên sản phẩm
        product_stats['product_name'] = product_stats['product_name'].fillna('Unknown').str.slice(0, 50) + '...'

        # Chọn Top 10 sản phẩm có rating trung bình thấp nhất (có ít nhất vài đánh giá để hợp lý)
        filtered = product_stats[product_stats['num_ratings'] >= 3]  # loại sản phẩm chỉ có 1-2 rating để tránh sai lệch
        lowest_rated = filtered.sort_values(by='avg_rating', ascending=True).head(10)
        lowest_rated = lowest_rated.sort_values(by='avg_rating', ascending=True)

        # Vẽ biểu đồ
        fig, ax1 = plt.subplots(figsize=(12, 6))

        product_names = lowest_rated['product_name']
        x = range(len(product_names))

        # Trục 1: Bar chart cho số lượng đánh giá
        bars = ax1.bar(x, lowest_rated['num_ratings'], color='lightgray', label='Số lượng đánh giá')
        ax1.set_ylabel('Số lượng đánh giá', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        # Trục 2: Đường biểu diễn rating trung bình
        ax2 = ax1.twinx()
        ax2.plot(x, lowest_rated['avg_rating'], color='red', marker='o', label='Rating trung bình')
        ax2.set_ylabel('Rating trung bình', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Trục 3: Đường biểu diễn giá
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))  # đẩy trục ra ngoài
        ax3.plot(x, lowest_rated['price'], color='green', marker='s', label='Giá', linestyle='--')
        ax3.set_ylabel('Giá sản phẩm', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # X-axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels(product_names, rotation=45, ha='right')
        ax1.set_title('🔻 Top 10 sản phẩm bị đánh giá thấp nhất')

        # Thêm legend
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))

        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)

        # 3. Trung bình rating theo sản phẩm (top 10)
        st.markdown("### Top 10 sản phẩm có rating trung bình trung bình")
        # Tính số lượng đánh giá mỗi sản phẩm
        rating_counts = ratings_df['product_id'].value_counts().reset_index()
        rating_counts.columns = ['product_id', 'num_ratings']

        # Tính rating trung bình mỗi sản phẩm
        avg_rating = ratings_df.groupby('product_id')['rating'].mean().reset_index()
        avg_rating.columns = ['product_id', 'avg_rating']

        # Gộp dữ liệu
        product_stats = rating_counts.merge(avg_rating, on='product_id')

        # Gộp thêm thông tin giá và tên sản phẩm từ products_df
        product_stats = product_stats.merge(
            products_df[['product_id', 'product_name', 'price']],
            on='product_id',
            how='left'
        )

        # Xử lý tên sản phẩm
        product_stats['product_name'] = product_stats['product_name'].fillna('Unknown').str.slice(0, 50) + '...'

        # Chỉ giữ sản phẩm có số lượng đánh giá đủ lớn (tránh sản phẩm ít đánh giá gây nhiễu)
        filtered = product_stats[product_stats['num_ratings'] >= 3]

        # Tính trung vị rating để chọn nhóm "trung bình"
        median_rating = filtered['avg_rating'].median()

        # Lấy 10 sản phẩm có rating gần trung vị nhất
        filtered['rating_diff'] = (filtered['avg_rating'] - median_rating).abs()
        middle_rated = filtered.sort_values(by='rating_diff').head(10).sort_values(by='avg_rating', ascending=True)

        # Vẽ biểu đồ
        fig, ax1 = plt.subplots(figsize=(12, 6))

        product_names = middle_rated['product_name']
        x = range(len(product_names))

        # Trục 1: Bar chart cho số lượng đánh giá
        bars = ax1.bar(x, middle_rated['num_ratings'], color='lightgray', label='Số lượng đánh giá')
        ax1.set_ylabel('Số lượng đánh giá', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        # Trục 2: Đường biểu diễn rating trung bình
        ax2 = ax1.twinx()
        ax2.plot(x, middle_rated['avg_rating'], color='orange', marker='o', label='Rating trung bình')
        ax2.set_ylabel('Rating trung bình', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Trục 3: Đường biểu diễn giá
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.plot(x, middle_rated['price'], color='green', marker='s', label='Giá', linestyle='--')
        ax3.set_ylabel('Giá sản phẩm', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # Trục X: Tên sản phẩm
        ax1.set_xticks(x)
        ax1.set_xticklabels(product_names, rotation=45, ha='right')
        ax1.set_title('🟰 Top 10 sản phẩm có rating trung bình trung bình')

        # Legend
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))

        # Hiển thị biểu đồ
        st.pyplot(fig)

        
        st.subheader("📦 Dữ liệu sản phẩm (products_df)")
        st.dataframe(products_df.head(3), use_container_width=False)

        # Bắt đầu phần thống kê
        st.markdown("## 📊 Tổng quan từ dữ liệu sản phẩm")

        # Tính toán các thông số từ products_df
        num_products = products_df['product_id'].nunique()
        num_subcategories = products_df['sub_category'].nunique()
        num_with_image = products_df['image'].notna().sum()
        num_with_description = products_df['description'].notna().sum()

        # Thống kê giá
        price_stats = products_df['price'].describe()
        max_price = price_stats['max']
        avg_price = price_stats['mean']
        min_price = price_stats['min']
        std_price = price_stats['std']

        # Thống kê rating
        rating_stats = products_df['rating'].describe()
        max_rating = rating_stats['max']
        avg_rating = rating_stats['mean']
        min_rating = rating_stats['min']
        std_rating = rating_stats['std']

        # Hiển thị giao diện theo bố cục 4 cột x 3 hàng
        st.markdown("---")

        row1 = st.columns(4)
        with row1[0]:
            st.markdown("###### Số lượng sản phẩm")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#2C6EF2'>{num_products}</div>", unsafe_allow_html=True)
        with row1[1]:
            st.markdown("###### Số sub-category")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#2C6EF2'>{num_subcategories}</div>", unsafe_allow_html=True)
        with row1[2]:
            st.markdown("###### Hình ảnh miêu tả")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#2C6EF2'>{num_with_image}</div>", unsafe_allow_html=True)
        with row1[3]:
            st.markdown("###### Sản phẩm có mô tả")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#2C6EF2'>{num_with_description}</div>", unsafe_allow_html=True)
        st.markdown("---")
        row2 = st.columns(4)
        with row2[0]:
            st.markdown("###### Giá cao nhất")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#E74C3C'>{max_price:,.0f} đ</div>", unsafe_allow_html=True)
        with row2[1]:
            st.markdown("###### Giá trung bình")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#27AE60'>{avg_price:,.0f} đ</div>", unsafe_allow_html=True)
        with row2[2]:
            st.markdown("###### Giá thấp nhất")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#2980B9'>{min_price:,.0f} đ</div>", unsafe_allow_html=True)
        with row2[3]:
            st.markdown("###### Độ lệch chuẩn giá")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#F39C12'>{std_price:,.0f} đ</div>", unsafe_allow_html=True)
        st.markdown("---")
        row3 = st.columns(4)
        with row3[0]:
            st.markdown("###### Rating cao nhất")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#E74C3C'>{max_rating:.2f}</div>", unsafe_allow_html=True)
        with row3[1]:
            st.markdown("###### Rating trung bình")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#27AE60'>{avg_rating:.2f}</div>", unsafe_allow_html=True)
        with row3[2]:
            st.markdown("###### Rating thấp nhất")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#2980B9'>{min_rating:.2f}</div>", unsafe_allow_html=True)
        with row3[3]:
            st.markdown("###### Độ lệch chuẩn rating")
            st.markdown(f"<div style='font-size:24px; font-weight:bold; color:#F39C12'>{std_rating:.2f}</div>", unsafe_allow_html=True)

        st.markdown("---")
        
        st.subheader("📊 Thống kê Sub-category")
        st.markdown ("(số lượng sản phẩm, price và rating)")
        # Nhóm thống kê theo sub_category
        # Chuẩn bị dữ liệu
        # Chuẩn bị dữ liệu với std
        subcat_stats = products_df.groupby('sub_category').agg(
            num_products=('product_id', 'count'),
            avg_price=('price', 'mean'),
            std_price=('price', 'std'),
            avg_rating=('rating', 'mean'),
            std_rating=('rating', 'std')
        ).reset_index()

        subcat_stats = subcat_stats.sort_values(by='avg_rating')
        x = range(len(subcat_stats))
        labels = subcat_stats['sub_category']
        x_rating = [i + 0.2 for i in x]  # Dịch điểm rating sang phải

        # Vẽ biểu đồ
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Trục 1 – Số sản phẩm
        ax1.bar(x, subcat_stats['num_products'], color='lightgray', label='Số sản phẩm')
        ax1.set_ylabel('Số sản phẩm', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        # Trục 2 – Rating ± std
        ax2 = ax1.twinx()
        ax2.errorbar(x_rating, subcat_stats['avg_rating'], yerr=subcat_stats['std_rating'],
                    fmt='s--', color='blue', capsize=4, label='Rating trung bình ± Std')
        ax2.set_ylabel('Rating', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Trục 3 – Giá ± std
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.errorbar(x, subcat_stats['avg_price'], yerr=subcat_stats['std_price'],
                    fmt='o-', color='green', capsize=4, label='Giá trung bình ± Std')
        ax3.set_ylabel('Giá (đ)', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # Định dạng trục giá
        def format_price(x, pos):
            return f"{int(x):,} đ"
        ax3.yaxis.set_major_formatter(FuncFormatter(format_price))

        # Trục X
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_title('Thống kê theo Sub-category (Price và Rating với ± Std)')

        # Gộp legend
        lines, labels = [], []
        for ax in [ax1, ax2, ax3]:
            l, lab = ax.get_legend_handles_labels()
            lines += l
            labels += lab
        fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.01, 0.99))

        # Hiển thị
        st.pyplot(fig)
    
        
        # Lấy Top 3 sản phẩm có rating cao nhất
        top3_products = products_df.dropna(subset=['rating']).sort_values(by='rating', ascending=False).head(3)

        st.markdown("## 🔝 Top 3 sản phẩm có Rating cao nhất")

        # Tạo 3 cột song song
        cols = st.columns(3)

        for col, (_, row) in zip(cols, top3_products.iterrows()):
            with col:
                with st.container():
                    st.markdown("----")

                    # 1. Tên sản phẩm (tiêu đề)
                    st.markdown(f"###### 🛍️ {row['product_name']}")

                    # 2. Ảnh sản phẩm
                    if pd.notna(row['image']):
                        st.image(row['image'], caption="Ảnh sản phẩm")
                    else:
                        st.warning("Không có ảnh sản phẩm")

                    # 3. Mã sản phẩm
                    st.markdown(f"**📌 Mã sản phẩm:** `{row['product_id']}`")

                    # 4. Danh mục
                    st.markdown(f"`#{row['category']}`, `#{row['sub_category']}`")

                    # 5. Giá và đánh giá
                    st.markdown(f"**💰 Giá:** `{int(row['price']):,} đ`")
                    st.markdown(f"**⭐ Rating:** `{row['rating']:.2f} ⭐`")

                    # 6. Mô tả có thể mở rộng
                    st.markdown("**📝 Mô tả sản phẩm:**")
                    if pd.notna(row['description']):
                        short_desc = row['description'][:150] + "..."
                        with st.expander("📖 Xem toàn bộ mô tả"):
                            st.write(row['description'])
                        st.markdown(short_desc)
                    else:
                        st.info("Không có mô tả sản phẩm.")
        # Lấy Top 3 sản phẩm có rating thấp nhất
        bottom3_products = products_df.dropna(subset=['rating']).sort_values(by='rating', ascending=True).head(3)

        st.markdown("## 🔻 Top 3 sản phẩm có Rating thấp nhất")

        # Tạo 3 cột song song
        cols = st.columns(3)

        for col, (_, row) in zip(cols, bottom3_products.iterrows()):
            with col:
                with st.container():
                    st.markdown("----")

                    # 1. Tên sản phẩm (tiêu đề)
                    st.markdown(f"###### 🛍️ {row['product_name']}")

                    # 2. Ảnh sản phẩm
                    if pd.notna(row['image']):
                        st.image(row['image'], caption="Ảnh sản phẩm")
                    else:
                        st.warning("Không có ảnh sản phẩm")

                    # 3. Mã sản phẩm
                    st.markdown(f"**📌 Mã sản phẩm:** `{row['product_id']}`")

                    # 4. Danh mục
                    st.markdown(f"`#{row['category']}`, `#{row['sub_category']}`")

                    # 5. Giá và đánh giá
                    st.markdown(f"**💰 Giá:** `{int(row['price']):,} đ`")
                    st.markdown(f"**⭐ Rating:** `{row['rating']:.2f} ⭐`")

                    # 6. Mô tả có thể mở rộng
                    st.markdown("**📝 Mô tả sản phẩm:**")
                    if pd.notna(row['description']):
                        short_desc = row['description'][:150] + "..."
                        with st.expander("📖 Xem toàn bộ mô tả"):
                            st.write(row['description'])
                        st.markdown(short_desc)
                    else:
                        st.info("Không có mô tả sản phẩm.")

elif menu == "Model Building":
    st.title("🧠 Model Building")
    st.markdown("---")
    st.markdown("### 🔄 Tình trạng mô hình hiện tại của hệ thống")

    status_placeholder = st.empty()

    # Hiệu ứng chớp tắt 5 lần
    for i in range(3):
        status_placeholder.success("✅ Hệ thống hiện tại sử dụng **Content-Based (Cosine)** & **Collaborative (Surprise)**.  Đang hoạt động...")
        time.sleep(0.5)
        status_placeholder.info("⚙️ Đang xử lý dữ liệu mô hình, vui lòng chờ...")
        time.sleep(0.5)

    # Giữ lại trạng thái cuối cùng
    status_placeholder.success("✅ Hệ thống đang hoạt động ổn định với mô hình Cosine + Surprise.  Hướng đến Hybrid System trong tương lai.")
    # BƯỚC 1: DATA PREPROCESSING
    st.markdown("### 🔹 Bước 1: Xử lý dữ liệu đầu vào")
    st.markdown(""" 
    **Lưu ý:**  
        - Mô hình được tiến hành tiền xử lý và xây dựng dựa trên bộ dữ liệu cơ bản  
        - Để cập nhật thông tin và mô hình vui lòng liên hệ admin
    """)
    st.image("image/data_preprocessing.png", caption="Sơ đồ xử lý dữ liệu đầu vào")

    st.markdown("""
    - **Tổng hợp dữ liệu:**  
      `Data = product_name + description`
    
    - **Tiền xử lý cơ bản:**  
        - Tách dòng → tạo dataset  
        - Chuyển về chữ thường  
        - Kiểm tra `category`, `sub_category`  
        - Lọc từ không hợp lệ (`invalid_vietnames`)  
        - Loại bỏ stopwords  
    
    - **Quy tắc chọn từ:**  
        - Chọn 10 dòng đầu tiên  
        - Giới hạn 200 từ
    
    - **Tiền xử lý nâng cao (Pre-data):**
        - Token hóa  
        - Loại bỏ chữ số  
        - Loại ký tự đặc biệt  
        - Loại tiếp `invalid_vietnames`  
        - → Kết quả: `Final data`
    """)
    st.markdown("---")
    # BƯỚC 2: CONTENT-BASED MODEL
    st.markdown("### 🔹 Bước 2: Xây dựng Mô hình Content-Based Filtering")
    # Bảng so sánh Gensim vs Cosine
    cb_data = {
        "Metric": ["Precision", "Recall", "MRR", "MAP"],
        "Gensim": [1.00, 1.00, 1.00, 0.99],
        "Cosine": [0.80, 0.80, 0.52, 0.70]
    }
    cb_df = pd.DataFrame(cb_data)
    st.markdown("#### 📊 So sánh hiệu quả giữa Gensim và Cosine")
    st.table(cb_df)

    st.markdown("**📌 Mô hình sử dụng:** `Cosine Similarity`")

    st.markdown("""
    **🎯 Lý do chọn Cosine:**
    - Precision = **0.80**, Recall = **0.80**
    - MAP = **0.70**
    - Dễ triển khai, hiệu suất tốt
    
    🔍 So với Gensim:
    - Gensim đạt gần **1.00**, nhưng phức tạp và nặng hơn
    - Cosine là lựa chọn phù hợp để cân bằng giữa hiệu quả và tốc độ
    """)

    st.markdown("---")

    # BƯỚC 3: COLLABORATIVE MODEL
    st.markdown("### 🔹 Bước 3: Xây dựng Mô hình Collaborative Filtering")
    
    # Bảng so sánh ALS vs Surprise
    collab_data = {
        "Score": ["RMSE (càng nhỏ càng tốt)", "MAE (càng nhỏ càng tốt)"],
        "ALS": [1.16, 0.80],
        "Surprise": [1.01, 0.68]
    }
    collab_df = pd.DataFrame(collab_data)
    st.markdown("#### 📊 So sánh hiệu quả giữa ALS và Surprise")
    st.table(collab_df)
    st.markdown("**📌 Mô hình so sánh:** `ALS` và `Surprise`")

    st.markdown("""
    **🎯 Lý do chọn Surprise:**
    - RMSE thấp hơn (**1.01** so với 1.16)
    - MAE thấp hơn (**0.68** so với 0.80)
    - Phù hợp với ma trận sparse
    - Dễ triển khai và đánh giá
    """)
    st.markdown("---")
    st.markdown("### Tình trạng mô hình hiện tại của hệ thống")

    # Tổng kết
    st.success("✅ Hệ thống hiện tại sử dụng cả hai hướng: Content-Based (Cosine) & Collaborative (Surprise). Hướng đến Hybrid System trong tương lai.")

elif menu == "Gợi ý sản phẩm":

    # 3. Giao diện gợi ý sản phẩm
    # Load dữ liệu
    ratings_df = pd.read_csv("data/Products_ThoiTrangNam_rating_raw.csv", sep='\t')
    products_df = pd.read_csv("data/Products_ThoiTrangNam_raw.csv")
    content_df = pd.read_csv("data/content_with_tokens.csv")
    content_df['tokens'] = content_df['tokens'].apply(ast.literal_eval)

    # Load mô hình Gensim
    @st.cache_resource
    def load_tfidf_models():
        os.makedirs("models", exist_ok=True)

        drive_files = {
            "tfidf_gensim_dictionary.pkl": "1Ataz0dHDc0Oq8D4i0PhMdQD4AaIPfLbr",
            "tfidf_gensim_model.pkl": "1zpNzHVZQ9p1MydnmQiQeGOsirRc8Hxp7",
            "tfidf_gensim_index.index": "1MmO3ds8KJ3wFP0jIlgxqUGbxu7VbU6"
        }

        for filename, file_id in drive_files.items():
            file_path = os.path.join("models", filename)
            if not os.path.exists(file_path):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)

        with open("models/tfidf_gensim_dictionary.pkl", "rb") as f:
            dictionary = joblib.load(f)
        with open("models/tfidf_gensim_model.pkl", "rb") as f:
            tfidf_model = joblib.load(f)
        index = similarities.SparseMatrixSimilarity.load("models/tfidf_gensim_index.index")

        return dictionary, tfidf_model, index
    dictionary, tfidf_model, index = load_tfidf_models()

    @st.cache_resource
    def load_svd_model():
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_svd_model.pkl"
        file_id = "1zpNzHVZQ9p1MydnmQiQeGOsirRc8Hxp7"

        if not os.path.exists(model_path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

        return joblib.load(model_path)

    # Giao diện
    
    st.title("🛍️ Hệ thống Gợi ý Sản phẩm")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("👤 Nhập thông tin người dùng")

        mode = st.radio("🔧 Chọn chế độ người dùng:", [
            "🔢 Nhập ID bằng tay",
            "👤 Nhập username bằng tay",
            "🎲 Chọn ngẫu nhiên",
            "🚫 Không nhập thông tin"
        ])

        user_id = None
        username = None

        if mode == "🔢 Nhập ID bằng tay":
            input_user_id = st.text_input("Nhập User ID:")
            if st.button("💡 Gợi ý 5 ID bất kỳ"):
                sample_ids = random.sample(list(ratings_df['user_id'].unique()), 5)
                st.info(f"Gợi ý: {', '.join(map(str, sample_ids))}")
            if input_user_id:
                try:
                    input_user_id = int(input_user_id)
                    if input_user_id in ratings_df['user_id'].values:
                        info = ratings_df[ratings_df['user_id'] == input_user_id].iloc[0]
                        user_id = info['user_id']
                        username = info['user']
                        st.success(f"✅ Đã chọn User ID: {user_id} - Tên: {username}")
                    else:
                        st.warning("⚠️ ID không tồn tại!")
                except:
                    st.error("❌ Vui lòng nhập số hợp lệ!")

        elif mode == "👤 Nhập username bằng tay":
            input_user = st.text_input("Nhập Username:")
            if st.button("💡 Gợi ý 5 Username bất kỳ"):
                sample_users = random.sample(list(ratings_df['user'].unique()), 5)
                st.info(f"Gợi ý: {', '.join(sample_users)}")
            if input_user:
                matches = ratings_df[ratings_df['user'].str.lower() == input_user.strip().lower()]
                if not matches.empty:
                    info = matches.iloc[0]
                    user_id = info['user_id']
                    username = info['user']
                    st.success(f"✅ Đã chọn Username: {username} - ID: {user_id}")
                else:
                    st.warning("⚠️ Không tìm thấy username!")

        elif mode == "🎲 Chọn ngẫu nhiên":
            info = ratings_df.sample(1).iloc[0]
            user_id = info['user_id']
            username = info['user']
            st.success(f"✅ Hệ thống chọn ngẫu nhiên: ID: {user_id} - Tên: {username}")
        else:
            st.info("🔕 Bạn đã chọn không sử dụng thông tin người dùng.")

    with col2:
        st.subheader("📂 Chọn sản phẩm quan tâm")

        all_subs = sorted(products_df['sub_category'].dropna().unique())
        selected_sub = st.selectbox("Chọn danh mục phụ:", ["-- Chọn danh mục --"] + ["-- Tất cả --"] + all_subs)

        selected_product = None
        if selected_sub == "-- Chọn danh mục --":
            st.warning("⚠️ Vui lòng chọn danh mục trước khi chọn sản phẩm")
        else:
            if selected_sub == "-- Tất cả --":
                filtered_products = products_df.copy()
            else:
                filtered_products = products_df[products_df['sub_category'] == selected_sub]

            if not filtered_products.empty:
                product_names = filtered_products['product_name'].sample(
                    min(10, len(filtered_products))
                ).tolist()
                selected_product = st.selectbox("🎯 Chọn sản phẩm bạn quan tâm:", product_names)

        run_recommend = False
        if selected_product:
            run_recommend = st.button("🚀 Xem gợi ý sản phẩm")

    if selected_product and run_recommend:
        product_info = filtered_products[filtered_products['product_name'] == selected_product].iloc[0]

        st.markdown("### 📝 Thông tin sản phẩm đã chọn")
        col_img, col_info = st.columns([1, 2])

        with col_img:
            if isinstance(product_info['image'], str) and product_info['image'].startswith("http"):
                st.image(product_info['image'])
                fallback_path = "image/Image_not_available.png"
                if os.path.exists(fallback_path):
                    st.image(fallback_path)
                else:
                    st.warning("Không có ảnh", icon="⚠️")

        with col_info:
            st.markdown(f"**📌 Mã sản phẩm:** `{product_info['product_id']}`")
            st.markdown(f"**📦 Tên sản phẩm:** {product_info['product_name']}")
            st.markdown(f"**💰 Giá:** `{int(product_info['price']):,} đ`")
            st.markdown(f"**⭐ Rating:** `{product_info['rating']:.2f}`")
            st.markdown(f"**📂 Danh mục:** `{product_info['category']}` / `{product_info['sub_category']}`")
            st.markdown("**📝 Mô tả chi tiết:**")
            st.markdown(f"""
            <div style='font-size:18px; background-color:#f9f9f9; padding:10px; border-radius:10px; border:1px solid #ddd; line-height:1.5'>
            {product_info['description']}</div>
            """, unsafe_allow_html=True)

        # GỢI Ý THEO TF-IDF GENSIM
        st.markdown("---")
        idx = products_df[products_df['product_name'] == selected_product].index[0]
        tokens = content_df.loc[idx, 'tokens']
        query_bow = dictionary.doc2bow(tokens)
        query_tfidf = tfidf_model[query_bow]
        sims = index[query_tfidf]
        top_indices = np.argsort(sims)[::-1][1:6]

        st.markdown("## 🔄 Các sản phẩm tương tự:")
        cols = st.columns(len(top_indices))
        for col, i in zip(cols, top_indices):
            with col:
                row = products_df.iloc[i]
                st.markdown("----")
                if isinstance(row['image'], str) and row['image'].startswith("http"):
                    st.image(row['image'])
                else:
                    st.image("image/Image_not_available.png")
                st.markdown(f"**{row['product_name']}**")
                st.markdown(f"`#{row['category']}`, `#{row['sub_category']}`")
                st.markdown(f"💰 {int(row['price']):,} đ")
                st.markdown(f"⭐ {row['rating']}")

        # --- GỢI Ý THEO NGƯỜI DÙNG ---
        if user_id is not None:
            st.markdown("## 🤝 Gợi ý sản phẩm phù hợp với bạn trong danh mục")
            rated_ids = ratings_df[ratings_df['user_id'] == user_id]['product_id'].unique()
            filtered_unrated = filtered_products[~filtered_products['product_id'].isin(rated_ids)]

            if filtered_unrated.empty:
                st.info("🛑 Bạn đã đánh giá hết các sản phẩm trong danh mục.")
            else:
                svd_model = load_svd_model()
                predictions = [
                    (row['product_id'], svd_model.predict(user_id, row['product_id']).est)
                    for _, row in filtered_unrated.iterrows()
                ]
                top_preds = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
                top_ids = [pid for pid, _ in top_preds]
                recommended_df = filtered_unrated[filtered_unrated['product_id'].isin(top_ids)]

                cols = st.columns(len(recommended_df))
                for col, (_, row) in zip(cols, recommended_df.iterrows()):
                    with col:
                        st.markdown("----")
                        if isinstance(row['image'], str) and row['image'].startswith("http"):
                            st.image(row['image'])
                        else:
                            st.image("image/Image_not_available.png")
                        st.markdown(f"#### 🛍️ {row['product_name']}")
                        st.markdown(f"**📌 Mã sản phẩm:** `{row['product_id']}`")
                        st.markdown(f"`#{row['category']}`, `#{row['sub_category']}`")
                        st.markdown(f"**💰 Giá:** `{int(row['price']):,} đ`")
                        st.markdown(f"**⭐ Rating:** `{row['rating']:.2f}`")