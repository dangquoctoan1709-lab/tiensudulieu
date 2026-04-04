import pandas as pd
import re
import os

# ==========================================
# HÀM LÀM SẠCH VĂN BẢN (TEXT CLEANING)
# ==========================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

# ==========================================
# XÁC ĐỊNH ĐƯỜNG DẪN TỰ ĐỘNG
# ==========================================
base_path = os.path.join(os.getcwd(), "Lab4", "Lab4")

# ==========================================
# BÀI 1: ALBUM REVIEWS
# ==========================================
print("--- 1. ALBUM REVIEWS ---")

df_album = pd.read_csv(os.path.join(base_path, 'ITA105_Lab_4_Album_reviews.csv'))

df_album['clean_review'] = df_album['review_text'].apply(clean_text)

album_stats = df_album.groupby('genre')['rating'].mean().round(2).reset_index()
print("Điểm trung bình theo thể loại nhạc:\n", album_stats)
print("-" * 30)

# ==========================================
# BÀI 2: HOTEL REVIEWS
# ==========================================
print("\n--- 2. HOTEL REVIEWS ---")

df_hotel = pd.read_csv(os.path.join(base_path, 'ITA105_Lab_4_Hotel_reviews.csv'))

df_hotel.dropna(inplace=True)

df_hotel['clean_review'] = df_hotel['review_text'].apply(clean_text)

hotel_stats = df_hotel.groupby('hotel_name')['rating'].mean().round(2).reset_index()
print("Điểm đánh giá trung bình của các khách sạn:\n", hotel_stats)
print("-" * 30)

# ==========================================
# BÀI 3: MATCH COMMENTS
# ==========================================
print("\n--- 3. MATCH COMMENTS ---")

df_match = pd.read_csv(os.path.join(base_path, 'ITA105_Lab_4_Match_comments.csv'))

df_match['clean_comment'] = df_match['comment_text'].apply(clean_text)

match_stats = df_match['team'].value_counts().reset_index()
match_stats.columns = ['team', 'comment_count']
print("Số lượng bình luận theo đội bóng:\n", match_stats)
print("-" * 30)

# ==========================================
# BÀI 4: PLAYER FEEDBACK
# ==========================================
print("\n--- 4. PLAYER FEEDBACK ---")

df_player = pd.read_csv(os.path.join(base_path, 'ITA105_Lab_4_Player_feedback.csv'))

df_player['score'] = pd.to_numeric(df_player['score'], errors='coerce')
df_player.dropna(subset=['score'], inplace=True)

df_player['clean_feedback'] = df_player['feedback_text'].apply(clean_text)

player_stats = df_player.groupby('device')['score'].mean().round(2).reset_index()
print("Điểm đánh giá trung bình theo thiết bị:\n", player_stats)