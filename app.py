import streamlit as st
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.header("Book Recommender System")

books_name = pickle.load(open('/Users/Dell/OneDrive/Desktop/DataAnalysis/Bookrecommender/book_title.pkl', 'rb'))
pt = pickle.load(open('/Users/Dell/OneDrive/Desktop/DataAnalysis/Bookrecommender/book_pivot.pkl', 'rb'))
image_url = pickle.load(open('/Users/Dell/OneDrive/Desktop/DataAnalysis/Bookrecommender/image_url.pkl', 'rb'))

# image_url list of URLs
image_url = image_url['Image-URL-L'].tolist()

similarity_score  = cosine_similarity(pt)

def recommend(selected_books):
    # index fetch
    index = np.where(pt.index == selected_books)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x:x[1], reverse=True)[1:11] # enumerate shows index along with matrix values
    
    recommended_books = []
    poster_urls = []

    for i in similar_items:
        recommended_books.append(pt.index[i[0]])
        poster_urls.append(image_url[i[0]])
        
    return recommended_books, poster_urls
    
selected_books = st.selectbox(
    "Type or select a book",
    books_name
)

if st.button('Show Recommendation'):
    recommendation_books, poster_urls = recommend(selected_books)

    num_columns = 3
    num_rows = 4
    book_index = 0
    
    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col in cols:
            if book_index < len(recommendation_books):
                col.write(recommendation_books[book_index])
                col.image(poster_urls[book_index], caption=recommendation_books[book_index], use_column_width=True)
                book_index += 1