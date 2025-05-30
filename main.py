# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(movies_path, ratings_path):
    # Load data
    data_movie = pd.read_csv(movies_path)
    data_rating = pd.read_csv(ratings_path)

    # Preprocessing movies
    data_movie['year'] = data_movie['title'].str.extract(r'\((\d{4})\)', expand=False)
    data_movie['title'] = data_movie['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.lower()
    data_movie['genres'] = data_movie['genres'].str.replace('|', ',', regex=False).str.lower()

    data_movie = data_movie.dropna(subset=['year'])
    data_movie['year'] = pd.to_numeric(data_movie['year'], errors='coerce')

    # Preprocessing ratings
    data_rating['timestamp'] = pd.to_datetime(data_rating['timestamp'], unit='s')

    # Merge datasets
    merged_data = pd.merge(data_rating, data_movie, on="movieId", how="inner")

    # Hitung rata-rata rating untuk setiap movie
    average_ratings = data_rating.groupby("movieId")["rating"].mean().round(1).reset_index()
    average_ratings.columns = ["movieId", "movie_rating"]

    # Merge rata-rata rating
    data = pd.merge(merged_data, average_ratings, on='movieId', how='left')

    # Label encoding
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    data['user_encoded'] = user_encoder.fit_transform(data['userId'])
    data['movie_encoded'] = movie_encoder.fit_transform(data['movieId'])

    # Scaling
    scaler_rating = StandardScaler()
    scaler_year = StandardScaler()
    data['movie_rating_scaled'] = scaler_rating.fit_transform(data[['movie_rating']])
    data['year_scaled'] = scaler_year.fit_transform(data[['year']])

    # Input features
    X = {
        'user_id': data['user_encoded'].values,
        'movie_id': data['movie_encoded'].values,
        'movie_rating': data['movie_rating_scaled'].values,
        'year': data['year_scaled'].values
    }
    y = data['rating'].values

    # Metadata & mapping
    movie_metadata = data[['movieId', 'title', 'movie_rating', 'year', 'genres']].drop_duplicates().set_index('movieId')
    mappings = {
        'user_id_to_encoded': dict(zip(data['userId'], data['user_encoded'])),
        'movie_id_to_encoded': dict(zip(data['movieId'], data['movie_encoded'])),
        'encoded_to_user_id': dict(zip(data['user_encoded'], data['userId'])),
        'encoded_to_movie_id': dict(zip(data['movie_encoded'], data['movieId']))
    }

    return data, X, y, data['user_encoded'].nunique(), data['movie_encoded'].nunique(), scaler_rating, scaler_year, movie_metadata, mappings


def build_model(n_users, n_movies, embedding_dim=64, hidden_units=[128, 64, 32], dropout_rate=0.3, l2_reg=0.001):
    user_input = layers.Input(shape=(), name='user_id')
    movie_input = layers.Input(shape=(), name='movie_id')
    rating_input = layers.Input(shape=(), name='movie_rating')
    year_input = layers.Input(shape=(), name='year')

    user_embedding = layers.Embedding(n_users, embedding_dim, embeddings_regularizer=keras.regularizers.l2(l2_reg))(user_input)
    movie_embedding = layers.Embedding(n_movies, embedding_dim, embeddings_regularizer=keras.regularizers.l2(l2_reg))(movie_input)

    user_vec = layers.Flatten()(user_embedding)
    movie_vec = layers.Flatten()(movie_embedding)

    dot_product = layers.Dot(axes=1)([user_vec, movie_vec])
    element_wise = layers.Multiply()([user_vec, movie_vec])

    concat = layers.Concatenate()([
        user_vec, movie_vec, dot_product, element_wise,
        layers.Reshape((1,))(rating_input),
        layers.Reshape((1,))(year_input)
    ])

    x = concat
    for units in hidden_units:
        x = layers.Dense(units, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs=[user_input, movie_input, rating_input, year_input], outputs=output)
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae', 'mse'])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint('./models/best_model.h5', save_best_only=True)
    ]

    history = model.fit(
        x=[X_train['user_id'], X_train['movie_id'], X_train['movie_rating'], X_train['year']],
        y=y_train,
        validation_data=([
            X_val['user_id'], X_val['movie_id'], X_val['movie_rating'], X_val['year']
        ], y_val),
        epochs=50,
        batch_size=1024,
        callbacks=callbacks
    )
    return model, history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict([X_test['user_id'], X_test['movie_id'], X_test['movie_rating'], X_test['year']]).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")

def recommend_for_user(user_id, model, movie_metadata, mappings, scaler_rating, scaler_year, data):
    if user_id not in mappings['user_id_to_encoded']:
        print(f"❌ User ID {user_id} tidak ditemukan.")
        return

    user_enc = mappings['user_id_to_encoded'][user_id]
    all_movie_ids = movie_metadata.index.tolist()
    user_watched = data[data['userId'] == user_id]['movieId'].tolist()
    candidate_movie_ids = [mid for mid in all_movie_ids if mid not in user_watched]

    if not candidate_movie_ids:
        print(f"🎯 User {user_id} telah menonton semua film.")
        return

    movie_encodings = [mappings['movie_id_to_encoded'][mid] for mid in candidate_movie_ids]
    user_array = np.array([user_enc] * len(candidate_movie_ids))
    movie_array = np.array(movie_encodings)

    movie_ratings = np.array([movie_metadata.loc[mid]['movie_rating'] for mid in candidate_movie_ids]).reshape(-1, 1)
    years = np.array([movie_metadata.loc[mid]['year'] for mid in candidate_movie_ids]).reshape(-1, 1)

    rating_scaled = scaler_rating.transform(movie_ratings)
    year_scaled = scaler_year.transform(years)

    predicted = model.predict([user_array, movie_array, rating_scaled.flatten(), year_scaled.flatten()]).flatten()

    results = pd.DataFrame({
        'movieId': candidate_movie_ids,
        'predicted_rating': predicted
    })
    results['title'] = results['movieId'].map(movie_metadata['title'])
    results['year'] = results['movieId'].map(movie_metadata['year'])
    results['imdb_rating'] = results['movieId'].map(movie_metadata['movie_rating'])
    results['genres'] = results['movieId'].map(movie_metadata['genres'])

    top_n = results.sort_values(by='predicted_rating', ascending=False).head(10)

    print(f"\n🎬 RECOMMENDATIONS FOR USER {user_id}")
    print("=" * 80)
    for i, row in enumerate(top_n.itertuples(), 1):
        print(f"{i:2d}. 🎬 {row.title} ({int(row.year)})")
        print(f"    ⭐ IMDB: {round(row.imdb_rating, 1)}/10 | 🎭 {row.genres}")
        print()
    print(f"📊 Evaluated {len(candidate_movie_ids)} movies")

def main():
    data_movies = "./data/movies.csv"
    data_ratings = "./data/ratings.csv"
    data, X, y, n_users, n_movies, scaler_rating, scaler_year, movie_metadata, mappings = load_and_preprocess_data(data_movies, data_ratings)

    # Split Data
    X_temp, X_test, y_temp, y_test = {}, {}, None, None
    for key in X:
        X_temp[key], X_test[key] = train_test_split(X[key], test_size=0.2, random_state=42)
    y_temp, y_test = train_test_split(y, test_size=0.2, random_state=42)

    val_ratio = 0.1 / 0.8
    X_train, X_val = {}, {}
    for key in X_temp:
        X_train[key], X_val[key] = train_test_split(X_temp[key], test_size=val_ratio, random_state=42)
    y_train, y_val = train_test_split(y_temp, test_size=val_ratio, random_state=42)

    # Train and Evaluate
    model = build_model(n_users, n_movies)
    model, history = train_model(model, X_train, y_train, X_val, y_val)

    # Save & Evaluate
    model.save("./models/final_model.keras")
    evaluate_model(model, X_test, y_test)

    # Recommend for user
    target_user_id = 1
    loaded_model = keras.models.load_model("./models/final_model.keras")
    recommend_for_user(target_user_id, loaded_model, movie_metadata, mappings, scaler_rating, scaler_year, data)

if __name__ == "__main__":
    main()
