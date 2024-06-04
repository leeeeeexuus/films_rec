import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import bcrypt

# Загрузка данных из файла
file_path = 'IMDb_Top_250_Movies.csv'
movies = pd.read_csv(file_path, encoding='ISO-8859-1')

movies['movieId'] = movies.index + 1

# данные
movies['Actors'] = movies['Stars'].str.split(', ')
movies['Directors'] = movies['Director'].str.split(', ')
movies['Genres'] = movies['Certificate'].str.split(', ')


ratings = movies[['movieId', 'Name', 'Rating']].copy()
ratings.columns = ['movieId', 'Name', 'rating']
user_ids = np.random.randint(1, 21, size=len(ratings))
ratings['userId'] = user_ids

# Создание матрицы признаков фильмов
def create_feature_matrix(movies):
    features = ['Rating', 'Actors', 'Directors', 'Genres']
    df = movies[features].copy()

    for feature in features[1:]:
        df = df.explode(feature)

    for feature in features[1:]:
        dummies = pd.get_dummies(df[feature])
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[feature], inplace=True)

    df = df.groupby(df.index).mean()
    df = (df - df.min()) / (df.max() - df.min())
    return df

# Создание матрицы признаков фильмов
movie_features = create_feature_matrix(movies)

# Вычисление косинусного сходства между фильмами
item_sim = cosine_similarity(movie_features)
item_sim_df = pd.DataFrame(item_sim, index=movies['Name'], columns=movies['Name'])

# Функция для вычисления взвешенного среднего сходства
def weighted_mean(similar_items, weights):
    return np.average(similar_items, weights=weights)

# Рекомендация фильмов на основе списка избранных фильмов пользователя
def recommend_based_on_favorites(favorite_movies, user_favorites, num_recommend=5):
    all_similarities = []
    for movie in favorite_movies:
        similar_items = item_sim_df[movie]
        all_similarities.append(similar_items)

    all_similarities = np.array(all_similarities)
    weights = np.ones(len(favorite_movies)) / len(favorite_movies)
    combined_similarities = np.dot(weights, all_similarities)

    combined_similarities = pd.Series(combined_similarities, index=item_sim_df.index)
    combined_similarities = combined_similarities.drop(labels=favorite_movies + user_favorites)
    recommendations = combined_similarities.sort_values(ascending=False).head(num_recommend)

    return recommendations.index

# Отображение информации о фильмах
def movie_info(movie):
    if movie not in movies['Name'].unique():
        return f"Фильм '{movie}' не найден."
    movie_data = movies[movies['Name'] == movie].iloc[0]
    info = f"Информация о фильме '{movie}':\n"
    info += f"№: {movie_data['Sl_No']}\n"
    info += f"Название: {movie_data['Name']}\n"
    info += f"Год выпуска: {movie_data['Release_Year']}\n"
    info += f"Продолжительность: {movie_data['Duration']} минут\n"
    info += f"Сертификат: {movie_data['Certificate']}\n"
    info += f"Рейтинг: {movie_data['Rating']}\n"
    info += f"Голоса: {movie_data['Votes']}\n"
    info += f"Режиссер: {movie_data['Director']}\n"
    info += f"Звезды: {movie_data['Stars']}\n"
    info += f"Описание: {movie_data['Description']}\n"
    return info

# Класс для управления аккаунтами
class UserManager:
    def __init__(self, filename='users.json'):
        self.filename = filename
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                json.dump({}, file)
        with open(filename, 'r') as file:
            self.users = json.load(file)

    def save(self):
        with open(self.filename, 'w') as file:
            json.dump(self.users, file)

    def register(self, username, password):
        if username in self.users:
            return False
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        self.users[username] = {'password': hashed_password.decode('utf-8'), 'favorites': [], 'watchlist': []}
        self.save()
        return True

    def login(self, username, password):
        if username in self.users:
            stored_password = self.users[username]['password'].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                if 'watchlist' not in self.users[username]:
                    self.users[username]['watchlist'] = []
                if 'favorites' not in self.users[username]:
                    self.users[username]['favorites'] = []
                self.save()
                return True
        return False

    def add_favorite(self, username, movie):
        if username in self.users:
            if movie not in self.users[username]['favorites']:
                self.users[username]['favorites'].append(movie)
                self.save()

    def remove_favorite(self, username, movie):
        if username in self.users:
            if movie in self.users[username]['favorites']:
                self.users[username]['favorites'].remove(movie)
                self.save()

    def get_favorites(self, username):
        if username in self.users:
            return self.users[username]['favorites']
        return []

    def add_watchlist(self, username, movie):
        if username in self.users:
            if movie not in self.users[username]['watchlist']:
                self.users[username]['watchlist'].append(movie)
                self.save()

    def remove_watchlist(self, username, movie):
        if username in self.users:
            if movie in self.users[username]['watchlist']:
                self.users[username]['watchlist'].remove(movie)
                self.save()

    def get_watchlist(self, username):
        if username in self.users:
            return self.users[username]['watchlist']
        return []

# Создание интерфейса
class MovieRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система рекомендаций фильмов")
        self.user_manager = UserManager()
        self.current_user = None


        self.canvas = tk.Canvas(root)
        self.scroll_y = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(root, orient="horizontal", command=self.canvas.xview)

        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        # Установка стиля
        style = ttk.Style()
        style.configure("TLabel", padding=6, font=("Helvetica", 12))
        style.configure("TButton", padding=6, font=("Helvetica", 12))
        style.configure("TEntry", padding=6, font=("Helvetica", 12))
        style.configure("TText", padding=6, font=("Helvetica", 12))

        # Заголовок
        title = ttk.Label(self.scrollable_frame, text="Система рекомендаций фильмов", font=("Helvetica", 16, "bold"))
        title.pack(pady=10)

        # Ввод имени пользователя
        self.username_label = ttk.Label(self.scrollable_frame, text="Имя пользователя:")
        self.username_label.pack()
        self.username_entry = ttk.Entry(self.scrollable_frame, width=100)
        self.username_entry.pack()

        # Ввод пароля
        self.password_label = ttk.Label(self.scrollable_frame, text="Пароль:")
        self.password_label.pack()
        self.password_entry = ttk.Entry(self.scrollable_frame, width=100, show='*')
        self.password_entry.pack()

        # Кнопки регистрации и входа
        self.register_button = ttk.Button(self.scrollable_frame, text="Регистрация", command=self.register)
        self.register_button.pack(pady=5)
        self.login_button = ttk.Button(self.scrollable_frame, text="Вход", command=self.login)
        self.login_button.pack(pady=5)

        # Ввод любимых фильмов
        self.movies_label = ttk.Label(self.scrollable_frame, text="Введите ваши любимые фильмы (через запятую):")
        self.movies_label.pack()
        self.movies_entry = ttk.Entry(self.scrollable_frame, width=100)
        self.movies_entry.pack()

        self.paste_movies_button = ttk.Button(self.scrollable_frame, text="Вставить", command=self.paste_movies)
        self.paste_movies_button.pack(pady=5)

        self.recommend_button = ttk.Button(self.scrollable_frame, text="Получить рекомендации",
                                           command=self.get_recommendations)
        self.recommend_button.pack(pady=10)

        # Отображение рекомендаций
        self.recommendations_text = tk.Text(self.scrollable_frame, height=10, width=100, font=("Helvetica", 12))
        self.recommendations_text.pack(pady=10)

        # Ввод названия фильма для информации
        self.movie_label = ttk.Label(self.scrollable_frame, text="Введите название фильма для информации:")
        self.movie_label.pack()
        self.movie_entry = ttk.Entry(self.scrollable_frame)
        self.movie_entry.pack()

        self.paste_movie_button = ttk.Button(self.scrollable_frame, text="Вставить", command=self.paste_movie)
        self.paste_movie_button.pack(pady=5)

        self.movie_info_button = ttk.Button(self.scrollable_frame, text="Получить информацию о фильме",
                                            command=self.get_movie_info)
        self.movie_info_button.pack(pady=10)

        # Отображение информации о фильмах
        self.movie_info_text = tk.Text(self.scrollable_frame, height=10, width=100, font=("Helvetica", 12))
        self.movie_info_text.pack(pady=10)

        # Кнопка для добавления в избранное
        self.add_favorite_button = ttk.Button(self.scrollable_frame, text="Добавить в избранное",
                                              command=self.add_to_favorites)
        self.add_favorite_button.pack(pady=10)

        # Ввод названия фильма для удаления из избранного
        self.remove_favorite_label = ttk.Label(self.scrollable_frame, text="Введите название фильма для удаления из избранного:")
        self.remove_favorite_label.pack()
        self.remove_favorite_entry = ttk.Entry(self.scrollable_frame)
        self.remove_favorite_entry.pack()

        self.remove_favorite_button = ttk.Button(self.scrollable_frame, text="Удалить из избранного",
                                                 command=self.remove_from_favorites)
        self.remove_favorite_button.pack(pady=5)

        # Кнопка для добавления в список для просмотра
        self.add_watchlist_button = ttk.Button(self.scrollable_frame, text="Добавить в список для просмотра",
                                               command=self.add_to_watchlist)
        self.add_watchlist_button.pack(pady=10)

        # Ввод названия фильма для удаления из списка для просмотра
        self.remove_watchlist_label = ttk.Label(self.scrollable_frame, text="Введите название фильма для удаления из списка для просмотра:")
        self.remove_watchlist_label.pack()
        self.remove_watchlist_entry = ttk.Entry(self.scrollable_frame)
        self.remove_watchlist_entry.pack()

        self.remove_watchlist_button = ttk.Button(self.scrollable_frame, text="Удалить из списка для просмотра",
                                                  command=self.remove_from_watchlist)
        self.remove_watchlist_button.pack(pady=5)

        # Отображение списка избранных фильмов
        self.favorites_label = ttk.Label(self.scrollable_frame, text="Избранные фильмы:")
        self.favorites_label.pack()
        self.favorites_text = tk.Text(self.scrollable_frame, height=10, width=100, font=("Helvetica", 12))
        self.favorites_text.pack(pady=10)

        # Отображение списка фильмов для просмотра
        self.watchlist_label = ttk.Label(self.scrollable_frame, text="Список для просмотра:")
        self.watchlist_label.pack()
        self.watchlist_text = tk.Text(self.scrollable_frame, height=10, width=100, font=("Helvetica", 12))
        self.watchlist_text.pack(pady=10)

    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if self.user_manager.register(username, password):
            messagebox.showinfo("Успех", "Регистрация прошла успешно!")
        else:
            messagebox.showerror("Ошибка", "Имя пользователя уже существует.")

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if self.user_manager.login(username, password):
            self.current_user = username
            messagebox.showinfo("Успех", "Вход выполнен успешно!")
            self.update_favorites()
            self.update_watchlist()
        else:
            messagebox.showerror("Ошибка", "Неверное имя пользователя или пароль.")

    def get_recommendations(self):
        if self.current_user is None:
            messagebox.showerror("Ошибка", "Пожалуйста, войдите в систему.")
            return

        try:
            favorite_movies = [movie.strip() for movie in self.movies_entry.get().split(',')]
            for movie in favorite_movies:
                if movie not in item_sim_df.index:
                    messagebox.showerror("Ошибка", f"Фильм '{movie}' не найден в базе данных.")
                    return
                self.user_manager.add_favorite(self.current_user, movie)
            user_favorites = self.user_manager.get_favorites(self.current_user)
            recommendations = recommend_based_on_favorites(favorite_movies, user_favorites)
            self.recommendations_text.delete(1.0, tk.END)
            self.recommendations_text.insert(tk.END, f"Рекомендации на основе ваших любимых фильмов:\n")
            for rec in recommendations:
                self.recommendations_text.insert(tk.END, f"{rec}\n")
                self.user_manager.add_watchlist(self.current_user, rec)
            self.update_favorites()
            self.update_watchlist()
        except ValueError:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректный список фильмов.")

    def get_movie_info(self):
        movie = self.movie_entry.get()
        info = movie_info(movie)
        self.movie_info_text.delete(1.0, tk.END)
        self.movie_info_text.insert(tk.END, info)

    def paste_movies(self):
        self.movies_entry.insert(tk.END, self.root.clipboard_get())

    def paste_movie(self):
        self.movie_entry.insert(tk.END, self.root.clipboard_get())

    def add_to_favorites(self):
        if self.current_user is None:
            messagebox.showerror("Ошибка", "Пожалуйста, войдите в систему.")
            return

        favorite_movie = self.movie_entry.get()
        if favorite_movie not in movies['Name'].unique():
            messagebox.showerror("Ошибка", f"Фильм '{favorite_movie}' не найден.")
        else:
            self.user_manager.add_favorite(self.current_user, favorite_movie)
            messagebox.showinfo("Успех", f"Фильм '{favorite_movie}' добавлен в избранное.")
            self.update_favorites()

    def remove_from_favorites(self):
        if self.current_user is None:
            messagebox.showerror("Ошибка", "Пожалуйста, войдите в систему.")
            return

        favorite_movie = self.remove_favorite_entry.get()
        if favorite_movie not in movies['Name'].unique():
            messagebox.showerror("Ошибка", f"Фильм '{favorite_movie}' не найден.")
        else:
            self.user_manager.remove_favorite(self.current_user, favorite_movie)
            messagebox.showinfo("Успех", f"Фильм '{favorite_movie}' удален из избранного.")
            self.update_favorites()

    def add_to_watchlist(self):
        if self.current_user is None:
            messagebox.showerror("Ошибка", "Пожалуйста, войдите в систему.")
            return

        watch_movie = self.movie_entry.get()
        if watch_movie not in movies['Name'].unique():
            messagebox.showerror("Ошибка", f"Фильм '{watch_movie}' не найден.")
        else:
            self.user_manager.add_watchlist(self.current_user, watch_movie)
            messagebox.showinfo("Успех", f"Фильм '{watch_movie}' добавлен в список для просмотра.")
            self.update_watchlist()

    def remove_from_watchlist(self):
        if self.current_user is None:
            messagebox.showerror("Ошибка", "Пожалуйста, войдите в систему.")
            return

        watch_movie = self.remove_watchlist_entry.get()
        if watch_movie not in movies['Name'].unique():
            messagebox.showerror("Ошибка", f"Фильм '{watch_movie}' не найден.")
        else:
            self.user_manager.remove_watchlist(self.current_user, watch_movie)
            messagebox.showinfo("Успех", f"Фильм '{watch_movie}' удален из списка для просмотра.")
            self.update_watchlist()

    def update_favorites(self):
        if self.current_user:
            favorites = self.user_manager.get_favorites(self.current_user)
            self.favorites_text.delete(1.0, tk.END)
            self.favorites_text.insert(tk.END, "\n".join(favorites))

    def update_watchlist(self):
        if self.current_user:
            watchlist = self.user_manager.get_watchlist(self.current_user)
            self.watchlist_text.delete(1.0, tk.END)
            self.watchlist_text.insert(tk.END, "\n".join(watchlist))

# Запуск
root = tk.Tk()
root.geometry("800x1000")
app = MovieRecommenderApp(root)
root.mainloop()
