import requests
import pandas as pd

class TMDbScraper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.root_url = "https://api.themoviedb.org/3"
        self.headers = {"Content-Type": "application/json;charset=utf-8"}

    def get_top_rated_movies(self, num_movies):
        movies = []
        page = 1
        while len(movies) < num_movies:
            url = f"{self.root_url}/movie/top_rated?api_key={self.api_key}&language=en-US&page={page}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                movies.extend(data['results'])
                page += 1
            else:
                print(f"Error fetching top-rated movies: {response.status_code}")
                break

        return movies[:num_movies]

    def get_movie_reviews(self, movie_id, num_reviews):
        reviews = []
        page = 1
        while len(reviews) < num_reviews:
            url = f"{self.root_url}/movie/{movie_id}/reviews?api_key={self.api_key}&language=en-US&page={page}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                if not data["results"]:  
                    break  
                for review in data['results']:
                    if review['author_details']['rating'] is not None:  # Filter out reviews without ratings
                        reviews.append(review)
                page += 1
            else:
                print(f"Error fetching reviews for movie {movie_id}: {response.status_code}")
                break
        return reviews[:num_reviews]


    def get_movies_by_genre(self, genre_id, num_movies):
        movies = []
        page = 1
        
        while len(movies) < num_movies:
            url = f"{self.root_url}/discover/movie?api_key={self.api_key}&with_genres={genre_id}&page={page}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                movies.extend(data['results'])
                print(f" Number of collected movies: {len(movies)} from genre {genre_id}")
                page += 1
            else:
                print(f"Error fetching movies by genre {genre_id}: {response.status_code}")
                break

        return movies[:num_movies]


    def collect_reviews(self, num_movies, num_reviews_per_movie):
        # Get list of popular genres
        genre_ids = [28, 35, 18, 10749, 27, 878]  # Action, Comedy, Drama, Romance, Horror, Science Fiction

        all_movies = []
        for genre_id in genre_ids:
            all_movies.extend(self.get_movies_by_genre(genre_id, num_movies // len(genre_ids)))

        all_reviews = []
        review_count = 0
        movie_count = 0

        for movie in all_movies:
            movie_id = movie['id']
            movie_title = movie['title']
            reviews = self.get_movie_reviews(movie_id, num_reviews_per_movie)
            for review in reviews:
                all_reviews.append({
                    "movie_id": movie_id,
                    "movie_title": movie_title,
                    "review_id": review['id'],
                    "author": review['author'],
                    "content": review['content'],
                    "rating": review['author_details']['rating']
                })
                review_count += 1

            movie_count += 1
            print(f"Collected {review_count} reviews after processing movie {movie_count}: {movie_title}")

            if review_count >= 1000:
                reviews_df = pd.DataFrame(all_reviews)
                reviews_df.to_csv("tmdb_reviews.csv", mode="a", header=False, index=False)
                all_reviews = []
                review_count = 0

        return pd.DataFrame(all_reviews)

if __name__ == "__main__":
    api_key = "346df8842ce5bdafd9442ade6d80e87c"
    scraper = TMDbScraper(api_key)
    num_movies = 20000
    num_reviews_per_movie = 1000
    reviews_df = scraper.collect_reviews(num_movies, num_reviews_per_movie)
    print(reviews_df)
    reviews_df.to_csv("tmdb_reviews.csv",mode='a', index=False)

