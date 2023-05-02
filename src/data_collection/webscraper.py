import requests
from bs4 import BeautifulSoup
import random
import time
from langdetect import detect
import pandas as pd

class WebScraper:
    def __init__(self, random_seed) -> None:
        self.random_seed = random_seed
        self.headers = {"User-Agent": "Chrome/58.0.3029.110 (Windows NT 10.0; Win64; x64)"}
        self.language = "en"

    def get_review_urls(self, id, n_reviews):
        """
        Return review URLS
        """
        review_urls = []
        num_pages = (n_reviews +24) // 25 # Because there are 25 reviews per page

        for i in range(1, num_pages +1):
            # sort=helpfulnessScore: Reviews are sorted by Helpfulness scores. (How helpfull did other user find it)
            # dir=desc: Most helpful reviews first
            # ratingFilter=0: Show all reviews regardless of their score
            # start={25*(i-1)}: Determine the starting index for every page, 
            # but page 1 has starting index = 0, therefore i-1.
            url = f"https://www.imdb.com/title/{id}/reviews/_ajax?sort=helpfulnessScore&dir=desc&ratingFilter=0&start={25*(i-1)}"
            review_urls.append(url)
        return review_urls
    
    def get_reviews(self, movie_id, num_reviews, language="en"):
        "Return a list of dictionaries with review data"

        print("In get_reviews")

        urls = self.get_review_urls(movie_id, num_reviews)

        reviews = []
        response_fail_counter = 0
        for url in urls:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Request failed with response: {response.status_code}")
                response_fail_counter += 1
            else:
                soup = BeautifulSoup(response.text, "html.parser")
                
                for review in soup.find_all("div", {'class': 'review-container'}):
                    rating = review.find("span", {"class": "rating-other-user-rating"})
                    if rating:
                        rating  = int(rating.span.text)
                    else:
                        # Don't need a review without a rating
                        continue
                    title = review.find("a", {'class': 'title'}).text.strip()
                    text = review.find('div', {'class': 'text'}).text.strip()

                    try:
                        det_language = detect(text)
                    except:
                        det_language = None

                    if det_language == language:
                        reviews.append({ "review_title": title, "text": text, "rating": rating})
                time.sleep(random.uniform(0.5, 1.5))  # Sleep between requests to not get blocked
            return reviews
        

    def reviews_to_dataframe(self, reviews):
        print(len(reviews))
        return pd.DataFrame(reviews)
    
    def get_movie_ids_from_search(self, start_year, end_year, min_rating, max_rating, genre, sort, start=1):
        """
        Returns a list of IMDb movie IDs from Advanced Title Search with the specified
        parameters
        """
        url = f"https://www.imdb.com/search/title/?title_type=feature&release_date={start_year}-01-01,{end_year}-12-31&user_rating={min_rating},{max_rating}&genres={genre}&sort={sort}&start={start}"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")
        movie_ids = [a["href"].split("/")[2] for a in soup.select("div.lister-item-image a")]
        return movie_ids

    def get_random_movie_ids(self, num_movies, random_seed):
        """
        Uses random search parameters to get random movie ids.
        This is reducing any form of bias in the dataset. 
        """
        random.seed(random_seed)
        movie_ids = set()
        rating_min = 1.0
        rating_max = 10.0
        start_search = 1
        genres =["action", "family", "music", "documentary", "sci-fi", "horror", 
                 "adventure", "history", "thriller", "fantasy", 
                 "romance", "sport", "musical", "western", "crime", "drama", 
                 "animation", "mystery", "comedy", "war", "biography"]

        sort_methods = ["release_date,asc", "release_date,desc", "user_rating,asc", 
                        "user_rating,desc", "num_votes,asc", "num_votes,desc"]

        while len(movie_ids) < num_movies:
            start_year = random.randint(1990, 2015)
            end_year = start_year + 5
            genre = random.choice(genres)
            sort = random.choice(sort_methods)
            
            ids = self.get_movie_ids_from_search(start_year, end_year, rating_min, rating_max, genre, sort, start_search)
            movie_ids.update(ids)
            start_search += len(ids)

        return random.sample(list(movie_ids), num_movies)
    
    def get_movie_reviews(self, num_movies, num_reviews_per_movie, language="en"):
        """
        Returns a DataFrame with reviews and rating for random movies from a range of popular movies.
        """
        
        all_reviews = []
        movie_ids = self.get_random_movie_ids(num_movies, self.random_seed)
        for movie_id in movie_ids:
            try:
                reviews = self.get_reviews(movie_id, num_reviews_per_movie, language)
                all_reviews.extend(reviews)
            except Exception as e:
                print(f"Error for movie ID {movie_id}: {e}")
            time.sleep(random.uniform(1, 2))  # Sleep between different movies to not get blocked

        return self.reviews_to_dataframe(all_reviews)
    
if __name__ == "__main__":
    num_movies = 500
    num_reviews_per_movie = 100
    random_seed = 42
    language = "en"
    scraper = WebScraper(random_seed=random_seed)
    reviews_df = scraper.get_movie_reviews(num_movies, num_reviews_per_movie, language)
    reviews_df.to_csv(path_or_buf="imdb_reviews_rand.csv")