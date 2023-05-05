import os
import requests
import json
from bs4 import BeautifulSoup
import random
import time
from langdetect import detect
import pandas as pd
import matplotlib.pyplot as plt

class WebScraper:
    def __init__(self, random_seed) -> None:
        self.random_seed = random_seed
        self.headers = {"User-Agent": "Chrome/58.0.3029.110 (Windows NT 10.0; Win64; x64)"}
        self.language = "en"


    def save_movie_ids(self, movie_ids, filename):
        with open(filename, "w") as outfile:
            json.dump(movie_ids, outfile)

    def load_movie_ids(self, filename):
        with open(filename, "r") as infile:
            return json.load(infile)
            
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
            url = f"https://www.imdb.com/title/{id}/reviews/_ajax?sort=submissionDate&dir=desc&ratingFilter=0&start={25*(i-1)}"
            review_urls.append(url)
        return review_urls
    
    def get_reviews(self, movie_id, num_reviews, language="en"):
        "Return a list of dictionaries with review data"

        print(f"Movie ID: {movie_id}")

        reviews = []
        response_fail_counter = 0
        start_index = 0
        review_set = set()  # Make sure to only get unique reviews (set).
        while len(reviews) < num_reviews:
            url = f"https://www.imdb.com/title/{movie_id}/reviews/_ajax?sort=submissionDate&dir=desc&ratingFilter=0&start={start_index}"
            print(f"URL: {url}")
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Request failed with response: {response.status_code}")
                response_fail_counter += 1
            else:
                soup = BeautifulSoup(response.text, "html.parser")
                prev_len = len(reviews)

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

                    if det_language == language and (title, text) not in review_set:
                        reviews.append({ "review_title": title, "text": text, "rating": rating})
                        review_set.add((title,text))

                # Calculate the actual number of collected reviews in the current iteration
                collected_reviews = len(reviews) - prev_len

                # If no new reviews are added, break the loop
                if collected_reviews == 0:
                    break

                start_index += collected_reviews
                time.sleep(0.5)  # Sleep between requests to not get blocked
        print(f"Number of collected reviews: {len(reviews)}")
        return reviews


        

    def reviews_to_dataframe(self, reviews):
        print(len(reviews))
        return pd.DataFrame([{"review_title": title, "text": text, "rating": rating} for title, text, rating in reviews])

    
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
        genres =["action", "family", "sci-fi", "horror", 
                 "adventure", "thriller", "fantasy", 
                 "romance", "sport", "musical", "western", "crime", "drama", 
                 "animation", "mystery", "comedy", "war", "biography"]

        sort_methods = ["release_date,asc", "release_date,desc", "user_rating,asc", 
                        "user_rating,desc", "num_votes,asc", "num_votes,desc"]

        while len(movie_ids) < num_movies:
            start_year = random.randint(1970, 2023)
            end_year = start_year + 10
            genre = random.choice(genres)
            sort = random.choice(sort_methods)
            
            ids = self.get_movie_ids_from_search(start_year, end_year, rating_min, rating_max, genre, sort, start_search)
            movie_ids.update(ids)
            start_search += len(ids)
            print(len(movie_ids))

        return list(movie_ids)
    
    def get_movie_reviews(self, num_movies, movie_ids, num_reviews_per_movie, language="en"):
        """
        Returns a DataFrame with reviews and rating for random movies from a range of popular movies.
        """
        movie_ids = set(movie_ids)
        all_reviews = set()
        total_reviews_required = num_movies * num_reviews_per_movie
        current_reviews_count = 0
        reviews_since_last_save = 0
        while current_reviews_count < total_reviews_required:
            for movie_id in movie_ids:
                #try:
                reviews = self.get_reviews(movie_id, num_reviews_per_movie, language)
                all_reviews_len_bef = len(all_reviews)
                all_reviews.update((review['review_title'], review['text'], review['rating']) for review in reviews)
                current_reviews_count += len(all_reviews) - all_reviews_len_bef
                reviews_since_last_save += len(all_reviews) - all_reviews_len_bef
                # except Exception as e:
                #     print(f"Error for movie ID {movie_id}: {e}")
                print(f"Number of currently collected review: {len(all_reviews)}")
                if reviews_since_last_save >= 1000:
                    print("Saving after at least 1000 collected reviews since last save...")
                    reviews_df = self.reviews_to_dataframe(all_reviews)
                    out_path = os.path.join("..", "..","data", "imdb_reviews_rand3.csv")
                    reviews_df.to_csv(path_or_buf=out_path)
                    del reviews_df
                    print("Done with Saving. Continuing to collect reviews.")
                    reviews_since_last_save = 0
                if current_reviews_count >= total_reviews_required:
                    break

                time.sleep(1.5)  # Sleep between different movies to not get blocked

            # If not enough reviews have been collected, get more movie IDs
            if current_reviews_count < total_reviews_required:
                num_movies_to_fetch = max((total_reviews_required - current_reviews_count) // num_reviews_per_movie, 1)
                new_movie_ids = self.get_random_movie_ids(num_movies_to_fetch, self.random_seed)
                movie_ids.update(new_movie_ids)

        return self.reviews_to_dataframe(all_reviews)
    
    def data_evaluation(self, reviews_df):
        reviews_df["rating"].hist(bins=10)
        plt.title("Distribution of the Ratings in Dataset")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        save_path = save_path = os.path.join("..", "..", "figures", "Data_distribution2.png")
        plt.savefig(save_path)

    
if __name__ == "__main__":
    load_ids = False
    num_movies = 6000
    num_reviews_per_movie = 1000
    random_seed = 42
    language = "en"
    scraper = WebScraper(random_seed=random_seed)

    # Save movie ids when load_ids = False. Load movie ids when load_ids = True 
    movie_ids_file = "movie_ids.json"
    if load_ids:
        movie_ids = scraper.load_movie_ids(movie_ids_file)
    else:
        movie_ids = scraper.get_random_movie_ids(num_movies, random_seed)
        scraper.save_movie_ids(movie_ids, movie_ids_file)

    reviews_df = scraper.get_movie_reviews(num_movies, movie_ids, num_reviews_per_movie, language)
    out_path = os.path.join("..", "..","data", "imdb_reviews_rand3.csv")
    reviews_df.to_csv(path_or_buf=out_path)
    scraper.data_evaluation(reviews_df)