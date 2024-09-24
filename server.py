import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')
predefined_location = ('Albuquerque', 'New Mexico', 'Carlsbad, California', 'Chula Vista', 'California', 'Colorado Springs, Colorado', 'Denver, Colorado', 'El Cajon, California', 'El Paso, Texas', 'Escondido, California', 'Fresno, California', 'La Mesa, California', 'Las Vegas, Nevada', 'Los Angeles, California', 'Oceanside, California', 'Phoenix, Arizona', 'Sacramento, California', 'Salt Lake City, Utah', 'Salt Lake City, Utah', 'San Diego, California', 'Tucson, Arizona')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":            
            # Write your code here
            # To get the query parameters
            query_params = parse_qs(environ.get("QUERY_STRING", ""))
            location = query_params.get("location", [None])[0]
            start_date = query_params.get("start_date", [None])[0]
            end_date = query_params.get("end_date", [None])[0]

            filtered_reviews = reviews

            # To check if we have location filter and that location is vaild
            if location and location in predefined_location:
                filtered_reviews = [
                    review for review in filtered_reviews if review['Location'] == location
                ]

            # To filter by start_date
            if start_date:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
                filtered_reviews = [
                    review for review in filtered_reviews
                    if datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") >= start_date
                ]

            # To filter by end_date
            if end_date:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
                filtered_reviews = [
                    review for review in filtered_reviews
                    if datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") <= end_date
                ]

            response_data = []
            for review in filtered_reviews:
                sentiment = self.analyze_sentiment(review['ReviewBody'])
                response_data.append({
                    "ReviewId": review['ReviewId'],
                    "ReviewBody": review['ReviewBody'],
                    "Location": review['Location'],
                    "Timestamp": review['Timestamp'],
                    "sentiment": sentiment
                })

            # To sort the data compound value in descending order
            response_data.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(response_data, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                content_length = int(environ.get("CONTENT_LENGTH", 0))
                body = environ["wsgi.input"].read(content_length).decode("utf-8")
                params = parse_qs(body)

                review_body = params.get("ReviewBody", [None])[0]
                location = params.get("Location", [None])[0]

                # To validate the body data
                if not review_body:
                    raise ValueError("missing review_body")
                elif not location:
                    raise ValueError("missing location")
                elif location not in predefined_location:
                    raise ValueError("invalid location")

                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp
                }

                reviews.append(new_review)

                response_body = json.dumps(new_review, indent=2).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])

                return [response_body]

            except Exception as e:
                response_body = json.dumps({"error": str(e)}, indent=2).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()