import requests

BASE_URL = "http://localhost:8085"

# Step 1: Register user with A/B Testing
def register_user():
    response = requests.post(f"{BASE_URL}/ab_testing", json={"email": "alice@example.com"})
    response.raise_for_status()
    data = response.json()
    print("Registered User:", data)
    return data["token"], data["experiment"]

# Step 2: Search movies by name
def search_movies(query="matrix", skip=0, limit=5):
    response = requests.get(f"{BASE_URL}/movies", params={
        "search": query,
        "skip": skip,
        "limit": limit
    })
    response.raise_for_status()
    movies = response.json()
    print("Movies Found:", movies)
    return movies

# Step 3: Send user interactions (rate a movie)
def send_rating(user_id, imdb_id, rating=5):
    payload = {
        "user_id": user_id,
        "interactions": [{
            "item_id": imdb_id,
            "rating": rating
        }]
    }
    response = requests.post(f"{BASE_URL}/profile/", json=payload)
    response.raise_for_status()
    print("Rating submitted.")

# Step 4: Get recommendations for user
def get_recommendations(user_id):
    response = requests.get(f"{BASE_URL}/recommendation/{user_id}")
    response.raise_for_status()
    recommendations = response.json()["data"]
    print("Recommendations:", recommendations)
    return recommendations


# --- Run the full flow ---
if __name__ == "__main__":
    user_id, experiment = register_user()
    movies = search_movies("inception")
    
    if movies:
        send_rating(user_id, movies[0]["imdb_id"], rating=4)
    
    get_recommendations(user_id)
