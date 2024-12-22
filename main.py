import os
import dotenv
import pandas as pd

from youtube.youtube_api import YouTubeAPI

youtube_category_ids = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    19: "Travel & Events",
    20: "Gaming",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "How-to & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism"
}

dotenv.load_dotenv()

keywords = input("Keywords for your search (leave blank if no keywords should be used): ")

while True:
    try:
        category = input("Video category ID for your search (leave blank to accept all categories): ")
        if not category:
            break
        category = int(category)
        if category and category not in youtube_category_ids:
            raise RuntimeError()
        break
    except:
        print("Please provide a valid category ID or leave it blank!")

while True:
    try:
        num_results = int(input("Number of results to show (1-10000): "))
        if num_results < 1 or num_results > 10000:
            raise RuntimeError()
        break
    except:
        print("Please enter a valid integer between 1 and 10000!")

output_csv = input("Name of output .csv file (existing ones will be overwritten): ") + ".csv"
output_path = os.path.join("data", output_csv)

client = YouTubeAPI(api_key=os.getenv("GOOGLE_API_KEY"))
result = client.search_request(keywords, category, limit=num_results)

result_df = result["video_df"]
result_df = result_df.drop(["caption", "description"], axis=1)
result_df.to_csv(output_path)

with open(os.path.join("data", "request-history.txt"), 'a') as file:
    file.write(output_csv + ",    keywords: " + keywords + ",    categoryFilter: " + str(category) + ",    numberOfResults: " + str(num_results) + "\n")

print("Results successfully saved to", output_path)
