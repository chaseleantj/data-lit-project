import os
import dotenv
import pandas as pd

from youtube_api import YouTubeAPI

dotenv.load_dotenv()

keywords = ["machine learning"]
num_results = 50
output_path = "output.csv"

client = YouTubeAPI(api_key=os.getenv("GOOGLE_API_KEY"))
result = client.search_by_keywords(keywords, limit=num_results)

result_df = result["video_df"]
result_df = result_df.drop(["caption", "description"], axis=1)
result_df.to_csv(output_path)

print("Results successfully saved to", output_path)
