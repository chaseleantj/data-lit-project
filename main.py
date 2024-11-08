import os
import dotenv
import pandas as pd

from youtube_api import YouTubeAPI

dotenv.load_dotenv()

client = YouTubeAPI(api_key=os.getenv("GOOGLE_API_KEY"))
result = client.search_by_keywords(["machine learning"], limit=50)

result_df = result["video_df"]
result_df = result_df.drop(["caption", "description"], axis=1)
result_df.to_csv("output.csv")
