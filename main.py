import os
import dotenv
import pandas as pd

from youtube.youtube_api import YouTubeAPI

dotenv.load_dotenv()

keywords = input("Keywords for your search: ")

while True:
    try:
        num_results = int(input("Number of results to show (1-10000): "))
        if num_results < 1 or num_results > 10000:
            raise RuntimeError()
        break
    except:
        print("Please enter a valid integer between 1 and 10000!")

output_path = os.path.join("data", input("Name of output .csv file (existing ones will be overwritten): ") + ".csv")

client = YouTubeAPI(api_key=os.getenv("GOOGLE_API_KEY"))
result = client.search_by_keywords(keywords, limit=num_results)

result_df = result["video_df"]
result_df = result_df.drop(["caption", "description"], axis=1)
result_df.to_csv(output_path)

print("Results successfully saved to", output_path)
