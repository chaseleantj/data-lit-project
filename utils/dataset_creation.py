import pandas as pd
import os
from file_utils import download_image

SIZE_OF_CATEGORY_DS = 10000

# change this
csv_files = ["baseball-sports.csv", "basketball-sports.csv", "cricket-sports.csv", "football-sports.csv", "hockey-sports.csv", "soccer-sports.csv", "tennis-sports.csv", "volleyball-sports.csv"]
output_file = "sports.csv"

# first check if there are at least SIZE_OF_CATEGORY_DS potential (if valid) unique data points (otherwise collection does not make sense)
merged_df = pd.DataFrame()
for file in csv_files:
    df = pd.read_csv(os.path.join("data", "requests", file))
    merged_df = pd.concat([merged_df, df])
ids = []
for index, row in merged_df.iterrows():
    ids.append(row["video_id"])
ids = set(ids)
num_of_dp = len(ids)
if num_of_dp >= SIZE_OF_CATEGORY_DS:
    print(f"Maximum number of potential unique data points: {num_of_dp}")
else:
    raise RuntimeError(f"There are only {num_of_dp} potential unique data points, so you cannot create a dataset with size {SIZE_OF_CATEGORY_DS}!")

row_dict = {}
line_counts = {file: 0 for file in csv_files}
num_of_lines = 0

for file in csv_files:
    df = pd.read_csv(os.path.join("data", "requests", file))
    for index, row in df.iterrows():
        image_url = row['thumbnail-url']
        views = row['viewCount']
        
        # check if thumbnail is accessible (try to download it for at most three times)
        image_received = False
        for i in range(0, 3):
            try:
                download_image(image_url)
                image_received = True
                break
            except:
                pass
        if not image_received:
            print(f"Failed to download image from {image_url}")
            continue

        # check if view count is valid
        try:
            views = int(views)
        except:
            print(f"Failed to get view count for video {row['video_id']}")
            continue

        # prevent duplicates
        key = row["video_id"]
        if key in row_dict:
            continue
        
        row_dict[key] = row
        num_of_lines += 1
        line_counts[file] += 1
        if num_of_lines % 100 == 0:
            print(f"{num_of_lines} data points collected so far.")

        if num_of_lines >= SIZE_OF_CATEGORY_DS:
            break

if num_of_lines < SIZE_OF_CATEGORY_DS:
    print(f"WARNING: Found only {num_of_lines} unique and valid data points, but {SIZE_OF_CATEGORY_DS} were requested!")
df = pd.DataFrame.from_dict(row_dict, orient='index')
df = df.drop(df.columns[0], axis=1)
df.reset_index(drop=True, inplace=True)
df.to_csv(os.path.join("data", "dataset", output_file), index=True)
with open(os.path.join("data", "dataset", "metadata.txt"), "a") as file:
    file.write(output_file + ": " + str(line_counts) + "\n")