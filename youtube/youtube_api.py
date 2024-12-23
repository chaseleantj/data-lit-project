import json
import os
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from joblib import Memory
from joblib.memory import expires_after
import datetime
import math


cachedir = './cache'
if not os.path.exists(cachedir):
    os.makedirs(cachedir)
memory = Memory(cachedir, verbose=0)

cache_validation_callback = expires_after(days=1)

RESULT_TEMPLATE = {
            "ok": True,
            "error": None,
            "warning": None,
            "video_df": pd.DataFrame()
        }


class YouTubeAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key if api_key else os.getenv('GOOGLE_API_KEY')
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

        self.get_keyword_video_ids = memory.cache(self.get_kwvids, ignore=['self'], cache_validation_callback=cache_validation_callback)
        self.get_video_details = memory.cache(self.get_vdetails, ignore=['self'], cache_validation_callback=cache_validation_callback)

    def get_one_cstats(self, channel_id):
        request = self.youtube.channels().list(
            part="snippet,contentDetails,statistics",
            id=channel_id
        )
        response = request.execute()
        channels_data = []
        if response.get('items'):
            for item in response['items']:
                data={
                    'ProfilePic': item['snippet']['thumbnails']['default']['url'],
                    'ChannelName': item['snippet']['title'],
                    'Subscribers': item['statistics']['subscriberCount'],
                    'Views': item['statistics']['viewCount'],
                    'TotalViews': item['statistics']['videoCount'],
                    'PlaylistID': item['contentDetails']['relatedPlaylists']['uploads'],
                }
                channels_data.append(data)
        df=pd.DataFrame(channels_data)
        return df
    
    
    def get_kwvids(self, keywords, category, limit=100):
        next_page_token = None
        video_ids = []

        # do multiple queries with different date filters for larger requests due to the limited page token count
        max_results_per_date = 450
        date_parts = math.ceil(limit / max_results_per_date)
        
        # consider videos since 2015 (ten years)
        start_date = datetime.datetime(2015, 1, 1)
        end_date = datetime.datetime.now()
        total_days = (end_date - start_date).days
        days_per_part = total_days // date_parts
        date_ranges = []
        for i in range(date_parts):
            part_start_date = start_date + datetime.timedelta(days=i * days_per_part)
            part_end_date = part_start_date + datetime.timedelta(days=days_per_part - 1)
            if part_end_date > end_date:
                part_end_date = end_date
            date_ranges.append((part_start_date, part_end_date))

        counter = 0
        date_range_idx = 0
        video_count_for_this_date = 0
        while counter < limit:
            prev_idx = date_range_idx
            date_range_idx = counter // max_results_per_date
            current_start_date, current_end_date = date_ranges[date_range_idx]
            video_count_for_this_date = 0 if prev_idx != date_range_idx else video_count_for_this_date

            params = {
                "part": "snippet",
                "type": "video",
                "maxResults": 50 if limit > 50 else limit,
                "pageToken": next_page_token if next_page_token else None,
                "publishedAfter": current_start_date.isoformat("T") + "Z",
                "publishedBefore": current_end_date.isoformat("T") + "Z"
            }

            if keywords:
                params["q"] = keywords
                    
            if category:
                params["videoCategoryId"] = str(category)

            request = self.youtube.search().list(**params)
            response = request.execute()
            next_page_token = response.get('nextPageToken', None)

            for item in response['items']:
                video_count_for_this_date += 1
                video_ids.append(item['id']['videoId'])

            counter += 50

            # if we still need results for this date range, but there is no next page
            if not next_page_token and counter < (date_range_idx + 1) * max_results_per_date:
                print(f"Found only {video_count_for_this_date} of {max_results_per_date} requested videos between {current_start_date.date()} to {current_end_date.date()} for the specified query!")
                counter = (date_range_idx + 1) * max_results_per_date
        
        video_ids = list(set(video_ids)) # remove duplicates
        return video_ids
    

    def get_vdetails(self, video_ids):
        all_info=[]
        for i in range(0, len(video_ids), 50):
            request = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(video_ids[i:i+50])
            )
            response = request.execute()
            for video in response['items']:
                keepstats={'snippet':['channelTitle','title','description','tags','publishedAt'],
                    'statistics':['viewCount','likeCount','commentCount','subscriberCount'],
                    'contentDetails':['duration','definition','caption']
                    }
                video_info={}
                video_info['video_id']=video['id']
                for k in keepstats.keys():
                    for v in keepstats[k]:
                        try:
                            video_info[v]=video[k][v]
                        except:
                            video_info[v]=None
                video_info["thumbnail-url"] = video["snippet"]["thumbnails"]["high"]["url"]
                all_info.append(video_info)
        return (pd.DataFrame(all_info))
    

    def get_video_details_from_playlists(self, playlist_ids, limit=100):
        df = pd.DataFrame()
        for playlist_id in playlist_ids:
            video_ids = self.get_single_channel_video_ids(playlist_id, limit)
            single_df = self.get_video_details(video_ids)
            df = pd.concat([df, single_df], ignore_index=True)
        return df


    def search_request(self, keywords, category, limit=100):

        result = RESULT_TEMPLATE.copy()

        try:
            video_ids = self.get_keyword_video_ids(keywords, category, limit)
            result["video_df"] = self.get_video_details(video_ids)
        except HttpError as e:
            result["ok"] = False
            error_content = json.loads(e.content.decode())
            if 'error' in error_content and 'message' in error_content['error']:
                result["error"] = error_content['error']['message']
            else:
                result["error"] = str(e)

        return result
