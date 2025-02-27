# Data Literacy Project

Uses the YouTube API to analyze statistics of YouTube videos.

<a href="https://docs.google.com/presentation/d/1phEsqlQO2012Iom6U7mQYLvB4e3FpMUX/edit#slide=id.g31862cc1a42_0_1">Link to slides on Google Drive.</a>

## How to access the API

- Get an API key from the <a href="https://console.cloud.google.com/">Google Cloud Console</a> by enabling the YouTube Data API v3.

- Clone the repositiory.

```
git clone origin https://github.com/chaseleantj/Project_TuebingenDL_WS24_25
```

- Install the requirements.

```
pip install -r requirements.txt
```

- Create a `.env` file in the root directory and store your API key there. See `.env_example`.

- Run `main.py` and follow the instructions to obtain the top YouTube search results for the specified request.

## Common YouTube video category IDs

| **Category**            | **ID** |
|-------------------------|------:|
| Film & Animation        |      1 |
| Autos & Vehicles        |      2 |
| Music                   |     10 |
| Pets & Animals          |     15 |
| Sports                  |     17 |
| Travel & Events         |     19 |
| Gaming                  |     20 |
| People & Blogs          |     22 |
| Comedy                  |     23 |
| Entertainment           |     24 |
| News & Politics         |     25 |
| How-to & Style          |     26 |
| Education               |     27 |
| Science & Technology    |     28 |
| Nonprofits & Activism   |     29 |

## Utility functions

To load an image:

```
from utils import file_utils

img_arr = file_utils.load_single_image("path/to/image.jpg")
```

To find the hue, saturation and lightness of an image:
```
from utils import file_utils, image_utils

hsv = image_utils.calculate_image_features(file_utils.load_single_image("path/to/image.jpg"))
```

Example output:
```
{'hue': np.float32(62.17), 'saturation': np.float32(0.6006), 'lightness': np.float32(0.5476), 'contrast': np.float32(0.1292), 'sharpness': np.float32(117.7611)}
```
To detect faces in an image:
```
from utils import face_detection_utils

faces = face_detection_utils.detect_faces("path/to/image.jpg")
num_faces = face_detection_utils.count_faces(faces)
```
