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

## Utility functions

To load an image:

```
from utils import file_utils

img_arr = file_utils.load_single_image("path/to/image.jpg")
```

To find the hue, saturation and lightness of an image:
```
from utils import file_utils, image_utils

hsv = image_utils.calculate_average_hsl_cv2(file_utils.load_single_image("pic.jpg"))
```
```
{'hue': 356.3891218844166, 'saturation': 0.86062485, 'lightness': 0.43709543}
```