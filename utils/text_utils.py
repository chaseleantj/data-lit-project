import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import warnings
from Project_TuebingenDL_WS24_25.utils.file_utils import download_image
import pytesseract
import keras_ocr
import easyocr
from easyocr.easyocr import all_lang_list
import cv2
from typing import List, Dict, Union, Any, Tuple
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
import pycountry


def process_row(row: pd.Series, add_language: bool, add_text: bool) -> Dict[str, Any]:
    """Recognise text from a single row of data."""

    title = row['title']
    image_url = row['thumbnail-url']
    detected_languages = detect_languages(title)
    
    # process image
    try:
        image = download_image(image_url)
        recognized_text = detect_text(
            np.array(image),
            lang=detected_languages,
            method=['pytesseract', 'keras_ocr', 'easyocr']
        )
        
        result = {'text_presence': int(sum(len(recognized_text[method]) > 0 
                                for method in recognized_text) > 1)}
    except Exception as e:
        print(f"Error processing image: {e}")
        recognized_text = {'keras_ocr': []}
        result = {'text_presence': pd.NA}
        
    if add_language:
        result['lang'] = detected_languages
    if add_text:
        result['recognized_text'] = recognized_text['keras_ocr']
        
    return result



def add_text_columns(dataframe: pd.DataFrame,
                     add_language: bool = True, add_text: bool = True) -> pd.DataFrame:
    """
    Add binary column (defalult) text_presence, and additionally recognized text and languages columns to the dataframe.
    Arsgs:
        dataframe (pd.DataFrame): dataframe with information about videos
        add_language (bool): add "lang" with recognized languages based on title, default=True
        add_text (bool): add "recognized_text" column with Keras OCR output, default=True
    Returns: dataframe
    """

    if not all(col in dataframe.columns for col in ['title', 'thumbnail-url']):
        raise ValueError('Dataframe should contain columns "title" and "thumbnail-url"')

    df = dataframe.copy()
    df['text_presence'] = pd.NA
    if add_language:
        df['lang'] = pd.NA
    if add_text:
        df['recognized_text'] = pd.NA

    # process each row|
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        result = process_row(row, add_language, add_text)
        for col, value in result.items():
            df.at[idx, col] = value

    return df


def detect_text(image: Union[str, np.ndarray], # | works only in Python 3.10+
                lang: Union[str, List[str]] = None,
                plot: bool = False, show_image: bool = False,
                method: Union[str, List[str]] = "pytesseract") -> Dict[str, List[str]]:
    """
    Detect text on the given image using given OCR method(s), return lists of detected "words"
    for each detection method. For more details on "words" definition see function clear_text below.
    Args:
        image (str or np.ndarray): Path to the image file or image as numpy array
        method (str): Name of OCR method to use or list of names
                      ('pytesseract', 'keras_ocr', 'easyocr', 'TrOCRProcessor')
    Returns:
        Dict[str, List[str]]: Dictionary with OCR method names as keys and lists of detected "words" as values
    """

    # setting directory to save images
    if plot and not os.path.exists('output_images'):
        os.makedirs('output_images')

    # methods check
    available_methods = {'pytesseract': run_pytesseract,
                        'keras_ocr': run_keras_ocr,
                        'easyocr': run_easy_ocr,
                        'TrOCRProcessor': run_trocr}
    if isinstance(method, str):
        method_list = [method]
    elif isinstance(method, list):
        method_list = method
    else:
        raise ValueError("Method should be either a string or a list of strings")
    if not all([m in available_methods.keys() for m in method_list]):
        raise ValueError("Check available methods!")
    
    # download and preprocess image
    if isinstance(image, str):
        image = preprocess_image(image)
    elif isinstance(image, np.ndarray):
        image = image
    else:
        raise ValueError("Image should be either path to the image file or image as numpy array")
    
    # run recognition
    rec_text_dict = dict()
    for ocr_method in method_list:
        rec_text = available_methods[ocr_method](image, plotted=plot, show_image=show_image, lang=lang)
         # clear recognized text
        rec_text_dict[ocr_method] = clear_text(rec_text)
        
    return rec_text_dict


# @time_it
def run_pytesseract(image: np.ndarray, lang: List[str] = None,
                    config: str = r'--oem 3',
                    plotted: bool = False, show_image: bool = False) -> List[str]:
    """Recognize text on the given image using pytesseract OCR."""

    '''
    Checking language, accepts ISO 639-2 codes (3 letters). 
    It is important to keep the order of languages, since (1) they are sorted by probability and
    (2) OCRs prioritize the first language in the list.
    '''
    available_languages = set(pytesseract.get_languages(config=''))
    if lang is not None:
        langs_list = ['eng'] if 'eng' in lang else []
        for lang_code in lang:
            # convert 2 letter ISO 639-1 to 3 letter ISO 639-2
            if len(lang_code) == 2:
                lang_code = lang_convert_2to3(lang_code)
            if lang_code in available_languages:
                langs_list.append(lang_code)
            else:
                warnings.warn(f"Unknown language: {lang_code}, it will be ignored. Check the spelling or install the language data.")
    else:
        langs_list = ['eng']

    results = pytesseract.image_to_data(image, output_type='dict',
                                        lang='+'.join(langs_list), config=config)

    if plotted:
        img = image.copy()
        for i in range(len(results["text"])):
            if results["text"][i].strip():  # ignore empty text
                text = results["text"][i]
                x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
                # draw boxes
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                # add text
                img = cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 1, cv2.LINE_AA)

        if show_image:
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            filename = check_filename(os.path.join('output_images', 'pytesseract_output.jpg'))
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f'Image with bounding boxes was saved as {filename}')

    return results['text']


# @time_it
def run_keras_ocr(image: np.ndarray, lang: Union[str, List[str]] = None, # not used in the function
                  plotted: bool = False, show_image: bool = False) -> List[str]:
    """Recognize text on the given image using keras-ocr OCR."""

    pipeline = keras_ocr.pipeline.Pipeline()
    results = pipeline.recognize([image]) # list of (text, box) tuples
    res_text = [results[0][i][0] for i in range(len(results[0]))]

    if plotted:
        img = image.copy()
        for text, box in results[0]:
            # draw boxes
            pts = box.astype(int)
            img = cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # add text
            x, y = pts[0]
            img = cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 1, cv2.LINE_AA)
        if show_image:
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            filename = check_filename(os.path.join('output_images','kerasocr_output.jpg'))
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f'Image with bounding boxes was saved as {filename}')

    return res_text


# @time_it
def run_easy_ocr(image: np.ndarray, lang: Union[str, List[str]] = None,
                  plotted: bool = False, show_image: bool = False) -> List[str]:
    """Recognize text on the given image using EasyOCR."""

    # check languages, accepts ISO 639-1 codes (2 letters) or random 2-3 char names... :(
    langs_list = ['en']
    if lang is not None:
        for language in lang:
            if language == 'en':
                continue
            if (language not in all_lang_list) and (lang_convert_2to3(language) not in all_lang_list):
                warnings.warn(f"Unknown language: {language}, it will be ignored. Check the spelling or install the language data.")
            elif language == 'zh-cn':
                langs_list += ['ch_sim', 'ch_tra', 'en']
            else:
                langs_list.append(language)

    try:
        reader = easyocr.Reader(langs_list)
    except ValueError as e:
        if "is only compatible with English" in str(e):
            reader = easyocr.Reader(langs_list[:2])
        else:
            warnings.warn(e)

    results = reader.readtext(image)
    results_text = [result[1] for result in results]

    if plotted:
        img = image.copy()
        for box,text,prob in results:
            # draw boxes
            pts = np.array(box).astype(int)
            img = cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # add text
            x, y = pts[0]
            img = cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 1, cv2.LINE_AA)

        if show_image:
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            filename = check_filename(os.path.join('output_images','easyocr_output.jpg'))
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f'Image with bounding boxes was saved as {filename}')

    return results_text


# @time_it
def run_trocr(image, plotted: bool = False, lang=None):
    # Implement the function or remove it if not needed
    raise NotImplementedError("run_trocr function is not implemented yet")


def check_filename(filename):
    """Function to check if the plot with the given filename already exists."""
    if os.path.exists(filename):
        name, ext = os.path.splitext(filename)
        i = 1
        while True:
            new_filename = f'{name}_{i}{ext}'
            if not os.path.exists(new_filename):
                filename = new_filename
                break
            i += 1
        return new_filename
    else: 
        return filename


def clear_text(text_list: List[str]) -> List[str]:
    """
    Clear strings in the list of recognized text to include only strings of
    alphanumeric characters with length >1 symbols (="word").
    """
    alphanumeric = [re.sub(r'[^a-zA-Z0-9\s]', '', line).strip() for line in text_list if line.strip()]
    # remove only numeric strings and strings with length < 2
    words = [s for s in alphanumeric if len(s) > 1 and re.search(r"[a-zA-Z]", s)]
    return words


# MOVE TO FILE_UTILS?
def preprocess_image(image_path: str, resize: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Load image from the given path and preprocess it by resizing to the given shape.
    Args:
        image_path (str): Path to the image file
        resize (tuple[int, int]): Shape to resize the image
    Returns:
        np.ndarray: Preprocessed image
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize)

    return image


def detect_languages(title: str) -> List[str]:
    """
    Detect language of the given text (video title).
    Returns list of detected languages as ISO 639-1 codes (2 letters).
    """

    try:
        lang_probabilities = detect_langs(title)
    except LangDetectException:
        print('No letters detected in the text!')
        return []
    # get languages with probability > 0.1; not that many languages actually so no need to filter
    # final_list = [lang.lang for lang in lang_probabilities if lang.prob > 0.1]
    final_list = [lang.lang for lang in lang_probabilities]
    if len(final_list) < 1:
        final_list = lang_probabilities[0]
    return final_list


def lang_convert_2to3(language_code: str) -> str:
    """Convert 2 letter ISO 639-1 to 3 letter ISO 639-2 """

    try:
        lang = pycountry.languages.get(alpha_2=language_code)
        return lang.alpha_3 if lang else language_code
    except AttributeError:
        warnings.warn(f"Could not convert language code {language_code} to 3 letter ISO 639-2")
        return language_code
