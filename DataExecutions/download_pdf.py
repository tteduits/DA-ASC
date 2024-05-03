import pyautogui
import time
import pandas as pd
import os
import shutil
from pathlib import Path
from main import PDF_LINKS_EXCEL, COL_LENGTH, ROW_LENGTH, EXCEL_FOLDER, EXTERNAL_DISK
from DataFunctions.pdf_read_functions import clean_full_text


CHECK_COORDINATES = False
full_text_df = pd.DataFrame()


def get_screen_dimensions(screen_index):
    screen = pyautogui.size()  # Get the primary screen size
    if screen_index >= pyautogui.count_monitors():
        raise ValueError("Invalid screen index")
    screen_rect = pyautogui.getAllMonitors()[screen_index]
    return screen_rect.left, screen_rect.top, screen_rect.width, screen_rect.height


if CHECK_COORDINATES:
    try:
        while True:
            x, y = pyautogui.position()
            position_str = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
            try:
                screen_index = pyautogui.mouseInfo()['screen']
                left, top, width, height = get_screen_dimensions(screen_index)
                if left <= x < left + width and top <= y < top + height:
                    pixel_color = pyautogui.screenshot(region=(left, top, width, height)).getpixel((x - left, y - top))
                    position_str += ' RGB: (' + str(pixel_color[0]).rjust(3)
                    position_str += ', ' + str(pixel_color[1]).rjust(3)
                    position_str += ', ' + str(pixel_color[2]).rjust(3) + ')'
                else:
                    position_str += ' RGB: (out of screen)'
            except ValueError as e:
                position_str += f' Error: {e}'
            print(position_str, end='')
            print('\b' * len(position_str), end='', flush=True)  # Erase the previous position
    except KeyboardInterrupt:
        print('\nDone.')

if not CHECK_COORDINATES:

    for csv_number in range(43, 44):
        url_df = pd.read_excel(PDF_LINKS_EXCEL + 'pdf_file_' + str(csv_number) + '.xlsx')
        data_combined = pd.read_excel(EXCEL_FOLDER + 'CombinedData\\combined_data' + str(csv_number) + '.xlsx')
        # Random click in python (needed)
        pyautogui.moveTo(451, 215)
        pyautogui.click(451, 215)

        for col in range(2, 3):
            x = 2480 + col * 84

            for row in range(15,16):
                y = 378 + 25 * row

                pyautogui.moveTo(x, y)
                pyautogui.click(x, y)
                time.sleep(0.5)
                pyautogui.click(x, y)
                time.sleep(1)

                url_link = url_df.iloc[row, col]
                found = False
                while not found:
                    for index, row_items in data_combined.iterrows():
                        most_recent_download_path = max((os.path.join(os.path.expanduser("~/Downloads"), f) for f in os.listdir(os.path.expanduser("~/Downloads")) if os.path.isfile(os.path.join(os.path.expanduser("~/Downloads"), f))), key=os.path.getmtime)
                        if 'pdf' not in most_recent_download_path:
                            break
                        if row_items['URL'] == url_link:
                            shutil.move(most_recent_download_path, EXTERNAL_DISK)
                            filename = os.path.basename(most_recent_download_path)
                            new_filename = url_link.split("clips/", 1)[-1] if "clips/" in url_link else ""
                            new_filepath = os.path.join(EXTERNAL_DISK, new_filename)
                            os.rename(os.path.join(EXTERNAL_DISK, filename), new_filepath)

                            found = True

                            break
            print('This is col: ' + str(col))

        pyautogui.click(5563, 29)

        time.sleep(1)
        print('This is csv number : ' + str(csv_number))
