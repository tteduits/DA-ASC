import pyautogui
import time
import pandas as pd
from main import PDF_LINKS_EXCEL


CHECK_COORDINATES = False


def move_mouse(x, y, duration=1):
    pyautogui.moveTo(x, y, duration=duration)


# Click the mouse at coordinates (x, y)
def click_mouse(x, y):
    pyautogui.click(x, y)


# Function to get the screen dimensions
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
    pdf_links_df = pd.read_excel(PDF_LINKS_EXCEL)
    num_rows, num_columns = pdf_links_df.shape

    for row in range(num_rows):
        y = -1032 + 17.5 * row

        for col in range(num_columns):
            if col == 25 and row == 41:
                break

            x = 71 + col * 60
            move_mouse(x, y)
            time.sleep(1)

            click_mouse(x, y)
            time.sleep(1)
            click_mouse(x, y)
            time.sleep(1)

            # Skip verification and accept download
            move_mouse(x, y)
            time.sleep(1)
            click_mouse(1314, -636)
            time.sleep(1)
