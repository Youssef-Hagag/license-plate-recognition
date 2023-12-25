import PySimpleGUI as sg
from PIL import Image
import random
import time
import threading
from project import *

loading = False
progress_thread = None
image_thread = None

def ProgressBar():
  global loading
  i = 0
  while True:
    if loading:
      window['-RESULT-'].update("Processing" + '.' * (i % 3 + 1))
      window['-PROGRESS-'].update((i + 1) % 101, visible=True)
      i = (i + 1) % 101
      time.sleep(0.1)
    else:
      window['-PROGRESS-'].update(0, visible=False)
      i = 0

def image_processing():
  global loading
  loading = True
  try:
    image = Image.open(filename)
  except IOError:
    window['-RESULT-'].update("Cannot identify image file !")
    loading = False
    return
  resized_image = image.resize((500, 300))
  resized_image.save('resized_image.png')
  window['-IMAGE-'].update(filename='resized_image.png')
  window['-BROWSE-'].update(disabled=True)
  # The heavy computation should be done here
  # ----------------------------------------------------------------
  plate = plate_detection_using_contours(filename)
  letters = PlateToLetters(plate)

  #extract features from the resulting letters
  letterFeatures = extract_features(letters)

  #train the knn then predict the letters
  trainKnn()
  
  # ----------------------------------------------------------------

  # To be assigned in Image processing code
  loading = False
  window['-BROWSE-'].update(disabled=False)
  result = predictKnn(letterFeatures)
  LED = random.choice(['Allowed', 'Banned'])

  window['-RESULT-'].update(result)

  if LED == 'Allowed':
    window['-LED-'].update(background_color='green')
  else:
    window['-LED-'].update(background_color='red')
def waitOnThread():
  global progress_thread
  global image_thread
  while True:
    # Stop previous threads if they are running
    if progress_thread and progress_thread.is_alive():
      print("Stopping progress thread")
      progress_thread.join()
      progress_thread = None
    if image_thread and image_thread.is_alive():
      print("Stopping image thread")
      image_thread.join()
      image_thread = None
layout = [
  [
    sg.Text('Select an image:', background_color='white', text_color='black'),
    sg.Input(key='-FILE-' ,enable_events=True, background_color='white', text_color='black'),
    sg.FileBrowse(key='-BROWSE-', button_color=('black', 'gold'))
  ],
  [
    sg.Column([
      [sg.Image(key='-IMAGE-', size=(300, 300), background_color='white')],
      [sg.Frame('Processing Result', [
        [sg.Text('Processing result:', background_color='grey', text_color='black'),
        sg.Text('', key='-RESULT-', size=(20, 1), background_color='white', text_color='black',
              font=('Helvetica', 15))],
        [sg.Text('LED:', size=(10, 1), background_color='grey', text_color='black'),
        sg.Text('', key='-LED-', size=(5, 2), background_color='white', text_color='black')]
      ], background_color='grey', size=(500, 100))]
    ], background_color='white'),
  ],
  [
    sg.ProgressBar(100, orientation='h', size=(20, 20), key='-PROGRESS-')
  ]
]

window = sg.Window('Image Processing', layout, resizable=True, background_color='white')
# threading.Thread(target=waitOnThread, args=(), daemon=True).start()
while True:
  event, values = window.read()
  if event in (sg.WIN_CLOSED, 'Exit'):
    break
  if progress_thread == None:
    progress_thread = threading.Thread(target=ProgressBar, args=(), daemon=True)
    progress_thread.start()

  if event == '-FILE-':
    filename = values['-FILE-']
    if filename:
      try:

        # Image processing code
        progress_bar = window['-PROGRESS-']
        window['-LED-'].update('', background_color='white')
        image_thread = threading.Thread(target=image_processing, args=(), daemon=True)
        image_thread.start()
      except Exception as e:
        print(f"An error occurred: {e}")

window.close()
