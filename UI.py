import PySimpleGUI as sg
from PIL import Image
import random
import time
import threading
from project import *

loading = False
progress_thread = None
image_thread = None

arr = [
    ['9', '9' ,'9' ,'9' ,'yeh', 'sen', 'sen'],
    ['1', '1', '1', 'lam', 'tah', 'beh'],
    ['1', '1', '1', 'gem', 'reh', 'gem'],
    ['1', '3', '6', '5', 'fih', 'waw'],
    ['8', '7', '8', '4', 'sad', 'sad']
]

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
  try:
      result = main(filename)  # Your main function call that might raise an exception
      print(result)
  except Exception as e:  # Catch any exception and store it in variable 'e'
      window['-RESULT-'].update("Plate not found")
      loading = False
      window['-BROWSE-'].update(disabled=False)
      return
  
  # ----------------------------------------------------------------

  # To be assigned in Image processing code
  loading = False
  window['-BROWSE-'].update(disabled=False)
  
  LED = 'Banned'
  
  for i in range(len(arr)):
      if np.array_equal(arr[i], result):
        LED = 'Allowed'
        break
        
        

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
        sg.Text('', key='-RESULT-', size=(30, 1), background_color='white', text_color='black',
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
