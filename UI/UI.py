import PySimpleGUI as sg
from PIL import Image
import random
import time
import threading

loading = False
def ProgressBar():
  global loading
  i=0;
  while True:
    if loading:
        window['-RESULT-'].update("Processing"+'.'*(i%3+1))
        window['-PROGRESS-'].update((i + 1)%101,visible=True)
        i = (i + 1)%101
        time.sleep(0.1)
    else: 
      window['-PROGRESS-'].update(0, visible=False);
      i=0;

def image_processing():
  global loading
  loading = True

  # The heavy computation should be done here 
  #----------------------------------------------------------------
  time.sleep(2)
  #----------------------------------------------------------------

  # To be assigned in Image processing code
  loading = False
  result = random.choice(['123 ا ب ج', '345 ق ص ر', '678 ث ت ن', '910 ح خ ذ', '1112 ز س ش', '1314 ض ط ظ','1516 ع غ ف', '1718 ق ك ل', '1920 م ن ه', '2122 و ي'])
  LED = random.choice(['Allowed', 'Banned'])

  window['-RESULT-'].update(result)

  if LED == 'Allowed':
    window['-LED-'].update(background_color='green')
  else:
    window['-LED-'].update(background_color='red')

layout = [
  [
    sg.Text('Select an image:', background_color='white', text_color='black'),
    sg.Input(key='-FILE-', enable_events=True, background_color='white', text_color='black'),
    sg.FileBrowse(button_color=('black', 'gold'))
  ],
  [
    sg.Column([
      [sg.Image(key='-IMAGE-', size=(300, 300), background_color='white')],
      [sg.Frame('Processing Result', [
        [sg.Text('Processing result:', background_color='grey', text_color='black'),
        sg.Text('', key='-RESULT-', size=(20, 1), background_color='white', text_color='black', font=('Helvetica', 15))],
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

while True:
  event, values = window.read()
  if event in (sg.WIN_CLOSED, 'Exit'):
    break

  if event == '-FILE-':
    filename = values['-FILE-']
    if filename:
      try:
        image = Image.open(filename)
        resized_image = image.resize((500, 300))
        resized_image.save('resized_image.png')
        window['-IMAGE-'].update(filename='resized_image.png')

        # Image processing code
        progress_bar = window['-PROGRESS-']
        window['-LED-'].update('', background_color='white')
        threading.Thread(target=image_processing, args=(), daemon=True).start()
        threading.Thread(target=ProgressBar, args=(), daemon=True).start()
      except Exception as e:
        print(f"An error occurred: {e}")

window.close()
