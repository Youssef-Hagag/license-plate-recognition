import PySimpleGUI as sg
from PIL import Image
import random

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
        [sg.Text('Processing result:', background_color='grey', text_color='black'), sg.Text('', key='-RESULT-', size=(20, 1), background_color='white', text_color='black')],
        [sg.Text('LED:', size=(10, 1), background_color='grey', text_color='black'), sg.Text('', key='-LED-', size=(5, 1), background_color='white', text_color='black')]
      ], background_color='grey',size=(500, 100))]
  ], background_color='white'),
]
]

window = sg.Window('Image Processing', layout, resizable=True, background_color='white')

while True:
  event, values = window.read()

  if event == sg.WINDOW_CLOSED:
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
        # ...


        # To be assigned in Image processing code 
        result = random.choice(['123 ا ب ج', '345 ق ص ر', '678 ث ت ن', '910 ح خ ذ', '1112 ز س ش', '1314 ض ط ظ', '1516 ع غ ف', '1718 ق ك ل', '1920 م ن ه', '2122 و ي'])
        LED = random.choice(['Allowed', 'Banned'])

        window['-RESULT-'].update(result)

        if LED == 'Allowed':
          window['-LED-'].update(background_color='green')
        else:
          window['-LED-'].update(background_color='red')

      except Exception as e:
        print(f"An error occurred: {e}")

window.close()
