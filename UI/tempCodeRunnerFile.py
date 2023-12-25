frame = sg.Frame('Processing Result', layout=[
        [sg.Text('Processing result:', background_color='white', text_color='black'), sg.Text('', key='-RESULT-', size=(20, 1), background_color='white', text_color='black')],
        [sg.Text('LED:', size=(10, 1), background_color='white', text_color='black'), sg.Text('', key='-LED-', size=(5, 1), background_color='white', text_color='black')]
      ], background_color='grey')