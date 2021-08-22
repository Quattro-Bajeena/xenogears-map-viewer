import PySimpleGUI as sg
from pathlib import Path

from xgl import MapViewer, load_level


resource_folder = Path(__file__).parent / "Resources"
thumbnails_folder = resource_folder / "Thumbnails"
icon_file = resource_folder / "xenogears logo.ico"

def lanch_viewer(fileIndex):
    model = load_level(fileIndex)
    map_viewer = MapViewer(model)
    map_viewer.main_loop()

sg.theme('DarkBrown4') 
pictures_area = [ [] ]
i = 1
while i <= 729:
    pictures_area[-1].append(
        sg.Button(
            key = f"Level_{i}", 
            button_color=(sg.theme_background_color(), sg.theme_background_color()),
            image_filename = thumbnails_folder / f"LevelThumbnail_{i}.png",
            #image_size=(50, 50), 
            image_subsample=1,
            border_width=1
            ))
    
    if i % 9 == 0:
        pictures_area.append([])
    i+=1

layout = [  [sg.Text('Enter level index, from 1 to 729, or click on a thumbnail to view the level'), sg.Button("Controls")],
            [sg.InputText(key="FileIndex", size=(10,10)), sg.OK()],
            [sg.Column(pictures_area, size=(1325, 800), scrollable=True, vertical_scroll_only=True)],
            ]
    
  
# Create the Window
window = sg.Window("Xenogears Map Viewer", icon=icon_file).Layout(layout)
# Event Loop to process "events"
while True:             
    event, values = window.read()
    #print(event, values)
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event == 'OK':
        try:
            fileIndex = int(values['FileIndex'])
            if fileIndex < 1 or fileIndex > 729:
                raise ValueError

            lanch_viewer(fileIndex)
        except:
            sg.Popup("Wrong value")
    elif str(event).split('_')[0] == "Level":
        fileIndex = int(str(event).split('_')[1])
        lanch_viewer(fileIndex)
    elif event == 'Controls':
        control_info = '''
        WASD - movement
        Mouse and arrows - looking around
        Mouse scroll - change movement speed
        Shift - go faster
        E - toggle wireframe
        ESC - close level viewer
        '''
        sg.Popup(control_info, title="Controls")


window.close()



