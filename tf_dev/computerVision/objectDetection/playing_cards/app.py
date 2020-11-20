"""
Developer: vkyprmr
Filename: app.py
Created on: 2020-11-20, Fr., 13:57:45
"""
"""
Modified by: vkyprmr
Last modified on: 2020-11-20, Fr., 13:57:47
"""

from tkinter import *
from tkinter import ttk
import PIL
from PIL import ImageTk
from tkinter import messagebox



class PCobjdet:
    def __init__(self, master):
        self.master = master
        self.master.configure(background='white')
        self.masterFrame = Frame(self.master, background='white')
        self.masterFrame.pack(expand=1, fill='both')
        self.home()

    def home(self):
        self.leftFrame = Frame(self.masterFrame, background='white')
        self.leftFrame.pack(side='left', fill='y', expand=1)
        self.sep = ttk.Separator(self.masterFrame, orient='vertical')
        self.sep.pack(side='left', fill='y', expand=1)
        self.rightFrame = Frame(self.masterFrame, background='white')
        self.rightFrame.pack(side='left', fill='both', expand=1)
        self.nameFrame = Frame(self.rightFrame, background='white')
        self.nameFrame.pack()
        img = PIL.Image.open('../../../../Data/playing_cards/icon/stock-vector-joker-card-vector-background-131267243'
                             '.jpg')
        self.img = ImageTk.PhotoImage(img)
        self.appName = Label(self.leftFrame, text='Playing Card Detector', background='white', font=('Comic Sans MS', 12))
        self.appName.pack(fill='y', expand=1)
        self.appNamePost = Label(self.leftFrame, background='white')
        self.appNamePost.pack(fill='y', expand=1)
        img_label = Label(self.leftFrame, image=self.img, background='white')
        img_label.pack(fill='y', expand=1)




# Run
root = Tk()
app = PCobjdet(root)
root.wm_iconbitmap('../../../../Data/playing_cards/icon/stock-vector-joker-card-vector-background-131267243.ico')
root.wm_title('Playing Card Object Detector')
root.mainloop()