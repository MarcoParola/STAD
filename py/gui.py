from tkinter import *
import time
import threading
import logging



def thread_function():
    print("Thread starting")
    x = x_cordinate.get()
    y = y_cordinate.get()
    
    # LOAD CLASSIFIER
    
    while TRUE:
        time.sleep(2)
        print('ciao')
        
    print("Thread finishing")




def startFunction(btn, string):
    x = threading.Thread(target=thread_function)
    x.start()

def stopFunction():
    print(x_cordinate.get())





def main_screen():
    global screen 
    screen = Tk()
    screen.geometry("700x700")
    screen.title("STAD")
    
    global x_cordinate, y_cordinate, km
    
    x_cordinate = StringVar()
    y_cordinate =StringVar()
    km = StringVar()
    
    Label(text="x", borderwidth=10).grid( row=1, column=1)
    x = Entry(textvariable = x_cordinate).grid( row=1, column=2)
    
    Label(text="y", borderwidth=10).grid( row=2, column=1)
    y = Entry(textvariable = y_cordinate).grid( row=2, column=2) 
    
    Label(text="km", borderwidth=10).grid( row=3, column=1)
    km = Entry(textvariable = km).grid( row=3, column=2) 
    
    text = Text().grid(row=5, column=1, columnspan=3)
    
    btn_start = Button(text="START", height="2", width="30", command=lambda: startFunction(btn_start, 'pippo'))
    btn_start.grid( row=6, column=2) 
    
    btn_stop = Button(text="STOP", height="2", width="30", command=stopFunction)
    btn_stop.grid( row=6, column=3) 
    btn_stop.config(state='disable')
    
    screen.mainloop()
    

main_screen()



