from tkinter import *

def helloFunction():
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
    
    btn_start = Button(text="START", height="2", width="30", command=helloFunction)
    btn_start.grid( row=6, column=3) 
    
    screen.mainloop()
    

main_screen()



