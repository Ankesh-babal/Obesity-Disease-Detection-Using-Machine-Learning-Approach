import tkinter as tk
from PIL import Image, ImageTk

##############################################+=============================================================
root = tk.Tk()
root.configure(background="brown")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Detection System")

# ++++++++++++++++++++++++++++++++++++++++++++
##### For background Image
image2 = Image.open('C:/Users/ADMIN/Downloads/all data/all data/100% Obesity Disease Detection/100% Obesity Disease Detection/2.jpg')
image2 = image2.resize((w, h), Image.Resampling.LANCZOS)  # Use Resampling.LANCZOS

background_image = ImageTk.PhotoImage(image2)

# Store a reference to prevent garbage collection
root.background_image = background_image

background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0)

label_l1 = tk.Label(root, text="Obesity Disease Detection", font=("Times New Roman", 20, 'bold'),
                    background="white", borderwidth=5, relief='solid', fg="red", padx=50, pady=50)
label_l1.place(x=50, y=300)

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def reg():
    from subprocess import call
    call(["python", "check.py"])

def log():
    from subprocess import call
    call(["python", "testing.py"])

def window():
    root.destroy()

button1 = tk.Button(root, text="SVM-model Test Obesity", command=log, width=20, height=1,
                    font=('times', 20, ' bold '), bg="black", fg="white")
button1.place(x=750, y=250)

button2 = tk.Button(root, text="other-Model Test Obesity", command=reg, width=20, height=1,
                    font=('times', 20, ' bold '), bg="black", fg="white")
button2.place(x=750, y=350)

button3 = tk.Button(root, text="Exit", command=window, width=20, height=1,
                    font=('times', 20, ' bold '), bg="red", fg="white")
button3.place(x=750, y=450)

root.mainloop()
