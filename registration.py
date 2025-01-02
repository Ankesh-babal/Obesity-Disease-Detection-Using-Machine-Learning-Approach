import tkinter as tk
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import re

window = tk.Tk()
w, h = window.winfo_screenwidth(), window.winfo_screenheight()
window.geometry("%dx%d+0+0" % (w, h))
window.title("REGISTRATION FORM")
window.configure(background="#bfc9ca")

Fullname = tk.StringVar()
username = tk.StringVar()
Email = tk.StringVar()
password = tk.StringVar()
password1 = tk.StringVar()

# database code
db = sqlite3.connect('evaluation.db')
cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS registration"
               "(Fullname TEXT, username TEXT, Email TEXT, password TEXT)")
db.commit()

def password_check(passwd):
    SpecialSym = ['$', '@', '#', '%']
    val = True

    if len(passwd) < 6:
        print('Length should be at least 6')
        val = False

    if len(passwd) > 20:
        print('Length should not be greater than 20')
        val = False

    if not any(char.isdigit() for char in passwd):
        print('Password should have at least one numeral')
        val = False

    if not any(char.isupper() for char in passwd):
        print('Password should have at least one uppercase letter')
        val = False

    if not any(char.islower() for char in passwd):
        print('Password should have at least one lowercase letter')
        val = False

    if not any(char in SpecialSym for char in passwd):
        print('Password should have at least one of the symbols $@#')
        val = False

    if val:
        return val

def insert():
    fname = Fullname.get()
    un = username.get()
    email = Email.get()
    pwd = password.get()
    cnpwd = password1.get()

    with sqlite3.connect('evaluation.db') as db:
        c = db.cursor()

    # Find Existing username if any take proper action
    find_user = ('SELECT * FROM registration WHERE username = ?')
    c.execute(find_user, [(username.get())])

    # Check email
    regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    if re.search(regex, email):
        a = True
    else:
        a = False

    # Validation
    if fname.isdigit() or (fname == ""):
        ms.showinfo("Message", "Please enter a valid name")
    elif (email == "") or (a == False):
        ms.showinfo("Message", "Please enter a valid email")
    elif c.fetchall():
        ms.showerror('Error!', 'Username Taken. Try a Different One.')
    elif pwd == "":
        ms.showinfo("Message", "Please enter a valid password")
    elif pwd == "" or password_check(pwd) != True:
        ms.showinfo("Message", "Password must contain at least 1 uppercase letter, 1 symbol, and 1 number")
    elif pwd != cnpwd:
        ms.showinfo("Message", "Password and Confirm Password must be the same")
    else:
        conn = sqlite3.connect('evaluation.db')
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO registration(Fullname, username, Email, password) VALUES(?,?,?,?)',
                (fname, un, email, pwd))

            conn.commit()
            conn.close()
            ms.showinfo('Success!', 'Account Created Successfully!')
            window.destroy()
            from subprocess import call
            call(["python", "login.py"])

# Store references to the images to prevent garbage collection
window.images = {}

# For background Image
image2 = Image.open('C:/Users/ADMIN/Downloads/all data/all data/100% Obesity Disease Detection/100% Obesity Disease Detection/3.jpg')
image2 = image2.resize((650, 700), Image.Resampling.LANCZOS)
background_image = ImageTk.PhotoImage(image2)
window.images['background_image'] = background_image  # Store reference to prevent garbage collection

background_label = tk.Label(window, image=background_image)
background_label.place(x=650, y=0)

image2 = Image.open('C:/Users/ADMIN/Downloads/all data/all data/100% Obesity Disease Detection/100% Obesity Disease Detection/reg.jpg')
image2 = image2.resize((200, 200), Image.Resampling.LANCZOS)
reg_image = ImageTk.PhotoImage(image2)
window.images['reg_image'] = reg_image  # Store reference to prevent garbage collection

background_label = tk.Label(window, image=reg_image)
background_label.place(x=1000, y=20)


frame = tk.LabelFrame(window, text="", width=370, height=579, bd=4,
                      font=('times', 14, ' bold '), bg="grey")
frame.place(x=127, y=40)

l2 = tk.Label(frame, text="Name", width=4, font=("Times new roman", 15, "bold"),
              bg="grey", bd=5, fg="white")
l2.place(x=25, y=5)
t1 = tk.Entry(frame, textvar=Fullname, width=26, font=('', 15), bd=4, bg="white")
t1.place(x=26, y=35)

l4 = tk.Label(frame, text="Username", width=7, font=("Times new roman", 15, "bold"), fg="white", bg="grey")
l4.place(x=25, y=95)
t3 = tk.Entry(frame, textvar=username, width=26, font=('', 15), bd=4, bg="white")
t3.place(x=26, y=125)

l5 = tk.Label(frame, text="E-mail", width=5, font=("Times new roman", 15, "bold"), fg="white", bg="grey")
l5.place(x=23, y=185)
t4 = tk.Entry(frame, textvar=Email, width=26, font=('', 15), bd=4, bg="white")
t4.place(x=26, y=215)

l9 = tk.Label(frame, text="Password", width=7, font=("Times new roman", 15, "bold"), fg="white", bg="grey")
l9.place(x=24, y=275)
t9 = tk.Entry(frame, textvar=password, width=26, font=('', 15), show="*", bd=4, bg="white")
t9.place(x=26, y=305)

l10 = tk.Label(frame, text="Confirm Password", width=13, font=("Times new roman", 15, "bold"), fg="white", bg="grey")
l10.place(x=26, y=365)
t10 = tk.Entry(frame, textvar=password1, width=26, font=('', 15), show="*", bd=4, bg="white")
t10.place(x=26, y=395)

def log():
    window.destroy()
    from subprocess import call
    call(["python", "login.py"])

btn = tk.Button(frame, text="Sign up", bg="green", font=("times", 17), fg="white", width=22, bd=3, command=insert)
btn.place(x=25, y=455)

l10 = tk.Label(frame, text="Already have an account?", width=20, font=("Times new roman", 13), fg="white", bg="grey")
l10.place(x=35, y=525)

btn = tk.Button(frame, text="Sign in.", bg="grey", font=('times 15 bold underline'), fg="blue", bd=0, command=log)
btn.place(x=218, y=518)

window.mainloop()
