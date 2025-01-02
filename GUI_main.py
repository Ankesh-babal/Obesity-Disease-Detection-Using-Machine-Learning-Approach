import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk

# Create the main application window
root = tk.Tk()
root.configure(background="brown")

# Set the window size to the full screen
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Detection System")

##### For background Image with Error Handling
try:
    # Provide the full path to your image file using forward slashes
    image2 = Image.open('1.png')
    image2 = image2.resize((w, h), Image.Resampling.LANCZOS)  # Resizing the image
    background_image = ImageTk.PhotoImage(image2)
except Exception as e:
    print(f"Error loading image: {e}")
    background_image = None

# Create a label to display the background image and store the reference if image loaded
if background_image:
    background_label = tk.Label(root, image=background_image)
    background_label.image = background_image  # Store a reference to prevent garbage collection
    background_label.place(x=0, y=0)  # Position the background image label

# Create a label for the title text
label_l1 = tk.Label(root, text="Obesity Disease Detection", font=("Times New Roman", 20, 'bold'),
                    background="white", borderwidth=5, relief='solid', fg="red", padx=50, pady=50)
label_l1.place(x=50, y=300)

##### Button Definitions
# Function for the Register button
def reg():
    from subprocess import call
    call(["python", "registration.py"])

# Function for the Login button
def log():
    from subprocess import call
    call(["python", "login.py"])

# Function to exit the program
def window():
    root.destroy()

# Create and place the Login button
button1 = tk.Button(root, text="LOGIN", command=log, width=14, height=1,
                    font=('times', 20, 'bold'), bg="#999999", fg="white")
button1.place(x=900, y=300)

# Create and place the Register button
button2 = tk.Button(root, text="REGISTER", command=reg, width=14, height=1,
                    font=('times', 20, 'bold'), bg="#999999", fg="white")
button2.place(x=900, y=400)

# Create and place the Exit button
button3 = tk.Button(root, text="Exit", command=window, width=14, height=1,
                    font=('times', 20, 'bold'), bg="red", fg="white")
button3.place(x=900, y=500)

# Start the Tkinter event loop
root.mainloop()
