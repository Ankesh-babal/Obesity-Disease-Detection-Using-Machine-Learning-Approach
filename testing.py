import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load the dataset
df = pd.read_csv('train1.csv')

# Encode categorical features
label_encoders = {}
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'MTRANS']
for column in categorical_features:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Encode the target variable
le_target = LabelEncoder()
df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])

# Separate features and labels
X = df.drop('NObeyesdad', axis=1).values
y = df['NObeyesdad'].values

# Convert labels to categorical (for multi-class classification)
y = to_categorical(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the training and testing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Keep this 2D
X_test = scaler.transform(X_test)  # Keep this 2D

# Reshape the data to 3D for the CNN
X_train = np.expand_dims(X_train, axis=2)  # Reshape to 3D
X_test = np.expand_dims(X_test, axis=2)  # Reshape to 3D

# Load the saved model from file
model = load_model('cnn_obesity_model1.h5')


from tkinter import *
from tkinter import ttk
import tkinter as tk
def Train():

    root = tk.Tk()
    root.geometry("800x850+250+5")
    root.title("System Using ML")
    root.configure(background="grey")
    
    Gender = tk.StringVar()
    Age = tk.IntVar()
    Height = tk.IntVar()
    Weight = tk.IntVar()
    family_history_with_overweight = tk.StringVar()
    FAVC = tk.StringVar()
    FCVC= tk.IntVar()
    NCP= tk.IntVar()
    CAEC = tk.StringVar()
    SMOKE = tk.StringVar()
    CH2O = tk.IntVar()
    SCC = tk.StringVar()
    FAF = tk.IntVar()
    TUE = tk.IntVar()
    CALC = tk.StringVar()
    MTRANS = tk.StringVar()
   
    
    
    
    
    
    #===================================================================================================================
    def Detect():
        e1= Gender.get()
        if e1=="Male":
            e1=1
        else:
            e1=0
        print(e1)
        
        e2=Age.get()
     
        print(e2)
        
        e3=Height.get()
        print(e3)
        
        e4=Weight.get()
  

        print(e4)
        
        e5=family_history_with_overweight.get() 
        if e5=="Yes":
            e5=1
        
        else:
            e5=0
        
         
        print(e5)
        
        e6=FAVC.get()
        if e6=="Yes":
            e6=0
        
        else:
            e6=1
        print(e6)
        
        e7=FCVC.get()
        print(e7)
        e8=NCP.get()
        print(e8)
        e9=CAEC.get()
        if e9=="Sometimes":
            e9=1
        elif e9=="Frequently":
            e9=2
        elif e9=="Always":
            e9=3
        else:
            e9=0
        print(e9)
        
        e10=SMOKE.get()
        if e10=="Yes":
            e10=1
        
        else:
            e10=0
        
        print(e10)
        e11=CH2O.get()
        print(e11)
        
        e12=SCC.get()
        if e12=="Yes":
            e12=1
        
        else:
            e12=0
        print(e12)
        e13=FAF.get()
        print(e13)
        
        e14=TUE.get()
        print(e14)
        
        e15=CALC.get()
        if e15=="Sometimes":
            e15=1
        elif e15=="Frequently":
            e15=2
        elif e15=="no":
            e15=0
        else:
            e15=3
        
        print(e15)
        
        e16=MTRANS.get()
        if e16=="Public_Transportation":
            e16=0
        elif e16=="Walking":
            e16=1
        elif e16=="Automobile":
            e16=2
            
        elif e16=="Motorbike":
             e16=3
        else:
            e16=4
        
        print(e16)
               
        
        # from joblib import dump , load
        # a1=load('rf_obesity_model.joblib')
        # v= a1.predict([[e1, e2, e3, e4, e5, e6, e7, e8, e9,e10, e11, e12, e13,e14, e15, e16]])
        # print(v)

        # Example new data (use real input data here)
        new_sample = np.array([[e1, e2, e3, e4, e5, e6, e7, e8, e9,e10, e11, e12, e13,e14, e15, e16]])  # Adjust features as needed

        # Standardize the new sample data using the same scaler
        new_sample = scaler.transform(new_sample)  # Keep this 2D

        # Reshape the sample to fit the input shape expected by the CNN
        new_sample = np.expand_dims(new_sample, axis=2)  # Reshape to 3D

        # Make predictions
        predictions = model.predict(new_sample)

        # Check the shape of predictions
        print(f"Predictions Shape: {predictions.shape}")

        # Get the predicted class (label)
        predicted_class = np.argmax(predictions, axis=1)
        print(f"Predicted Class: {predicted_class}")
        
        if predicted_class==[1]:
            print("Normal_Weight")
            yes = tk.Label(root,text="Normal_Weight",background="green",foreground="white",font=('times', 20, ' bold '),width=30,borderwidth=2,relief='solid')
            yes.place(x=400,y=450)
            
            # list1 = ["Precautions: ","Maintain a balanced diet rich in fruits, vegetables, lean proteins, and whole grains.",
            #          "Regular physical activity (at least 150 minutes of moderate exercise per week)."]
           
            label_l1 = tk.Label(root, text="Precautions: \n1) Maintain a balanced diet rich in fruits, vegetables, lean proteins, and whole grains."+
"\n2) Regular physical activity (at least 150 minutes of moderate exercise per week).",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=10, y=550)
            
            label_l1 = tk.Label(root, text="Precautions: \n3) Monitor your weight regularly to avoid gradual increases."+
"\n4) Get regular medical check-ups to track metabolic health markers like blood pressure, blood sugar, and cholesterol levels.",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=550, y=550)
            
            
                     
        elif predicted_class==[0]:
            print("Insufficient_Weight")
            no = tk.Label(root, text="Insufficient_Weight", background="black", foreground="white",font=('times', 20, ' bold '),width=30,borderwidth=2,relief='solid')
            no.place(x=400, y=450)
            
            label_l1 = tk.Label(root, text="Precautions: \n1) Consult with a healthcare professional to rule out any underlying health issues."+
"\n2) Focus on nutrient-dense foods, such as avocados, nuts, seeds, lean meats, and whole grains."+"\n3) Include strength training exercises to build muscle mass.",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=10, y=550)
            
            label_l1 = tk.Label(root, text="Precautions: \n4) Avoid skipping meals and ensure sufficient caloric intake to meet daily energy needs."+
"\n5) Ensure adequate intake of essential vitamins and minerals, especially calcium and iron.",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=550, y=550)
            
        elif predicted_class==[4]:
            print("Overweight_Level_I")
            no = tk.Label(root, text="Overweight_Level_I", background="red", foreground="white",font=('times', 20, ' bold '),width=30,borderwidth=2,relief='solid')
            no.place(x=400, y=450)
            
            label_l1 = tk.Label(root, text="Precautions: \n1) Adopt portion control and mindful eating to manage calorie intake."+
"\n2) Incorporate regular aerobic exercises such as walking, swimming, or cycling."+"\n3) Monitor lifestyle habits like snacking, sugar intake, and alcohol consumption.",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=10, y=550)
            
            label_l1 = tk.Label(root, text="Precautions: \n4) Address stress through activities like yoga, meditation, or hobbies."+
"\n5) Monitor for signs of weight-related health issues, like increased blood pressure or joint pain.",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=550, y=550)
            
        elif predicted_class==[2]:
            print("Obesity_Type_I")
            no = tk.Label(root, text="Obesity_Type_I", background="red", foreground="white",font=('times', 20, ' bold '),width=30,borderwidth=2,relief='solid')
            no.place(x=400, y=450)
            
            label_l1 = tk.Label(root, text="Precautions: \n1) Focus on a sustainable weight loss plan that includes a balanced diet and calorie deficit."+
"\n2) Engage in both aerobic exercises and strength training to improve metabolism."+"\n3) Behavioral changes, such as mindful eating, can help prevent further weight gain.",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=10, y=550)
            
            label_l1 = tk.Label(root, text="Precautions: \n4) Work with a healthcare provider to track health markers like blood sugar, cholesterol, and blood pressure."+
"\n5) Stay hydrated and ensure proper sleep to support weight management efforts.",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=550, y=550)
        
        elif predicted_class==[3]:
            print("Obesity_Type_II")
            no = tk.Label(root, text="Obesity_Type_II", background="red", foreground="white",font=('times', 20, ' bold '),width=30,borderwidth=2,relief='solid')
            no.place(x=400, y=450)
            
            label_l1 = tk.Label(root, text="Precautions: \n1) Consult with a healthcare provider or dietitian for personalized dietary and exercise advice."+
"\n2) Monitor blood sugar levels closely to prevent or manage diabetes."+"\n3) Include resistance training exercises to preserve muscle mass while losing weight.",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=10, y=550)
            
            label_l1 = tk.Label(root, text="Precautions: \n4) Consider behavior modification programs or therapy if overeating or emotional eating is an issue."+
"\n5) Explore medical treatments (such as medication or bariatric surgery) if lifestyle changes are insufficient.",font=("Times New Roman",10),
                                background="white",borderwidth=2,relief='solid', fg="red",padx=5,pady=5)
            label_l1.place(x=550, y=550)
            
            


    l1=tk.Label(root,text="Gender",background="black",fg='white',font=('times', 18, ' bold '),width=20,bd=5)
    l1.place(x=5,y=50)
    # Id=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Gender)
    # Id.place(x=400,y=50)
    monthchoosen = ttk.Combobox(root, width = 20, textvariable = Gender)

    # Adding combobox drop down list
    monthchoosen['values'] = ('Male',
    						'Female'
     					  )
    monthchoosen.place(x=400,y=50)
    #monthchoosen.grid(column = 1, row = 5)
    monthchoosen.current()

    l2=tk.Label(root,text="Age",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l2.place(x=5,y=100)
    Age=tk.Entry(root,bd=2,width=10,font=("TkDefaultFont", 18),textvar=Age)
    Age.place(x=400,y=100)
  

    l4=tk.Label(root,text="Height",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l4.place(x=5,y=150)
    Height=tk.Entry(root,bd=2,width=10,font=("TkDefaultFont", 18),textvar=Height)
    Height.place(x=400,y=150)
  

    l5=tk.Label(root,text="Weight",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l5.place(x=5,y=200)
    Weight=tk.Entry(root,bd=2,width=10,font=("TkDefaultFont", 18),textvar=Weight)
    Weight.place(x=400,y=200)
    


    l6=tk.Label(root,text="FamilyHistory_overweight",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l6.place(x=5,y=250)
    # timestamp=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=timestamp)
    # timestamp.place(x=400,y=250)
    
    monthchoosen = ttk.Combobox(root, width = 20, textvariable = family_history_with_overweight)

  # Adding combobox drop down list
    monthchoosen['values'] = ('yes','no'
						
 					  )
    monthchoosen.place(x=400,y=250)
#monthchoosen.grid(column = 1, row = 5)
    monthchoosen.current()


    l7=tk.Label(root,text="FAVC",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l7.place(x=5,y=300)
    # SubmitTime=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=FAVC)
    # SubmitTime.place(x=400,y=300)
    
    monthchoosen = ttk.Combobox(root, width = 20, textvariable = FAVC)

  # Adding combobox drop down list
    monthchoosen['values'] = ('yes','no'
						
 					  )
    monthchoosen.place(x=400,y=300)
#monthchoosen.grid(column = 1, row = 5)
    monthchoosen.current()

    l8=tk.Label(root,text="FCVC",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l8.place(x=5,y=350)
    FCVC=tk.Entry(root,bd=2,width=10,font=("TkDefaultFont", 18),textvar=FCVC)
    FCVC.place(x=400,y=350)

    l9=tk.Label(root,text="NCP",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l9.place(x=5,y=400)
    NCP=tk.Entry(root,bd=2,width=10,font=("TkDefaultFont", 18),textvar=NCP)
    NCP.place(x=400,y=400)
    
    ##########################

    l10=tk.Label(root,text="CAEC",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l10.place(x=700,y=50)
    # ResponseDelay=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=CAEC)
    # ResponseDelay.place(x=400,y=450)
    
    monthchoosen = ttk.Combobox(root, width = 20, textvariable = CAEC)

  # Adding combobox drop down list
    monthchoosen['values'] = ('Sometimes','Frequently','Always','no'
						
 					  )
    monthchoosen.place(x=1100,y=50)

    monthchoosen.current()

    l11=tk.Label(root,text="SMOKE",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l11.place(x=700,y=100)
    # FunctionDuration=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=SMOKE)
    # FunctionDuration.place(x=400,y=500)
    monthchoosen = ttk.Combobox(root, width = 20, textvariable = SMOKE)

  # Adding combobox drop down list
    monthchoosen['values'] = ('yes','no'
						
 					  )
    monthchoosen.place(x=1100,y=100)

    monthchoosen.current()

    l12=tk.Label(root,text="CH2O",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l12.place(x=700,y=150)
    CH2O=tk.Entry(root,bd=2,width=10,font=("TkDefaultFont", 18),textvar=CH2O)
    CH2O.place(x=1100,y=150)


    l13=tk.Label(root,text="SCC",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l13.place(x=700,y=200)
    # ActiveFunctionsAtResponse=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=SCC)
    # ActiveFunctionsAtResponse.place(x=400,y=600)
    
    monthchoosen = ttk.Combobox(root, width = 20, textvariable = SCC)

  # Adding combobox drop down list
    monthchoosen['values'] = ('yes','no'
						
 					  )
    monthchoosen.place(x=1100,y=200)

    monthchoosen.current()
  
    
    l14=tk.Label(root,text="FAF",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l14.place(x=700,y=250)
    FAF=tk.Entry(root,bd=2,width=10,font=("TkDefaultFont", 18),textvar=FAF)
    FAF.place(x=1100,y=250)
    
    l15=tk.Label(root,text="TUE",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l15.place(x=700,y=300)
    TUE=tk.Entry(root,bd=2,width=10,font=("TkDefaultFont", 18),textvar=TUE)
    TUE.place(x=1100,y=300)
    
    l16=tk.Label(root,text="CALC",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l16.place(x=700,y=350)
    # p95maxcpu=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=CALC)
    # p95maxcpu.place(x=1100,y=150)
    
    monthchoosen = ttk.Combobox(root, width = 20, textvariable = CALC)

  # Adding combobox drop down list
    monthchoosen['values'] = ('no','Frequently','Sometimes','Always'
						
 					  )
    monthchoosen.place(x=1100,y=350)

    monthchoosen.current()
    
    
    l17=tk.Label(root,text="MTRANS",background="black",fg='white',font=('times', 18, ' bold '),width=20)
    l17.place(x=700,y=400)
    monthchoosen = ttk.Combobox(root, width = 20, textvariable = MTRANS)

# Adding combobox drop down list
    monthchoosen['values'] = ('Public_Transportation','Walking','Automobile','Motorbike','Bike'
						
 					  )
    monthchoosen.place(x=1100,y=400)
#monthchoosen.grid(column = 1, row = 5)
    monthchoosen.current()
    

    
    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 18, ' bold '),width=10)
    button1.place(x=550,y=500)


    root.mainloop()


    
Train()
