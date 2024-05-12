import tkinter
import customtkinter as ctk
from PIL import Image
import numpy as np
from tensorflow import keras

# load net
classifier = keras.models.load_model('trained_model_test.hdf5')

# set theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# app
app = ctk.CTk()
app.title("VSC_classification")
app.resizable(False, False)
app.geometry("300x300")
app.grid_columnconfigure(0, weight=1)


# functions
def upload_file():
    global file_to_predict

    file_to_predict = tkinter.filedialog.askopenfilename(title="Select File", filetypes=[("CSV files", "*.csv")])
    if file_to_predict:
        file_label.configure(text="File uploaded")
    else:
        pass


def check():
    global file_to_predict

    if file_to_predict:
        with open(file_to_predict, 'r') as file:
            unknown_lines = file.readlines()

        # check len of line
        num_columns = len(unknown_lines[0].strip().split(';'))

        # predict or valid
        if num_columns > 7:
            valid()
        else:
            prediction()


def prediction():
    global file_to_predict

    if file_to_predict:
        with open(file_to_predict, 'r') as file:
            unknown_lines = file.readlines()

        # process unknown data
        X_unknown = []
        for line in unknown_lines[1:]:
            line = line.strip().split(';')

            X_unknown.append(line)

        # numpy
        X_unknown = np.array(X_unknown, dtype=np.float32)

        # predict
        predictions = classifier.predict(X_unknown)

        # just 0 and 1
        binary_predictions = (predictions >= 0.5).astype(int)

        # label it
        class_labels = ["classA", "classB"]
        class_predictions = [class_labels[prediction[0]] for prediction in binary_predictions]

        # write data
        header = unknown_lines[0].strip() + ';Prediction\n'
        lines_with_predictions = [f'{line.strip()};{class_pred}\n' for line, class_pred in
                                  zip(unknown_lines[1:], class_predictions)]

        with open('predictions.csv', 'w') as f:
            f.write(header)
            f.writelines(lines_with_predictions)

        predict_label.configure(text="Predictions are saved as 'predictions.csv'.")

    else:
        pass


def valid():
    global file_to_predict # here for validation

    if file_to_predict:
        with open(file_to_predict, 'r') as file:
            unknown_lines = file.readlines()

        # process data
        X_unknown = []
        true_classes = []  # correct classes
        for line in unknown_lines[1:]:
            line = line.strip().split(';')

            X_unknown.append(line[:-1])
            true_classes.append(line[-1])


        # numpy
        X_unknown = np.array(X_unknown, dtype=np.float32)

        # make predictions
        predictions = classifier.predict(X_unknown)

        # 0 and 1
        binary_predictions = (predictions >= 0.5).astype(int)

        # labels
        class_labels = ["classA", "classB"]
        class_predictions = [class_labels[prediction[0]] for prediction in binary_predictions]

        # compare predict with correct classes and sum
        correct_predictions = sum(1 for pred, true in zip(class_predictions, true_classes) if pred == true)

        # ACC = (TP+TN)/(P+N) ---> correct/all
        accuracy = correct_predictions / len(true_classes)

        predict_label.configure(text="ACC = " + str(accuracy))

    else:
        pass


# File upload button
file_icon_image = ctk.CTkImage(dark_image=Image.open("open-folder.png"), size=(30, 30))
file_button = ctk.CTkButton(app, image=file_icon_image, text="Your 'csv' file.", command=upload_file)
file_button.grid(row=0, column=0, padx=10, pady=(10, 0))

# File pload label
file_label = ctk.CTkLabel(app, text="Do not see file.")
file_label.grid(row=1, column=0, padx=10, pady=(10, 0))

# execute button
net_button = ctk.CTkButton(app, text="Classsification", command=check)
net_button.grid(row=2, column=0, padx=10, pady=(10, 0))

# predict label
predict_label = ctk.CTkLabel(app, text="")
predict_label.grid(row=3, column=0, padx=10, pady=(10, 0))

app.mainloop()