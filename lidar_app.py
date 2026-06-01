import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import joblib
import time
import threading


try:
    model = joblib.load('random_forest_model.pkl')

except:
    print("ERROR: 'random_forest_model.pkl' not found! Train the model first.")
    exit()


class autodash:

    def __init__(self,root):
        self.root = root
        self.root.title("BLM2502 - Obstacle Detection with LiDAR ")
        self.root.geometry("850x500")
        self.root.configure(bg="#1e1e2f")

        self.is_running = False
        self.current_data = None

        self.setup_ui()

    def setup_ui(self):
        lbl_title = tk.Label(self.root, text="Live Analysis" , font=("Arial", 20, "bold"), bg="#1e1e2f", fg="#aabdcc")
        lbl_title.pack(pady=20)


        frame_controls = tk.Frame(self.root, bg="#1e1e2f")
        frame_controls.pack(pady=10)

        btn_load = tk.Button(frame_controls, text="Upload .csv file", font = ("Arial", 12), bg="#4caf70", fg= "white", command=self.load_csv)
        btn_load.grid(row=0, column=0, padx=10)

        self.btn_start = tk.Button(frame_controls, text="Start live control", font=("Arial", 12), bg="#ff1020", fg="white", command=self.toggle_stream, state=tk.DISABLED)
        self.btn_start.grid(row=0, column=1, padx=10)

        self.label_file = tk.Label(frame_controls, text="Waiting for file...", font=("Arial", 10), bg="#1e1e2f", fg="#aaaaaa")
        self.label_file.grid(row=1, column=0, columnspan=2, pady=5)

        frame_dash = tk.Frame(self.root, bg="#282a36", bd=2, relief=tk.RIDGE)
        frame_dash.pack(pady=10, padx=40, fill="both", expand=True)

        frame_left = tk.Frame(frame_dash, bg="#282a36")
        frame_left.pack(side="left", padx=30, pady=20)

        tk.Label(frame_left, text="Current Sensor (LiDAR)", font=("Arial", 14, "bold"), bg = "#282a36", fg="#aa79c6").pack(anchor="w", pady=5)
        self.label_width = tk.Label(frame_left, text="Laser width : --", font=("Consolas", 12),bg="#282a36", fg="white")
        self.label_width.pack(anchor="w")
        self.label_rough = tk.Label(frame_left, text="Surface tension : --", font=("Consolas", 12), bg="#282a36", fg="white")
        self.label_rough.pack(anchor="w")
        self.label_dist = tk.Label(frame_left, text = "Closest distance : -- m", font=("Consolas", 12), bg="#282a36", fg="white")
        self.label_dist.pack(anchor="w")


        frame_right = tk.Frame(frame_dash, bg="#282a36")
        frame_right.pack(side="right", padx=30, pady=20)

        tk.Label(frame_right, text="Forest Decision", font=("Arial", 14, "bold"), bg="#282a36", fg="#bd93f9").pack(pady=5)
        self.label_prediction = tk.Label(frame_right, text="Ready to go", font=("Arial", 28, "bold"), bg="#282a36", fg="#50aa60")
        self.label_prediction.pack(pady=10)

        self.label_warning = tk.Label(frame_right, text="STATE : WAITING", font=("Arial", 16, "bold"), bg="#282a36", fg="#50aa60")
        self.label_warning.pack()

    
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

        if file_path:
            self.label_file.config(text=f"Uploaded : {file_path.split('/')[-1]}")
            df = pd.read_csv(file_path, header=None)
            infs_per_row = (~np.isfinite(df)).sum(axis=1)
            self.current_data=df[infs_per_row<=350].values
            self.btn_start.config(state=tk.NORMAL)

    
    def toggle_stream(self):
        if not self.is_running:
            self.is_running = True
            self.btn_start.config(text="Kill the process", bg="#da3434")
            threading.Thread(target=self.process_stream, daemon=True).start()
        else:
            self.is_running =False
            self.btn_start.config(text="Start live stream", bg="#ad0202")


    def process_stream(self):
        for row in self.current_data:
            if not self.is_running: break

            valid_points = row[np.isfinite(row)]
            if len(valid_points) == 0: continue

            width = len(valid_points)
            min_dist = valid_points.min()
            avg_dist = valid_points.mean()
            tension = valid_points.std() if len(valid_points)>1 else 0.0

            features = pd.DataFrame([[width, min_dist, avg_dist, tension]], columns=['Approx_of_laser', 'Minimum_space', 'Avg_distance', 'Surface_smoothness'])

            guess = model.predict(features)[0]

            self.root.after(0,self.update_gui, width, tension, min_dist, guess)

            for _ in range(3):
                if not self.is_running: break
                time.sleep(0.1)

        self.root.after(0, self.reset_button)


    def reset_button(self):
        self.is_running = False
        self.btn_start.config(text="Start live control", bg ="#ad0202")


        
    def update_gui(self, width, tension, min_dist, guess):
        self.label_width.config(text=f"Laser width : {width} amount")
        self.label_rough.config(text=f"Surface tension : {tension:.3f}")
        self.label_dist.config(text=f"Closest distance : {min_dist:.2f} m")

        self.label_prediction.config(text=f"{guess.upper()}")

        if min_dist < 1.0:
            self.label_warning.config(text="EMERGENCY BRAKE RISK!", fg="#ff5555")
            self.label_prediction.config(fg="#ff5555")

        else:
            self.label_warning.config(text="SECURE ROAD", fg="#34bb56")
            self.label_prediction.config(fg="#34bb56")



if __name__ == "__main__":
    root = tk.Tk()
    app = autodash(root)
    root.mainloop()