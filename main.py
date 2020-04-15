import math
import tkinter as tk
from keras.models import load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from utils import list_to_sparse_matrix, resize_img


class NumberGUI:

    def __init__(self, master, model):
        self.master = master
        self.master.geometry("750x380")
        self.master.resizable(False, False)
        self.master.title("MNIST Predictor")
        self.model = model
        self.outer_frame = tk.Frame(self.master,
                                    width=300,
                                    height=300)
        self.outer_frame.pack(side=tk.LEFT)

        self.frame = tk.Frame(self.outer_frame,
                              highlightthickness=5,
                              highlightbackground='black')
        self.frame.pack(side=tk.TOP)

        self.canvas = tk.Canvas(self.frame,
                                width=280,
                                height=280,
                                bd=10,
                                highlightthickness=0,
                                borderwidth=0,
                                background='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.old_coords = None
        self.canvas.coords_list = []

        self.message = tk.Label(self.outer_frame, text="Input number by clicking and dragging", padx=5, pady=5)
        self.message.pack(side=tk.BOTTOM)

        self.submit_button = tk.Button(self.outer_frame, text='Submit', command=lambda: self.submit(), width=10)
        self.submit_button.pack(side=tk.RIGHT, padx=25, pady=10)

        self.clear_button = tk.Button(self.outer_frame, text='Clear', command=lambda: self.clear_canvas(), width=10)
        self.clear_button.pack(side=tk.LEFT, padx=25, pady=10)

        self.close_button = tk.Button(self.outer_frame, text='Close', command=lambda: self.close_window(), width=10)
        self.close_button.pack(pady=10)

        self.plot = Plotter(self.master)

    def submit(self):
        if self.canvas.coords_list:
            img = resize_img(list_to_sparse_matrix(self.canvas.coords_list))
            preds = self.model.predict(img)[0].tolist()
            preds_dict = {i: j for i, j in enumerate(preds)}
            max_val = max(preds_dict, key=preds_dict.get)
            self.popup(f"Prediction: {max_val} Confidence: {round(preds_dict[max_val]*100,2)}%")
        else:
            self.popup('You must enter a digit.')

    def close_window(self):
        self.master.destroy()

    def clear_canvas(self):
        self.canvas.delete('all')
        self.canvas.coords_list = []
        self.plot.reset_plot()

    def paint(self, event):

        x, y = event.x, event.y

        if self.canvas.old_coords:
            x1, y1 = self.canvas.old_coords
            d = math.sqrt(((x - x1)**2) + ((y - y1)**2))
            if d <= 20:
                self.canvas.create_line(x, y, x1, y1, capstyle=tk.ROUND, width=10)
            else:
                self.canvas.old_coords = None

        self.canvas.old_coords = x, y

        if x < self.canvas.winfo_width() and y < self.canvas.winfo_height():

            self.canvas.coords_list.append((x, y))
            img = resize_img(list_to_sparse_matrix(self.canvas.coords_list))
            preds = self.model.predict(img)[0].tolist()
            preds_dict = {i: j for i, j in enumerate(preds)}
            #max_val = max(preds_dict, key=preds_dict.get)
            #print(f"Prediction: {max_val} Confidence: {round(preds_dict[max_val] * 100, 2)}%")
            self.plot.update_plot(preds_dict.values())

    def popup(self, msg):
        popup = tk.Tk()
        popup.wm_title("Results")
        label = tk.Label(popup, text=msg, font=(None, 24))
        label.pack(side="top", fill="x", pady=10, padx=10)
        b1 = tk.Button(popup, text="Ok", command=popup.destroy, width=10)
        b1.pack(padx=10, pady=10)
        popup.mainloop()


class Plotter:
    def __init__(self, master):
        self.master = master
        self.x = [0 for i in range(10)]
        self.y = [i for i in range(10)]
        self.fig = Figure(figsize=(3, 3))
        self.ax = self.fig.add_subplot(111)
        self.ax.set(xlim=[0, 1], yticks=[i for i in range(10)], title='Predicted Probability')
        self.bar_plot = self.ax.barh(self.y, self.x)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(pady=20)

    def update_plot(self, y):

        for bar, y_val in zip(self.bar_plot, y):
            bar.set_width(y_val)

        self.fig_canvas.draw()
        self.fig_canvas.flush_events()

    def reset_plot(self):

        y = [0 for i in range(10)]

        for bar, y_val in zip(self.bar_plot, y):
            bar.set_width(y_val)

        self.fig_canvas.draw()
        self.fig_canvas.flush_events()


def main():
    model = load_model('initial_mnist.h5')
    root = tk.Tk()
    NumberGUI(root, model)
    root.mainloop()


if __name__ == '__main__':
    main()