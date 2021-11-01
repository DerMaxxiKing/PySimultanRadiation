from tkinter import *
from tkinter.filedialog import askopenfile
from tkinter.scrolledtext import ScrolledText

import logging
from ..logger import logger
from ..shading_analysis import ProjectLoader

logger2 = logging.getLogger('PySimultan')


class WidgetLogger(logging.Handler):
    def __init__(self, widget):
        logging.Handler.__init__(self)
        self.setLevel(logging.DEBUG)
        self.widget = widget
        self.widget.config(state='disabled')
        self.widget.tag_config("INFO", foreground="black")
        self.widget.tag_config("DEBUG", foreground="grey")
        self.widget.tag_config("WARNING", foreground="orange")
        self.widget.tag_config("ERROR", foreground="red")
        self.widget.tag_config("CRITICAL", foreground="red", underline=1)

        self.red = self.widget.tag_configure("red", foreground="red")

    def emit(self, record):
        self.widget.config(state='normal')
        # Append message (record) to the widget
        self.widget.insert(END, self.format(record) + '\n', record.levelname)
        self.widget.see(END)  # Scroll to the bottom
        self.widget.config(state='disabled')
        self.widget.update() # Refresh the widget


class MainWindow:
    def __init__(self):
        self.app = Tk()
        self.app.title('PySimultanRadiation')
        self.app.geometry('650x650')

        x = 70

        Label(self.app, text='Username').place(x=5, y=40, width=x, height=30)
        Label(self.app, text='Password').place(x=5, y=80, width=x, height=30)
        Label(self.app, text='Project').place(x=5, y=120, width=x, height=30)

        self.entry_username = Entry(self.app)
        self.entry_password = Entry(self.app, show="*")
        self.entry_project = Entry(self.app)

        self.entry_username.place(x=x+10, y=40, width=550, height=30)
        self.entry_password.place(x=x+10, y=80, width=550, height=30)
        self.entry_project.place(x=x+10, y=120, width=350, height=30)

        Button(self.app, text='Choose Project', pady=5, padx=30, command=self.choose_project).place(x=450, y=120)
        Button(self.app, text='Run simulation', pady=5, padx=30, command=self.run_simulation).place(x=250, y=160)

        self.st = ScrolledText(self.app, state='disabled')
        self.st.configure(font='TkFixedFont')
        self.st.place(x=5, y=200, width=640, height=400)

        self.text_handler = WidgetLogger(self.st)
        logger.addHandler(self.text_handler)
        logger2.addHandler(self.text_handler)

    def run(self):
        self.app.mainloop()

    def choose_project(self):
        file = askopenfile(title='Select a SIMULTAN Project...',
                           filetypes=[("SIMULTAN", ".simultan")])
        if file is not None:
            self.entry_project.delete(0, "end")
            self.entry_project.insert(0, file.name)

    def run_simulation(self):
        print('running simulation')

        username = self.entry_username.get()
        password = self.entry_password.get()
        project = self.entry_project.get()

        print(username, password, project)

        project_loader = ProjectLoader(project_filename=project,
                                       user_name=username,
                                       password=password)

        project_loader.load_project()
        project_loader.run()
