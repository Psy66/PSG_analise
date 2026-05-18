# ui/base_tab.py
import tkinter as tk
from tkinter import ttk

class BaseTab(ttk.Frame):
    def __init__(self, parent, main_app):
        """
        parent: родительский виджет (вкладка Notebook)
        main_app: ссылка на главное окно (MainWindow) для доступа к общим методам
        """
        super().__init__(parent)
        self.main_app = main_app

    def log(self, message):
        """Отправляет сообщение в общий лог главного окна"""
        self.main_app.log(message)

    def set_status(self, text):
        self.main_app.set_status(text)

    def set_progress(self, value):
        self.main_app.set_progress(value)