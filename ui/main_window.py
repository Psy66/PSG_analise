# ui/main_window.py
import tkinter as tk
import pandas as pd
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime

class MainWindow:
    def __init__(self, root):
        self.root = root
        root.title("PSG Analysis Tool")
        root.geometry("1600x1200")
        root.configure(bg='#f0f0f0')

        self.status_var = tk.StringVar(value="Готов")
        self.progress_var = tk.IntVar(value=0)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Прогресс и статус
        progress_frame = ttk.Frame(root)
        progress_frame.pack(fill=tk.X, padx=10, pady=2)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_bar = ttk.Label(progress_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.RIGHT, padx=5)

        # Лог
        log_frame = ttk.LabelFrame(root, text="Лог", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=12, font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Кнопка выхода внизу справа
        button_frame = ttk.Frame(root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        exit_btn = ttk.Button(button_frame, text="Выход", command=self.quit_app)
        exit_btn.pack(side=tk.RIGHT)

        self.tabs = {}
        self._create_tabs()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def quit_app(self):
        """Завершает приложение."""
        if messagebox.askyesno("Подтверждение", "Выйти из программы?"):
            self.root.quit()

    def _create_tabs(self):
        from ui.tab_load import LoadDataTab
        from ui.tab_stats import StatsTab
        from ui.tab_lmm import LMMAnalysisTab
        from ui.tab_pca import PCAAnalysisTab
        from ui.tab_event_locked import EventLockedTab
        from ui.tab_gam import GAMTab
        from ui.tab_dfa_coherence import DfaCoherenceTab


        self.tabs['load'] = LoadDataTab(self.notebook, self)
        self.notebook.add(self.tabs['load'], text="Загрузка и фильтры")

        self.tabs['stats'] = StatsTab(self.notebook, self)
        self.stats_tab = self.tabs['stats']
        self.notebook.add(self.tabs['stats'], text="Статистика")

        self.tabs['lmm'] = LMMAnalysisTab(self.notebook, self)
        self.notebook.add(self.tabs['lmm'], text="LMM анализ")

        self.tabs['dfa_coherence'] = DfaCoherenceTab(self.notebook, self)
        self.notebook.add(self.tabs['dfa_coherence'], text="DFA и когерентность")

        self.tabs['event_locked'] = EventLockedTab(self.notebook, self)
        self.notebook.add(self.tabs['event_locked'], text="Event‑locked анализ")

        self.tabs['pca'] = PCAAnalysisTab(self.notebook, self)
        self.notebook.add(self.tabs['pca'], text="PCA анализ")

        self.tabs['gam'] = GAMTab(self.notebook, self)
        self.notebook.add(self.tabs['gam'], text="GAM анализ")



    def on_tab_changed(self, event):
        current = self.notebook.select()
        tab = self.notebook.nametowidget(current)
        if hasattr(tab, 'on_tab_selected'):
            tab.on_tab_selected()

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def set_status(self, text):
        self.status_var.set(text)

    def set_progress(self, value):
        self.progress_var.set(value)
        self.root.update_idletasks()

    def set_filtered_data(self, df):
        self.filtered_data = df

    def get_filtered_data(self):
        return getattr(self, 'filtered_data', None)

    def set_analysis_settings(self, group_by_severity, include_central_mixed):
        self.group_by_severity = group_by_severity
        self.include_central_mixed = include_central_mixed
        if hasattr(self, 'stats_tab') and hasattr(self.stats_tab, 'update_settings'):
            self.stats_tab.update_settings(group_by_severity, include_central_mixed)

    def get_analysis_settings(self):
        return (getattr(self, 'group_by_severity', True),
                getattr(self, 'include_central_mixed', False))

    def get_covariates_for_studies(self):
        """Возвращает DataFrame с ковариатами (study_id, age_at_study, gender_code, bmi)"""
        df = self.get_filtered_data()
        if df is None or df.empty:
            return pd.DataFrame()
        # Убеждаемся, что колонки есть
        needed = ['study_id', 'age_at_study', 'gender', 'bmi']
        for col in needed:
            if col not in df.columns:
                self.log(f"В отфильтрованных данных отсутствует колонка {col}, ковариаты не будут использованы.")
                return pd.DataFrame()
        cov = df[['study_id', 'age_at_study', 'gender', 'bmi']].drop_duplicates(subset=['study_id'])
        cov['gender_code'] = (cov['gender'] == 'M').astype(int)
        return cov