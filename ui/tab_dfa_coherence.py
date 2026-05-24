# ui/tab_dfa_coherence.py
"""
Вкладка DFA и когерентность (п. 2.4.4, 2.4.5, 2.5.7 Главы 2).
- LMM для DFA-экспонента и когерентности (has_apnea + ковариаты)
- ANOVA для DFA по группам тяжести ОАС
- Карты когерентности (brain map и тепловые матрицы)
- Диагностика остатков, бутстрап, отчёт
"""

import base64
import io
import os
import tempfile
import threading
import tkinter as tk
import warnings
import webbrowser
from tkinter import messagebox, ttk

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import false_discovery_control, shapiro, chi2, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from ui.base_tab import BaseTab
from core.api_client import get_epochs

warnings.filterwarnings("ignore", module="statsmodels")


class DfaCoherenceTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)

        # ---- Параметры модели (общие) ----
        self.analysis_mode = tk.StringVar(value="dfa")   # 'dfa' or 'coh'
        self.data_type = tk.IntVar(value=2)              # 1=тонический, 2=все эпохи, 3=фильтр по положению
        self.exclude_central_mixed = tk.BooleanVar(value=True)
        self.use_cache = tk.BooleanVar(value=True)

        self.include_stage = tk.BooleanVar(value=True)
        self.include_covariates = tk.BooleanVar(value=True)
        self.fdr_threshold = tk.DoubleVar(value=0.05)

        # ---- Выбор признаков (DFA) ----
        self.dfa_channels = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        self.selected_dfa_channels = []

        # ---- Выбор признаков (когерентность) ----
        self.coh_pairs = [('F3','C3'),('F3','F4'),('F3','C4'),('C3','C4'),('C3','O1'),
                          ('F4','C4'),('O1','O2'),('C3','O2')]
        self.coh_bands = ['delta','theta','alpha','sigma','beta','gamma']
        self.selected_coh_pairs = []
        self.selected_coh_bands = []

        # ---- Данные ----
        self.results_df = None          # DataFrame с результатами LMM
        self.current_figure = None      # текущий график LMM (forest/volcano)
        self.stop_flag = False
        self.epochs_df = None
        self.epochs_loaded = False

        # ---- Диагностика и бутстрап ----
        self.all_diagnostics = []
        self.bootstrap_info = None
        self.diag_model = None
        self.diag_feature_name = ""
        self.diag_figure = None

        # ---- Для карт когерентности ----
        self.current_coherence_band = tk.StringVar(value='delta')
        self.coherence_map_figure = None
        self.coherence_matrix_figure = None   # для тепловой матрицы

        # ---- Результаты ANOVA для DFA ----
        self.dfa_anova_results = None

        self.saved_dfa_indices = None
        self.saved_pair_indices = None
        self.saved_band_indices = None

        self._create_widgets()

    # ------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------
    def _create_widgets(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)

        # ---------- ЛЕВАЯ ПАНЕЛЬ ----------
        # Режим анализа
        mode_frame = ttk.LabelFrame(left_frame, text="Режим анализа", padding=5)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Radiobutton(mode_frame, text="LMM анализ DFA (экспонент α)", variable=self.analysis_mode,
                        value='dfa', command=self._toggle_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="LMM анализ когерентности", variable=self.analysis_mode,
                        value='coh', command=self._toggle_mode).pack(anchor=tk.W)

        # Настройки данных (общие)
        data_frame = ttk.LabelFrame(left_frame, text="Данные", padding=5)
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        # ttk.Label(data_frame, text="Тип набора эпох:").grid(row=0, column=0, sticky=tk.W)
        # ttk.Combobox(data_frame, textvariable=self.data_type, values=[1,2,3],
        #              state='readonly', width=5).grid(row=0, column=1, padx=5)
        # ttk.Label(data_frame, text="1=тонический, 2=все эпохи, 3=фильтр по положению").grid(row=0, column=2, sticky=tk.W)
        ttk.Checkbutton(data_frame, text="Исключить центральное/смешанное апноэ",
                        variable=self.exclude_central_mixed).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)
        ttk.Checkbutton(data_frame, text="Использовать кэш API", variable=self.use_cache).grid(row=2, column=0, columnspan=3, sticky=tk.W)

        # Параметры LMM
        lmm_frame = ttk.LabelFrame(left_frame, text="Параметры LMM", padding=5)
        lmm_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(lmm_frame, text="Включить стадию сна (N2/N3)", variable=self.include_stage).pack(anchor=tk.W)
        ttk.Checkbutton(lmm_frame, text="Включить ковариаты (возраст, пол, ИМТ)",
                        variable=self.include_covariates).pack(anchor=tk.W)
        ttk.Label(lmm_frame, text="Порог FDR (q):").pack(anchor=tk.W)
        ttk.Entry(lmm_frame, textvariable=self.fdr_threshold, width=6).pack(anchor=tk.W)

        # Панель выбора признаков (будет меняться в _toggle_mode)
        self.selection_frame = ttk.LabelFrame(left_frame, text="Выбор признаков", padding=5)
        self.selection_frame.pack(fill=tk.X, padx=5, pady=5)

        # DFA: список каналов (Listbox с мультивыбором)
        self.dfa_listbox = tk.Listbox(self.selection_frame, selectmode=tk.EXTENDED, height=6, width=20, exportselection=False)
        for ch in self.dfa_channels:
            self.dfa_listbox.insert(tk.END, ch)
        self.dfa_listbox.selection_set(0, tk.END)
        # Когерентность: пары и диапазоны
        self.pairs_listbox = tk.Listbox(self.selection_frame, selectmode=tk.EXTENDED, height=5, exportselection=False)
        for a,b in self.coh_pairs:
            self.pairs_listbox.insert(tk.END, f"{a}-{b}")
        self.pairs_listbox.selection_set(0, tk.END)
        self.bands_listbox = tk.Listbox(self.selection_frame, selectmode=tk.EXTENDED, height=4, exportselection=False)
        for b in self.coh_bands:
            self.bands_listbox.insert(tk.END, b)
        self.bands_listbox.selection_set(0, tk.END)

        self.dfa_listbox.bind('<<ListboxSelect>>', self._save_dfa_selection)
        self.pairs_listbox.bind('<<ListboxSelect>>', self._save_pair_selection)
        self.bands_listbox.bind('<<ListboxSelect>>', self._save_band_selection)

        # ---- Дополнительные анализы ----
        extra_frame = ttk.LabelFrame(left_frame, text="Дополнительные анализы (Глава 2, п.2.5.7)", padding=5)
        extra_frame.pack(fill=tk.X, padx=5, pady=5)
        self.anova_btn = ttk.Button(extra_frame, text="ANOVA для DFA по тяжести ОАС",
                                    command=self.run_dfa_anova, state=tk.DISABLED)
        self.anova_btn.pack(anchor=tk.W, padx=5, pady=2)
        self.matrix_btn = ttk.Button(extra_frame, text="Показать матрицу когерентности",
                                     command=self.show_coherence_matrix, state=tk.DISABLED)
        self.matrix_btn.pack(anchor=tk.W, padx=5, pady=2)

        # Проверка нормальности
        norm_frame = ttk.LabelFrame(left_frame, text="Проверка нормальности", padding=5)
        norm_frame.pack(fill=tk.X, padx=5, pady=5)
        row1 = ttk.Frame(norm_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Признак:").pack(side=tk.LEFT, padx=2)
        self.norm_selector = ttk.Combobox(row1, state="readonly", width=20)
        self.norm_selector.pack(side=tk.LEFT, padx=2)
        self.norm_btn = ttk.Button(row1, text="Проверить", command=self.check_normality, state=tk.DISABLED)
        self.norm_btn.pack(side=tk.LEFT, padx=2)
        self.save_norm_plot_btn = ttk.Button(row1, text="Сохранить PNG", command=self.save_norm_plot, state=tk.DISABLED)
        self.save_norm_plot_btn.pack(side=tk.LEFT, padx=2)

        # Диагностика и бутстрап
        diag_frame = ttk.LabelFrame(left_frame, text="Диагностика модели", padding=5)
        diag_frame.pack(fill=tk.X, padx=5, pady=5)
        row2 = ttk.Frame(diag_frame)
        row2.pack(fill=tk.X, pady=2)
        self.diag_selector = ttk.Combobox(row2, state="readonly", width=20)
        self.diag_selector.pack(side=tk.LEFT, padx=2)
        self.diag_btn = ttk.Button(row2, text="Диагностика", command=self.run_diagnostics, state=tk.DISABLED)
        self.diag_btn.pack(side=tk.LEFT, padx=2)
        self.save_diag_plot_btn = ttk.Button(row2, text="Сохранить PNG", command=self.save_diag_plot, state=tk.DISABLED)
        self.save_diag_plot_btn.pack(side=tk.LEFT, padx=2)
        row3 = ttk.Frame(diag_frame)
        row3.pack(fill=tk.X, pady=2)
        self.bootstrap_btn = ttk.Button(row3, text="Бутстрап (1000 итераций)", command=self.run_bootstrap, state=tk.DISABLED)
        self.bootstrap_btn.pack(side=tk.LEFT, padx=2)
        self.save_bootstrap_btn = ttk.Button(row3, text="Сохранить CSV", command=self.save_bootstrap_csv, state=tk.DISABLED)
        self.save_bootstrap_btn.pack(side=tk.LEFT, padx=2)

        # Основные кнопки
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Запустить LMM", command=self.run_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить CSV", command=self.save_results_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=2)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=2)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать отчёт", command=self.generate_report, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=2)

        # ---------- ПРАВАЯ ПАНЕЛЬ (Notebook) ----------
        self.right_notebook = ttk.Notebook(right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка результатов LMM
        self.tab_table = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_table, text="Результаты LMM")
        self.tree_frame = ttk.Frame(self.tab_table)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(self.tree_frame)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        # Вкладка графиков LMM
        self.tab_plots = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_plots, text="Графики LMM")
        self.plot_frame = ttk.Frame(self.tab_plots)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Вкладка проверки нормальности
        self.tab_norm = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_norm, text="Проверка нормальности")
        self.norm_plot_frame = ttk.Frame(self.tab_norm)
        self.norm_plot_frame.pack(fill=tk.BOTH, expand=True)

        # Вкладка диагностики модели
        self.tab_diag = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_diag, text="Диагностика модели")
        self.diag_plot_frame = ttk.Frame(self.tab_diag)
        self.diag_plot_frame.pack(fill=tk.BOTH, expand=True)

        # Вкладка карты когерентности (brain map)
        self.tab_coherence_map = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_coherence_map, text="Карта когерентности")
        map_control = ttk.Frame(self.tab_coherence_map)
        map_control.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(map_control, text="Диапазон:").pack(side=tk.LEFT, padx=5)
        self.band_selector = ttk.Combobox(map_control, textvariable=self.current_coherence_band,
                                          values=self.coh_bands, state='readonly', width=10)
        self.band_selector.pack(side=tk.LEFT, padx=5)
        self.band_selector.bind("<<ComboboxSelected>>", lambda e: self.update_coherence_map())
        self.map_frame = ttk.Frame(self.tab_coherence_map)
        self.map_frame.pack(fill=tk.BOTH, expand=True)

        # Вкладка матриц когерентности (тепловая карта)
        self.tab_coherence_matrix = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_coherence_matrix, text="Матрица когерентности")
        matrix_control = ttk.Frame(self.tab_coherence_matrix)
        matrix_control.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(matrix_control, text="Группа:").pack(side=tk.LEFT, padx=5)
        self.matrix_group = tk.StringVar()
        self.matrix_group_combo = ttk.Combobox(matrix_control, textvariable=self.matrix_group, state='readonly', width=15)
        self.matrix_group_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(matrix_control, text="Диапазон:").pack(side=tk.LEFT, padx=5)
        self.matrix_band = tk.StringVar(value='delta')
        self.matrix_band_combo = ttk.Combobox(matrix_control, textvariable=self.matrix_band,
                                              values=self.coh_bands, state='readonly', width=10)
        self.matrix_band_combo.pack(side=tk.LEFT, padx=5)
        self.matrix_show_btn = ttk.Button(matrix_control, text="Показать матрицу", command=self.show_coherence_matrix)
        self.matrix_show_btn.pack(side=tk.LEFT, padx=5)
        self.matrix_frame = ttk.Frame(self.tab_coherence_matrix)
        self.matrix_frame.pack(fill=tk.BOTH, expand=True)

        # Вкладка ANOVA для DFA
        self.tab_anova = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_anova, text="ANOVA (DFA)")
        self.anova_text = tk.Text(self.tab_anova, wrap=tk.WORD, font=("Courier New", 10))
        self.anova_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Вкладка лога
        self.tab_log = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_log, text="Лог")
        self.log_text = tk.Text(self.tab_log, wrap=tk.WORD, font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._toggle_mode()

    # ------------------------------------------------------------
    # Безопасные очистки
    # ------------------------------------------------------------
    def _clear_table(self):
        if hasattr(self, 'tree') and self.tree:
            for row in self.tree.get_children():
                self.tree.delete(row)

    def _clear_plots(self):
        if hasattr(self, 'plot_frame') and self.plot_frame:
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
        self.current_figure = None

    def _toggle_mode(self):
        """Показывает соответствующий список выбора признаков и сохраняет выделение."""
        if hasattr(self, 'analysis_mode'):
            if self.analysis_mode.get() == 'dfa':
                self.saved_dfa_indices = self.dfa_listbox.curselection()
            else:
                self.saved_pair_indices = self.pairs_listbox.curselection()
                self.saved_band_indices = self.bands_listbox.curselection()

        for w in self.selection_frame.winfo_children():
            w.pack_forget()

        self._clear_results()

        if self.analysis_mode.get() == 'dfa':
            self.dfa_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            if hasattr(self, 'saved_dfa_indices') and self.saved_dfa_indices:
                self.dfa_listbox.selection_clear(0, tk.END)
                for idx in self.saved_dfa_indices:
                    if idx < self.dfa_listbox.size():
                        self.dfa_listbox.selection_set(idx)
        else:
            ttk.Label(self.selection_frame, text="Пары отведений:").pack(anchor=tk.W)
            self.pairs_listbox.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(self.selection_frame, text="Частотные диапазоны:").pack(anchor=tk.W)
            self.bands_listbox.pack(fill=tk.X, padx=5, pady=2)
            if hasattr(self, 'saved_pair_indices') and self.saved_pair_indices:
                self.pairs_listbox.selection_clear(0, tk.END)
                for idx in self.saved_pair_indices:
                    if idx < self.pairs_listbox.size():
                        self.pairs_listbox.selection_set(idx)
            if hasattr(self, 'saved_band_indices') and self.saved_band_indices:
                self.bands_listbox.selection_clear(0, tk.END)
                for idx in self.saved_band_indices:
                    if idx < self.bands_listbox.size():
                        self.bands_listbox.selection_set(idx)

        self._update_norm_selector()
        self._update_diag_selector()

    # ------------------------------------------------------------
    # Логирование и остановка
    # ------------------------------------------------------------
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.main_app.log(msg)

    def stop_analysis(self):
        self.stop_flag = True
        self.log("Остановка запрошена...")

    # ------------------------------------------------------------
    # Получение ковариат
    # ------------------------------------------------------------
    def _get_covariates(self, study_ids):
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None:
            return None
        needed = ['study_id', 'age_at_study', 'gender', 'bmi']
        if not all(c in filtered_df.columns for c in needed):
            return None
        cov = filtered_df[needed].drop_duplicates(subset=['study_id'])
        cov = cov[cov['study_id'].isin(study_ids)]
        if cov.empty:
            return None
        cov['gender_code'] = (cov['gender'] == 'M').astype(int)
        return cov

    # ------------------------------------------------------------
    # Загрузка эпох
    # ------------------------------------------------------------
    def _load_epochs(self, study_ids, force_reload=False):
        if not force_reload and self.epochs_df is not None:
            return self.epochs_df
        def update_progress(page, total, _):
            if total > 0:
                self.main_app.set_progress(int(page / total * 100))
        self.main_app.set_progress(0)
        self.log("Загрузка эпох из API...")
        load_tab = self.main_app.tabs['load']
        epochs = get_epochs(
            load_tab.api_url.get(),
            load_tab.token.get(),
            study_ids=study_ids,
            data_type=self.data_type.get(),
            stop_check=lambda: self.stop_flag,
            progress_callback=update_progress,
            use_cache=self.use_cache.get()
        )
        self.main_app.set_progress(100)
        if not epochs or self.stop_flag:
            return None
        df = pd.DataFrame(epochs)
        self.log(f"Загружено {len(df)} эпох.")
        self.epochs_df = df
        self.epochs_loaded = True
        return df

    # ------------------------------------------------------------
    # Основной LMM (DFA или когерентность)
    # ------------------------------------------------------------
    def run_analysis(self):
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            messagebox.showerror("Ошибка", "Нет отфильтрованных данных. Сначала загрузите данные на вкладке 'Загрузка'.")
            return
        if self.exclude_central_mixed.get():
            allowed = ['no_impairment', 'mild', 'moderate', 'severe']
            filtered_df = filtered_df[filtered_df['breathing_impairment_severity'].isin(allowed)]
            if filtered_df.empty:
                messagebox.showerror("Ошибка", "После исключения центрального/смешанного апноэ нет данных.")
                return
        study_ids = filtered_df['study_id'].unique().tolist()
        self.log(f"Исследований: {len(study_ids)}")
        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_csv_btn.config(state=tk.DISABLED)
        self.save_plot_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        self.diag_btn.config(state=tk.DISABLED)
        self.bootstrap_btn.config(state=tk.DISABLED)
        self.anova_btn.config(state=tk.DISABLED)
        self.matrix_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_lmm_thread, args=(study_ids,), daemon=True).start()

    def _run_lmm_thread(self, study_ids):
        try:
            if self.epochs_df is None:
                df = self._load_epochs(study_ids)
            else:
                df = self.epochs_df
            if df is None or self.stop_flag:
                self.log("Нет данных для LMM.")
                return
            df = df.copy()
            df['has_apnea'] = df['has_apnea'].astype(bool)
            df['patient_id'] = df['patient_id'].astype(int)
            if 'epoch_stage' in df.columns:
                df['epoch_stage'] = df['epoch_stage'].astype('category')
            if 'gender' in df.columns and 'gender_code' not in df.columns:
                df['gender_code'] = (df['gender'] == 'M').astype(int)

            if self.include_covariates.get():
                cov_df = self._get_covariates(study_ids)
                if cov_df is not None and not cov_df.empty:
                    for c in ['age_at_study', 'gender_code', 'bmi']:
                        if c in df.columns:
                            df = df.drop(columns=[c])
                    df = df.merge(cov_df[['study_id', 'age_at_study', 'gender_code', 'bmi']], on='study_id', how='left')
                    self.log("Ковариаты добавлены.")
                else:
                    self.log("Ковариаты не загружены, продолжаем без них.")
                    self.include_covariates.set(False)

            mode = self.analysis_mode.get()
            if mode == 'dfa':
                tasks = self._build_dfa_tasks(df)
            else:
                tasks = self._build_coh_tasks(df)

            if not tasks:
                self.log("Нет доступных признаков для выбранного режима.")
                return
            self.log(f"Всего моделей: {len(tasks)}. Начинаем расчёт...")
            results = []
            total = len(tasks)
            for i, task in enumerate(tasks):
                if self.stop_flag:
                    break
                self.main_app.set_progress(int(i / total * 100))
                res = self._fit_lmm_model(df, task)
                if res:
                    results.append(res)
                if (i + 1) % 30 == 0:
                    self.log(f"Обработано {i + 1} из {total}")
            if not results:
                self.log("Нет результатов (модели не сошлись).")
                return
            res_df = pd.DataFrame(results)
            pvals = res_df['p_value'].values
            res_df['q_value'] = false_discovery_control(pvals, method='bh')
            res_df['significant'] = res_df['q_value'] < self.fdr_threshold.get()
            self.results_df = res_df
            self.main_app.root.after(0, self._display_results_table, res_df)
            self.main_app.root.after(0, self._plot_lmm_results, res_df)
            self.save_csv_btn.config(state=tk.NORMAL)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.diag_btn.config(state=tk.NORMAL)
            self.bootstrap_btn.config(state=tk.NORMAL)
            self.anova_btn.config(state=tk.NORMAL)
            self.matrix_btn.config(state=tk.NORMAL)
            self._update_norm_selector()
            self._update_diag_selector()
            if mode == 'coh':
                self.update_coherence_map()
                # Заполняем группы для матрицы
                groups = sorted(self._get_available_groups_for_coherence())
                self.matrix_group_combo['values'] = groups
                if groups:
                    self.matrix_group.set(groups[0])
            n_sign = res_df['significant'].sum()
            self.log(f"LMM завершён. Значимых признаков: {n_sign} (q<{self.fdr_threshold.get()})")
        except Exception as e:
            self.log(f"Ошибка: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.main_app.set_progress(0)

    def _build_dfa_tasks(self, df):
        selected_indices = self.dfa_listbox.curselection()
        if not selected_indices:
            channels = self.dfa_channels[:]
        else:
            channels = [self.dfa_listbox.get(i) for i in selected_indices]
        tasks = []
        for ch in channels:
            col = f"{ch}_dfa"
            if col in df.columns:
                tasks.append({
                    'type': 'dfa',
                    'channel': ch,
                    'col': col,
                    'display': ch
                })
            else:
                self.log(f"Столбец {col} не найден.")
        return tasks

    def _build_coh_tasks(self, df):
        pair_indices = self.pairs_listbox.curselection()
        if not pair_indices:
            pairs = [f"{a}-{b}" for a,b in self.coh_pairs]
        else:
            pairs = [self.pairs_listbox.get(i) for i in pair_indices]
        band_indices = self.bands_listbox.curselection()
        if not band_indices:
            bands = self.coh_bands[:]
        else:
            bands = [self.bands_listbox.get(i) for i in band_indices]
        tasks = []
        for pair_str in pairs:
            pair_clean = pair_str.replace('-','')
            for band in bands:
                col = f"{pair_clean}_coh_{band}"
                if col in df.columns:
                    tasks.append({
                        'type': 'coh',
                        'pair': pair_str,
                        'band': band,
                        'col': col,
                        'display': f"{pair_str} {band}"
                    })
                else:
                    self.log(f"Столбец {col} не найден, пропускаем.")
        return tasks

    def _fit_lmm_model(self, df, task):
        col = task['col']
        cols = [col, 'has_apnea', 'patient_id']
        if self.include_stage.get() and 'epoch_stage' in df.columns:
            cols.append('epoch_stage')
        if self.include_covariates.get():
            for c in ['age_at_study', 'gender_code', 'bmi']:
                if c in df.columns:
                    cols.append(c)
        sub = df[cols].dropna()
        if len(sub) < 30:
            return None
        if sub['has_apnea'].nunique() < 2:
            return None
        formula = f"{col} ~ has_apnea"
        if self.include_stage.get() and 'epoch_stage' in sub.columns:
            formula += " + epoch_stage"
        if self.include_covariates.get():
            for c in ['age_at_study', 'gender_code', 'bmi']:
                if c in sub.columns:
                    formula += f" + {c}"
        try:
            model = smf.mixedlm(formula, sub, groups=sub['patient_id'])
            result = model.fit(reml=True, method='lbfgs', maxiter=1000)
            # Проверка сходимости (если атрибут существует)
            # if hasattr(result, 'mle_retvals') and not result.mle_retvals.get('converged', True):
            #     return None
            beta = result.params.get('has_apnea[T.True]', result.params.get('has_apnea', None))
            pval = result.pvalues.get('has_apnea[T.True]', result.pvalues.get('has_apnea', None))
            if beta is None or pval is None:
                return None
            ci = result.conf_int().loc['has_apnea[T.True]' if 'has_apnea[T.True]' in result.params else 'has_apnea']
            res = {
                'beta': beta,
                'p_value': pval,
                'ci_low': ci[0],
                'ci_high': ci[1],
                'n_obs': len(sub)
            }
            if task['type'] == 'dfa':
                res['channel'] = task['channel']
                res['feature'] = 'DFA'
            else:
                res['pair'] = task['pair']
                res['band'] = task['band']
                res['feature'] = f"{task['pair']}_{task['band']}"
            return res
        except Exception as e:
            # Для отладки можно раскомментировать:
            # self.log(f"  {task['display']}: исключение {type(e).__name__}: {e}")
            return None

    def _display_results_table(self, df):
        self._clear_table()
        if df.empty:
            return
        cols = list(df.columns)
        self.tree['columns'] = cols
        self.tree['show'] = 'headings'
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='center')
        for _, row in df.iterrows():
            self.tree.insert('', 'end', values=[row[c] for c in cols])

    def _plot_lmm_results(self, results_df):
        self._clear_plots()
        if results_df.empty:
            return
        fig = Figure(figsize=(12,8), dpi=100)
        ax1 = fig.add_subplot(2,1,1)
        sign = results_df[results_df['significant']].copy()
        if sign.empty:
            ax1.text(0.5, 0.5, f"Нет значимых результатов (q < {self.fdr_threshold.get()})",
                     transform=ax1.transAxes, ha='center')
        else:
            top = sign.nsmallest(10, 'q_value')
            y_pos = np.arange(len(top))
            if 'channel' in top.columns:
                labels = top['channel'] + ' (DFA)'
            else:
                labels = top['pair'] + ' (' + top['band'] + ')'
            ax1.errorbar(top['beta'], y_pos,
                         xerr=[top['beta']-top['ci_low'], top['ci_high']-top['beta']],
                         fmt='o', capsize=5, color='blue', ecolor='gray')
            ax1.axvline(x=0, linestyle='--', color='gray')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.set_xlabel('Beta коэффициент (эффект апноэ)')
            ax1.set_title(f'Топ-10 наиболее значимых признаков (FDR < {self.fdr_threshold.get()})')
        ax2 = fig.add_subplot(2,1,2)
        colors = np.where(results_df['significant'], 'red', 'gray')
        ax2.scatter(results_df['beta'], -np.log10(results_df['p_value']), c=colors, alpha=0.6, s=20)
        ax2.axhline(y=-np.log10(0.05), linestyle='--', color='blue', label='p=0.05')
        ax2.axhline(y=-np.log10(self.fdr_threshold.get()), linestyle='--', color='green', label='FDR threshold')
        ax2.set_xlabel('Beta coefficient')
        ax2.set_ylabel('-log10(p-value)')
        ax2.set_title('Volcano plot')
        ax2.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig

    # ------------------------------------------------------------
    # DFA ANOVA по тяжести ОАС (п.2.5.7)
    # ------------------------------------------------------------
    def run_dfa_anova(self):
        if self.epochs_df is None:
            messagebox.showwarning("Нет данных", "Сначала выполните LMM анализ или загрузите данные.")
            return
        # Определяем группы тяжести на основе отфильтрованных данных
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None:
            self.log("Нет отфильтрованных данных для определения тяжести.")
            return
        severity_map = filtered_df.set_index('study_id')['breathing_impairment_severity'].to_dict()
        # Оставляем только основные группы
        valid = ['no_impairment', 'mild', 'moderate', 'severe']
        df = self.epochs_df.copy()
        df['severity'] = df['study_id'].map(severity_map)
        df = df[df['severity'].isin(valid)]
        if df.empty:
            self.log("Нет данных с допустимыми группами тяжести.")
            return
        # Получаем список каналов DFA
        channels = [self.dfa_listbox.get(i) for i in self.dfa_listbox.curselection()] or self.dfa_channels
        results = []
        for ch in channels:
            col = f"{ch}_dfa"
            if col not in df.columns:
                continue
            # Агрегируем по пациенту: средний DFA по всем эпохам пациента
            patient_means = df.groupby(['patient_id', 'severity'])[col].mean().reset_index()
            groups = []
            for sev in valid:
                vals = patient_means[patient_means['severity'] == sev][col].dropna().values
                if len(vals) > 0:
                    groups.append(vals)
            if len(groups) < 2:
                self.log(f"Для {ch} недостаточно групп для ANOVA.")
                continue
            # Проверка нормальности (можно пропустить)
            # Однофакторный ANOVA
            f_stat, p_val = f_oneway(*groups)
            # Пост-хок Tukey HSD
            posthoc = None
            if p_val < 0.05:
                # Собираем все данные для Tukey
                all_vals = []
                all_sev = []
                for sev, vals in zip(valid, groups):
                    if len(vals) > 0:
                        all_vals.extend(vals)
                        all_sev.extend([sev]*len(vals))
                if len(all_vals) > 0:
                    tukey = pairwise_tukeyhsd(all_vals, all_sev, alpha=0.05)
                    posthoc = tukey.summary()
            results.append({
                'channel': ch,
                'f_stat': f_stat,
                'p_value': p_val,
                'significant': p_val < self.fdr_threshold.get(),
                'posthoc': posthoc
            })
        if not results:
            self.log("Не удалось выполнить ANOVA ни для одного канала.")
            return
        # FDR коррекция
        pvals = [r['p_value'] for r in results]
        qvals = false_discovery_control(pvals, method='bh')
        for i, r in enumerate(results):
            r['q_value'] = qvals[i]
            r['significant_fdr'] = qvals[i] < self.fdr_threshold.get()
        self.dfa_anova_results = results
        # Отображение в текстовом виджете
        self.anova_text.delete(1.0, tk.END)
        self.anova_text.insert(tk.END, "=== ANOVA для DFA-экспонента по группам тяжести ОАС ===\n")
        self.anova_text.insert(tk.END, f"Группы: {', '.join(valid)}\n")
        self.anova_text.insert(tk.END, f"FDR порог: q = {self.fdr_threshold.get()}\n\n")
        for r in results:
            self.anova_text.insert(tk.END, f"Канал {r['channel']}: F = {r['f_stat']:.3f}, p = {r['p_value']:.4e}, q = {r['q_value']:.4f}\n")
            if r['significant_fdr']:
                self.anova_text.insert(tk.END, "  -> Значимые различия между группами (q<{})\n".format(self.fdr_threshold.get()))
                if r['posthoc'] is not None:
                    self.anova_text.insert(tk.END, "  Post-hoc Tukey HSD:\n")
                    self.anova_text.insert(tk.END, str(r['posthoc']) + "\n")
            else:
                self.anova_text.insert(tk.END, "  -> Нет значимых различий\n")
            self.anova_text.insert(tk.END, "\n")
        self.right_notebook.select(self.tab_anova)
        self.log("ANOVA для DFA завершена.")

    # ------------------------------------------------------------
    # Когерентность: тепловая матрица (средние значения)
    # ------------------------------------------------------------
    def _get_available_groups_for_coherence(self):
        """Возвращает список групп тяжести, присутствующих в данных."""
        if self.epochs_df is None:
            return []
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None:
            return []
        severity_map = filtered_df.set_index('study_id')['breathing_impairment_severity'].to_dict()
        df = self.epochs_df.copy()
        df['severity'] = df['study_id'].map(severity_map)
        valid = ['no_impairment', 'mild', 'moderate', 'severe']
        present = [s for s in valid if s in df['severity'].values]
        return present

    def show_coherence_matrix(self):
        """Показывает тепловую карту когерентности для выбранной группы и диапазона."""
        if self.results_df is None or self.analysis_mode.get() != 'coh':
            messagebox.showwarning("Нет данных", "Сначала выполните LMM анализ для когерентности.")
            return
        group = self.matrix_group.get()
        band = self.matrix_band.get()
        if not group or not band:
            messagebox.showwarning("Выбор", "Выберите группу и частотный диапазон.")
            return
        # Загружаем исходные данные для вычисления средних когерентностей по группе
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None:
            return
        severity_map = filtered_df.set_index('study_id')['breathing_impairment_severity'].to_dict()
        df = self.epochs_df.copy()
        df['severity'] = df['study_id'].map(severity_map)
        df = df[df['severity'] == group]
        if df.empty:
            self.log(f"Нет данных для группы {group}")
            return
        # Список всех пар и диапазонов (все возможные)
        all_pairs = [f"{a}-{b}" for a,b in self.coh_pairs]
        all_bands = self.coh_bands
        # Матрица 8x6 (пары x диапазоны) со средними значениями
        mean_matrix = np.zeros((len(all_pairs), len(all_bands)))
        for i, pair in enumerate(all_pairs):
            pair_clean = pair.replace('-','')
            for j, b in enumerate(all_bands):
                col = f"{pair_clean}_coh_{b}"
                if col in df.columns:
                    mean_matrix[i,j] = df[col].mean()
                else:
                    mean_matrix[i,j] = np.nan
        # Построение тепловой карты
        fig = Figure(figsize=(10,6), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(mean_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_xticks(np.arange(len(all_bands)))
        ax.set_xticklabels(all_bands, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(all_pairs)))
        ax.set_yticklabels(all_pairs)
        ax.set_title(f"Средняя когерентность (группа: {group}, диапазон: {band})")
        fig.colorbar(im, ax=ax, label='Когерентность')
        # Дополнительно можно выделить выбранный диапазон столбцом
        if band in all_bands:
            idx_band = all_bands.index(band)
            ax.axvline(x=idx_band-0.5, color='red', linewidth=2, linestyle='--')
            ax.axvline(x=idx_band+0.5, color='red', linewidth=2, linestyle='--')
        # Очистка и отображение
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.matrix_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.coherence_matrix_figure = fig
        self.right_notebook.select(self.tab_coherence_matrix)
        self.log(f"Матрица когерентности для {group}, {band} построена.")

    # ------------------------------------------------------------
    # Карта мозга для когерентности
    # ------------------------------------------------------------
    def update_coherence_map(self):
        """Перерисовывает карту связей для выбранного диапазона (эффект апноэ)."""
        if self.analysis_mode.get() != 'coh' or self.results_df is None:
            return
        band = self.current_coherence_band.get()
        band_df = self.results_df[self.results_df['band'] == band].copy()
        coords = {
            'F3': (0.2, 0.8), 'F4': (0.8, 0.8),
            'C3': (0.2, 0.5), 'C4': (0.8, 0.5),
            'O1': (0.2, 0.2), 'O2': (0.8, 0.2)
        }
        fig = Figure(figsize=(7,5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_aspect('equal')
        ax.axis('off')
        # Электроды
        for name, (x,y) in coords.items():
            circle = Circle((x,y), radius=0.04, facecolor='lightgray', edgecolor='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(x, y-0.05, name, ha='center', va='top', fontsize=9)
        if not band_df.empty:
            # Определяем границы для цвета (симметричные, чтобы 0 был посередине)
            max_abs_beta = max(abs(band_df['beta'].min()), abs(band_df['beta'].max()))
            norm = plt.Normalize(vmin=-max_abs_beta, vmax=max_abs_beta)
            for _, row in band_df.iterrows():
                pair = row['pair']
                a,b = pair.split('-')
                if a not in coords or b not in coords:
                    continue
                x1,y1 = coords[a]
                x2,y2 = coords[b]
                beta = row['beta']
                q = row['q_value']
                color = plt.cm.RdBu_r(norm(beta))
                linewidth = 1 + abs(beta)*3
                alpha = 0.7
                ax.plot([x1,x2], [y1,y2], color=color, linewidth=linewidth, alpha=alpha)
                if q < self.fdr_threshold.get():
                    mx, my = (x1+x2)/2, (y1+y2)/2
                    ax.text(mx, my, '*', fontsize=14, ha='center', va='center', color='gold', weight='bold')
            sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
            cbar.set_label('Beta (эффект апноэ)')
        else:
            ax.text(0.5, 0.5, f"Нет значимых результатов для диапазона {band}", ha='center', va='center')
        fig.suptitle(f'Карта когерентности: {band}\n(толщина ~ |beta|, звёздочка – значимо q<{self.fdr_threshold.get()})')
        for widget in self.map_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.map_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.coherence_map_figure = fig

    # ------------------------------------------------------------
    # Диагностика и бутстрап (оставлены без изменений, но для краткости не переписываю)
    # ------------------------------------------------------------
    def _update_norm_selector(self):
        if self.results_df is None or self.results_df.empty:
            self.norm_selector['values'] = []
            self.norm_btn.config(state=tk.DISABLED)
            return
        if self.analysis_mode.get() == 'dfa':
            # Для DFA нужен столбец 'channel'
            if 'channel' not in self.results_df.columns:
                items = []
            else:
                items = [f"{row['channel']}_DFA" for _, row in self.results_df.iterrows()]
        else:
            # Для когерентности нужны столбцы 'pair' и 'band'
            if 'pair' not in self.results_df.columns or 'band' not in self.results_df.columns:
                items = []
            else:
                items = [f"{row['pair']}_{row['band']}" for _, row in self.results_df.iterrows()]
        self.norm_selector['values'] = items
        if items:
            self.norm_selector.set(items[0])
            self.norm_btn.config(state=tk.NORMAL)
        else:
            self.norm_btn.config(state=tk.DISABLED)

    def _update_diag_selector(self):
        if self.results_df is None or self.results_df.empty:
            self.diag_selector['values'] = []
            self.diag_btn.config(state=tk.DISABLED)
            return
        if self.analysis_mode.get() == 'dfa':
            if 'channel' not in self.results_df.columns:
                items = []
            else:
                items = [f"{row['channel']}_DFA" for _, row in self.results_df.iterrows()]
        else:
            if 'pair' not in self.results_df.columns or 'band' not in self.results_df.columns:
                items = []
            else:
                items = [f"{row['pair']}_{row['band']}" for _, row in self.results_df.iterrows()]
        self.diag_selector['values'] = items
        if items:
            self.diag_selector.set(items[0])
            self.diag_btn.config(state=tk.NORMAL)
        else:
            self.diag_btn.config(state=tk.DISABLED)

    def check_normality(self):
        if self.epochs_df is None:
            messagebox.showwarning("Нет данных", "Сначала выполните LMM анализ.")
            return
        selection = self.norm_selector.get()
        if not selection:
            return
        if self.analysis_mode.get() == 'dfa':
            channel = selection.split('_')[0]
            col = f"{channel}_dfa"
        else:
            pair_band = selection.split('_')
            pair = pair_band[0]
            band = pair_band[1]
            col = f"{pair.replace('-','')}_coh_{band}"
        if col not in self.epochs_df.columns:
            self.log(f"Столбец {col} не найден.")
            return
        data = self.epochs_df[col].dropna()
        if len(data) < 3:
            messagebox.showwarning("Недостаточно данных", f"n={len(data)}")
            return
        for widget in self.norm_plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(6,5))
        ax = fig.add_subplot(111)
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"Q-Q plot: {selection}")
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.norm_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.norm_figure = fig
        self.save_norm_plot_btn.config(state=tk.NORMAL)
        self.right_notebook.select(self.tab_norm)
        if len(data) <= 5000:
            stat, p = shapiro(data)
            self.log(f"Shapiro-Wilk: p={p:.4e}")
            messagebox.showinfo("Нормальность", f"Shapiro-Wilk p={p:.4e}\n{'Нормальное' if p>0.05 else 'Ненормальное'}")
        else:
            messagebox.showinfo("Нормальность", "Выборка >5000, тест не применялся.\nСмотрите Q-Q plot.")

    def save_norm_plot(self):
        if hasattr(self, 'norm_figure') and self.norm_figure:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if path:
                self.norm_figure.savefig(path, dpi=150, bbox_inches='tight')
                self.log(f"График нормальности сохранён в {path}")

    def run_diagnostics(self):
        if self.results_df is None:
            messagebox.showwarning("Нет модели", "Сначала выполните LMM анализ.")
            return
        selection = self.diag_selector.get()
        if not selection:
            return
        if self.analysis_mode.get() == 'dfa':
            # selection имеет вид "F3_DFA"
            channel = selection.split('_')[0]
            row = self.results_df[(self.results_df['channel'] == channel) & (self.results_df['feature'] == 'DFA')]
            if row.empty:
                self.log(f"Не найден признак {selection}")
                return
            feature_desc = f"{channel}_DFA"
        else:
            parts = selection.split('_')
            # selection имеет вид "F3-C3_delta" или подобное
            pair = parts[0]
            band = parts[1]
            row = self.results_df[(self.results_df['pair'] == pair) & (self.results_df['band'] == band)]
            if row.empty:
                self.log(f"Не найден признак {selection}")
                return
            feature_desc = f"{pair}_{band}"
        self.log(f"Запуск диагностики для {feature_desc}...")
        threading.Thread(target=self._diagnostics_thread, args=(feature_desc,), daemon=True).start()

    def _diagnostics_thread(self, feature_desc):
        if self.epochs_df is None:
            self.log("Нет загруженных эпох.")
            return
        if self.analysis_mode.get() == 'dfa':
            channel = feature_desc.split('_')[0]
            col = f"{channel}_dfa"
        else:
            parts = feature_desc.split('_')
            pair = parts[0]
            band = parts[1]
            col = f"{pair.replace('-','')}_coh_{band}"
        if col not in self.epochs_df.columns:
            self.log(f"Столбец {col} отсутствует.")
            return
        df = self.epochs_df.copy()
        df['has_apnea'] = df['has_apnea'].astype(bool)
        df['patient_id'] = df['patient_id'].astype(int)
        if 'epoch_stage' in df.columns:
            df['epoch_stage'] = df['epoch_stage'].astype('category')
        if self.include_covariates.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is not None:
                cov = filtered_df[['study_id','age_at_study','gender','bmi']].drop_duplicates(subset=['study_id'])
                cov['gender_code'] = (cov['gender'] == 'M').astype(int)
                df = df.merge(cov[['study_id','age_at_study','gender_code','bmi']], on='study_id', how='left')
        formula = f"{col} ~ has_apnea"
        if self.include_stage.get() and 'epoch_stage' in df.columns:
            formula += " + epoch_stage"
        if self.include_covariates.get():
            for c in ['age_at_study','gender_code','bmi']:
                if c in df.columns:
                    formula += f" + {c}"
        sub = df[[col, 'has_apnea', 'patient_id'] + [c for c in ['epoch_stage','age_at_study','gender_code','bmi'] if c in df.columns]].dropna()
        if len(sub) < 30:
            self.log("Недостаточно наблюдений для диагностики.")
            return
        try:
            model = smf.mixedlm(formula, sub, groups=sub['patient_id'])
            result = model.fit(reml=False, method='lbfgs', maxiter=1000)
            if hasattr(result, 'cov_re') and np.linalg.matrix_rank(result.cov_re) < result.cov_re.shape[0]:
                self.log("Сингулярная ковариация – диагностика пропущена.")
                return
            fitted = result.fittedvalues
            resid = result.resid
            shapiro_p = None
            if len(resid) <= 5000:
                _, shapiro_p = shapiro(resid)
            resid2 = resid**2
            import statsmodels.api as sm
            X = sm.add_constant(fitted)
            bp_model = sm.OLS(resid2, X).fit()
            bp_stat = bp_model.rsquared * len(resid)
            bp_p = 1 - chi2.cdf(bp_stat, df=1)
            self.last_diagnostics = {'fitted':fitted, 'residuals':resid, 'shapiro_p':shapiro_p, 'bp_p':bp_p, 'n':len(resid)}
            fig = Figure(figsize=(10,8))
            ax1 = fig.add_subplot(2,2,1)
            ax1.scatter(fitted, resid, alpha=0.5)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Предсказанные')
            ax1.set_ylabel('Остатки')
            ax2 = fig.add_subplot(2,2,2)
            stats.probplot(resid, dist="norm", plot=ax2)
            ax2.set_title('Q-Q plot остатков')
            ax3 = fig.add_subplot(2,2,3)
            ax3.hist(resid, bins=30, edgecolor='black')
            ax3.set_xlabel('Остатки')
            ax3.set_ylabel('Частота')
            ax4 = fig.add_subplot(2,2,4)
            ax4.text(0.1,0.9, f"n={len(resid)}", fontsize=10)
            ax4.text(0.1,0.8, f"Shapiro p={shapiro_p:.4f}" if shapiro_p else "Shapiro: n>5000", fontsize=10)
            ax4.text(0.1,0.7, f"BP p={bp_p:.4f}", fontsize=10)
            ax4.axis('off')
            fig.tight_layout()
            self.main_app.root.after(0, self._show_diagnostic_plot, fig)
            self.diag_figure = fig
            self.diag_model = result
            self.diag_feature_name = feature_desc
            self.all_diagnostics.append({
                'feature': feature_desc,
                'shapiro_p': shapiro_p,
                'bp_p': bp_p,
                'n': len(resid)
            })
            self.log(f"Диагностика завершена. Shapiro p={shapiro_p}, BP p={bp_p:.4f}")
            self.right_notebook.select(self.tab_diag)
        except Exception as e:
            self.log(f"Ошибка диагностики: {e}")

    def _show_diagnostic_plot(self, fig):
        for widget in self.diag_plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.diag_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.save_diag_plot_btn.config(state=tk.NORMAL)

    def save_diag_plot(self):
        if hasattr(self, 'diag_figure') and self.diag_figure:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if path:
                self.diag_figure.savefig(path, dpi=150, bbox_inches='tight')
                self.log(f"График диагностики сохранён в {path}")

    def run_bootstrap(self):
        if self.diag_model is None:
            messagebox.showwarning("Нет модели", "Сначала выполните диагностику для выбранного признака.")
            return
        messagebox.showinfo("Бутстрап", "Будет выполнено 1000 итераций (блок-бутстрап по пациентам).\nЭто может занять 1-2 минуты.")
        threading.Thread(target=self._bootstrap_thread, daemon=True).start()

    def _bootstrap_thread(self):
        feature = self.diag_feature_name
        if self.analysis_mode.get() == 'dfa':
            channel = feature.split('_')[0]
            col = f"{channel}_dfa"
        else:
            parts = feature.split('_')
            pair = parts[0]
            band = parts[1]
            col = f"{pair.replace('-','')}_coh_{band}"
        df = self.epochs_df.copy()
        df['has_apnea'] = df['has_apnea'].astype(bool)
        df['patient_id'] = df['patient_id'].astype(int)
        if 'epoch_stage' in df.columns:
            df['epoch_stage'] = df['epoch_stage'].astype('category')
        if self.include_covariates.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is not None:
                cov = filtered_df[['study_id','age_at_study','gender','bmi']].drop_duplicates(subset=['study_id'])
                cov['gender_code'] = (cov['gender'] == 'M').astype(int)
                df = df.merge(cov[['study_id','age_at_study','gender_code','bmi']], on='study_id', how='left')
        formula = f"{col} ~ has_apnea"
        if self.include_stage.get() and 'epoch_stage' in df.columns:
            formula += " + epoch_stage"
        if self.include_covariates.get():
            for c in ['age_at_study','gender_code','bmi']:
                if c in df.columns:
                    formula += f" + {c}"
        use_cols = [col, 'has_apnea', 'patient_id']
        if self.include_stage.get() and 'epoch_stage' in df.columns:
            use_cols.append('epoch_stage')
        if self.include_covariates.get():
            for c in ['age_at_study','gender_code','bmi']:
                if c in df.columns:
                    use_cols.append(c)
        sub = df[use_cols].dropna()
        patient_ids = sub['patient_id'].unique()
        n_patients = len(patient_ids)
        if n_patients < 10:
            self.log("Слишком мало пациентов для бутстрапа.")
            return
        n_iter = 1000
        betas = []
        self.log(f"Бутстрап: {n_iter} итераций, пациентов={n_patients}")
        for i in range(n_iter):
            if self.stop_flag:
                break
            boot_patients = np.random.choice(patient_ids, size=n_patients, replace=True)
            boot_df = pd.concat([sub[sub['patient_id'] == pid] for pid in boot_patients], ignore_index=True)
            try:
                model_boot = smf.mixedlm(formula, boot_df, groups=boot_df['patient_id'])
                result_boot = model_boot.fit(reml=False, method='lbfgs', maxiter=1000)
                beta = result_boot.params.get('has_apnea[T.True]', result_boot.params.get('has_apnea', None))
                if beta is not None:
                    betas.append(beta)
            except Exception:
                pass
            if (i+1) % 100 == 0:
                self.log(f"Бутстрап: {i+1}/{n_iter} итераций")
        if betas:
            ci_low = np.percentile(betas, 2.5)
            ci_high = np.percentile(betas, 97.5)
            p_bootstrap = (np.sum(np.abs(betas) < 1e-8) * 2) / len(betas)
            self.bootstrap_info = {
                'feature': feature,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_bootstrap': p_bootstrap,
                'n_iter': len(betas)
            }
            self.log(f"Бутстрап CI: [{ci_low:.4f}, {ci_high:.4f}], p≈{p_bootstrap:.4f}")
            fig = Figure(figsize=(8,5))
            ax = fig.add_subplot(111)
            ax.hist(betas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=ci_low, color='red', linestyle='--', label=f'2.5%: {ci_low:.2f}')
            ax.axvline(x=ci_high, color='red', linestyle='--', label=f'97.5%: {ci_high:.2f}')
            ax.axvline(x=np.mean(betas), color='green', linestyle='-', label=f'Mean: {np.mean(betas):.2f}')
            ax.set_xlabel('Beta')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Bootstrap distribution: {feature}')
            ax.legend()
            self.main_app.root.after(0, self._show_diagnostic_plot, fig)
            self.save_bootstrap_btn.config(state=tk.NORMAL)
        else:
            self.log("Бутстрап не дал результатов.")

    def save_bootstrap_csv(self):
        if not self.bootstrap_info:
            messagebox.showwarning("Нет данных", "Сначала выполните бутстрап.")
            return
        df = pd.DataFrame([self.bootstrap_info])
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            df.to_csv(path, index=False)
            self.log(f"Бутстрап информация сохранена в {path}")

    # ------------------------------------------------------------
    # Сохранение и отчёт
    # ------------------------------------------------------------
    def save_results_csv(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Нет результатов для сохранения.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.results_df.to_csv(path, index=False)
            self.log(f"Результаты сохранены в {path}")

    def save_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "График не построен.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if path:
            self.current_figure.savefig(path, dpi=150, bbox_inches='tight')
            self.log(f"График сохранён в {path}")

    def generate_report(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Сначала выполните LMM анализ.")
            return
        filtered_df = self.main_app.get_filtered_data()
        n_patients = filtered_df['patient_id'].nunique() if filtered_df is not None else 0
        n_epochs = len(self.epochs_df) if self.epochs_df is not None else 0
        sign = self.results_df[self.results_df['significant']].copy()
        data_type_label = {1: "Тонический", 2: "Все эпохи", 3: "Фильтр по положению"}
        data_type_text = data_type_label.get(self.data_type.get(), "Неизвестно")
        mode = self.analysis_mode.get()
        # Гипотезы
        hypotheses_html = "<h2>Проверка гипотез (п.2.5.7)</h2>"
        if mode == 'dfa':
            hypotheses_html += "<h3>DFA (детрендовый флуктуационный анализ)</h3>"
            hypotheses_html += "<p>Ожидается снижение DFA-экспонента в лобных отведениях при апноэ.</p>"
            if not sign.empty:
                hypotheses_html += f"<p>✅ Подтверждена: {len(sign)} значимых эффектов (q<{self.fdr_threshold.get()}).</p>"
            else:
                hypotheses_html += "<p>❌ Не подтверждена: значимых эффектов не обнаружено.</p>"
            # Добавим результаты ANOVA
            if self.dfa_anova_results:
                hypotheses_html += "<h3>ANOVA для DFA по тяжести ОАС</h3>"
                hypotheses_html += "<table border='1'><tr><th>Канал</th><th>F</th><th>p-value</th><th>q-value</th><th>Значимо (q<{})</th></tr>".format(self.fdr_threshold.get())
                for r in self.dfa_anova_results:
                    hypotheses_html += f"<tr><td>{r['channel']}</td><td>{r['f_stat']:.3f}</td><td>{r['p_value']:.4e}</td><td>{r['q_value']:.4f}</td><td>{'Да' if r['significant_fdr'] else 'Нет'}</td></tr>"
                hypotheses_html += "</table>"
        else:
            hypotheses_html += "<h3>Когерентность ЭЭГ</h3>"
            hypotheses_html += "<p>Ожидается снижение когерентности в медленных диапазонах (δ,θ) при апноэ.</p>"
            if not sign.empty:
                hypotheses_html += f"<p>✅ Значимые изменения когерентности в {sign['band'].nunique()} диапазонах.</p>"
            else:
                hypotheses_html += "<p>❌ Значимых изменений не обнаружено.</p>"
        # Диагностика
        diag_html = ""
        if self.all_diagnostics:
            diag_html = "<h2>Диагностика остатков LMM</h2><table border='1'><tr><th>Признак</th><th>n</th><th>Shapiro p</th><th>BP p</th><tr>"
            for d in self.all_diagnostics[:5]:
                shapiro_str = f"{d['shapiro_p']:.4f}" if d['shapiro_p'] is not None else '>5000'
                diag_html += f"<tr><td>{d['feature']}</td><td>{d['n']}</td><td>{shapiro_str}</td><td>{d['bp_p']:.4f}</td></tr>"
            diag_html += "</table>"
        # Бутстрап
        bootstrap_html = ""
        if self.bootstrap_info:
            bi = self.bootstrap_info
            bootstrap_html = f"<h2>Бутстрап (блок-бутстрап по пациентам)</h2><p>Признак: {bi['feature']}, итераций: {bi['n_iter']}<br>95% CI: [{bi['ci_low']:.4f}, {bi['ci_high']:.4f}], p={bi['p_bootstrap']:.4f}</p>"
        # Графики LMM
        plot_html = ""
        if self.current_figure:
            buf = io.BytesIO()
            self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
        # Карта когерентности
        map_html = ""
        if mode == 'coh' and self.coherence_map_figure:
            buf = io.BytesIO()
            self.coherence_map_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            map_html = f'<h2>Карта когерентности (диапазон: {self.current_coherence_band.get()})</h2><div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
        # Матрица когерентности (если есть)
        matrix_html = ""
        if mode == 'coh' and self.coherence_matrix_figure:
            buf = io.BytesIO()
            self.coherence_matrix_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            matrix_html = f'<h2>Матрица когерентности (группа: {self.matrix_group.get()}, диапазон: {self.matrix_band.get()})</h2><div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
        # Таблица значимых
        table_html = "<h2>Значимые признаки (LMM)</h2><table border='1'><tr><th>Признак</th><th>Beta</th><th>p-value</th><th>q-value</th><th>95% CI</th></tr>"
        for _, r in sign.iterrows():
            if 'channel' in r:
                label = f"{r['channel']} DFA"
            else:
                label = f"{r['pair']} {r['band']}"
            table_html += f"<tr><td>{label}</td><td>{r['beta']:.4f}</td><td>{r['p_value']:.2e}</td><td>{r['q_value']:.4f}</td><td>[{r['ci_low']:.2f},{r['ci_high']:.2f}]</td></tr>"
        table_html += "</table>"
        # Полный HTML
        html = f"""
        <!DOCTYPE html>
        <html><head><meta charset="utf-8"><title>Отчёт DFA/когерентность</title>
        <style>body {{ font-family: Arial; margin:20px; }} table {{ border-collapse: collapse; width:100%; }} th,td {{ border:1px solid #ddd; padding:8px; }} .plot {{ margin:20px 0; }}</style>
        </head><body>
        <h1>LMM анализ DFA и когерентности ЭЭГ (п.2.5.7)</h1>
        <p>Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Пациентов: {n_patients}, эпох: {n_epochs}</p>
        <p>Тип набора: {data_type_text}, FDR q={self.fdr_threshold.get()}</p>
        {hypotheses_html}
        {table_html}
        {plot_html}
        {map_html}
        {matrix_html}
        {diag_html}
        {bootstrap_html}
        <p><em>Примечание:</em> Для DFA – снижение экспонента при апноэ свидетельствует о дезорганизации ритма. Для когерентности – синие линии на карте соответствуют отрицательному эффекту (снижение связности), красные – положительному.</p>
        </body></html>
        """
        fd, path = tempfile.mkstemp(suffix='.html', prefix='dfa_coh_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт: {path}")

    def _save_dfa_selection(self, event=None):
        self.saved_dfa_indices = self.dfa_listbox.curselection()

    def _save_pair_selection(self, event=None):
        self.saved_pair_indices = self.pairs_listbox.curselection()

    def _save_band_selection(self, event=None):
        self.saved_band_indices = self.bands_listbox.curselection()

    def _clear_results(self):
        """Сбрасывает результаты LMM при смене режима."""
        self.results_df = None
        self.current_figure = None
        self.coherence_map_figure = None
        self.coherence_matrix_figure = None
        self.all_diagnostics = []
        self.bootstrap_info = None
        self.dfa_anova_results = None
        self._clear_table()
        self._clear_plots()
        # Очистить фреймы карт
        for widget in self.map_frame.winfo_children():
            widget.destroy()
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        # Очистить текст в ANOVA
        if hasattr(self, 'anova_text'):
            self.anova_text.delete(1.0, tk.END)