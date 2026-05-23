# ui/tab_lmm.py
"""
Вкладка LMM (линейные смешанные модели) для выявления ЭЭГ-биомаркеров ОАС.
Реализация в соответствии с главой 2:
- LMM для всех комбинаций канал × признак с ковариатами (возраст, пол, ИМТ)
- FDR‑коррекция множественных сравнений
- Диагностика остатков для выбранной модели (Q‑Q, гомоскедастичность)
- Блок‑бутстрап по пациентам (1000 итераций) для робастных CI
- Генерация подробного отчёта (HTML + встроенные графики)
- Сохранение результатов нормальности и бутстрапа в CSV
- НОВОЕ: режим "Тонические признаки vs AHI" (хронический эффект тяжести ОАС)
"""

import tkinter as tk
from tkinter import messagebox, ttk
import warnings
import threading
import shutil
import os
import tempfile
import webbrowser
import base64
import io
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import false_discovery_control, shapiro, chi2
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui.base_tab import BaseTab
from core.api_client import get_epochs
from core.config import CACHE_API_DIR

# Подавляем назойливые предупреждения statsmodels
warnings.filterwarnings("ignore", module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=Warning, module="statsmodels.regression")


class LMMAnalysisTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        # ---- Параметры модели ----
        self.include_stage = tk.BooleanVar(value=True)
        self.include_covariates = tk.BooleanVar(value=True)
        self.fdr_threshold = tk.DoubleVar(value=0.05)
        self.use_cache = tk.BooleanVar(value=True)

        # НОВОЕ: выбор типа анализа LMM
        self.analysis_mode = tk.StringVar(value="compare_epochs")  # "compare_epochs" или "tonic_vs_ahi"

        # ---- Данные ----
        self.results_df = None
        self.current_figure = None
        self.stop_flag = False
        self.epochs_df = None
        self.epochs_loaded = False

        # ---- Хранилища результатов ----
        self.normality_results = {}
        self.bootstrap_results = {}
        self.diag_model = None
        self.diag_feature_name = ""
        self.norm_figure = None
        self.diag_figure = None
        self.all_diagnostics = []
        self.bootstrap_info = None

        # ---- Признаки ----
        self.all_channels = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        self.all_features = [
            'mean', 'std', 'min', 'max', 'range', 'rms',
            'abs_delta', 'rel_delta', 'abs_theta', 'rel_theta',
            'abs_alpha', 'rel_alpha', 'abs_sigma', 'rel_sigma',
            'abs_beta', 'rel_beta', 'tbr', 'dar', 'se50', 'gamma_power', 'sampen'
        ]
        self.coh_pairs = [('F3','C3'),('F3','F4'),('F3','C4'),('C3','C4'),('C3','O1'),
                          ('F4','C4'),('O1','O2'),('C3','O2')]
        self.coh_bands = ['delta','theta','alpha','sigma','beta','gamma']

        self._create_widgets()

    # --------------------------------------------------------------
    # Построение интерфейса
    # --------------------------------------------------------------
    def _create_widgets(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)

        # ---------- Левая панель ----------
        info_frame = ttk.LabelFrame(left_frame, text="Источник данных", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, text="Используются отфильтрованные данные из вкладки 'Загрузка'",
                  foreground="blue").pack(anchor=tk.W)
        ttk.Label(info_frame, text="(токен и URL не требуются)").pack(anchor=tk.W)

        # Пояснение о LMM
        lmm_desc_frame = ttk.LabelFrame(left_frame, text="Что делает LMM?", padding=5)
        lmm_desc_frame.pack(fill=tk.X, padx=5, pady=5)
        desc_text = ("Линейные смешанные модели (LMM) учитывают иерархическую структуру данных. "
                     "В режиме сравнения эпох оценивается острый эффект апноэ (beta). "
                     "В режиме тонические vs AHI оценивается вклад хронической тяжести (AHI).")
        ttk.Label(lmm_desc_frame, text=desc_text, justify=tk.LEFT, wraplength=600).pack(anchor=tk.W, fill=tk.X, padx=5, pady=2)
        ttk.Button(lmm_desc_frame, text="Показать инструкцию", command=self.show_instructions).pack(anchor=tk.W, padx=5, pady=2)

        model_frame = ttk.LabelFrame(left_frame, text="Настройки LMM", padding=5)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(model_frame, text="Включить стадию сна (N2/N3) как фактор",
                        variable=self.include_stage).pack(anchor=tk.W)
        ttk.Checkbutton(model_frame, text="Включить ковариаты (возраст, пол, ИМТ)",
                        variable=self.include_covariates).pack(anchor=tk.W)
        ttk.Label(model_frame, text="Порог FDR (q):").pack(anchor=tk.W)
        ttk.Entry(model_frame, textvariable=self.fdr_threshold, width=6).pack(anchor=tk.W)
        self.exclude_central_mixed = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_frame, text="Исключить центральное/смешанное апноэ",
                        variable=self.exclude_central_mixed).pack(anchor=tk.W)

        # ---- Выбор типа набора эпох (только для режима сравнения) ----
        data_type_frame = ttk.Frame(model_frame)
        data_type_frame.pack(fill=tk.X, pady=5)
        # выбор режима анализа
        mode_frame = ttk.LabelFrame(left_frame, text="Режим LMM", padding=5)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Radiobutton(mode_frame, text="Сравнение эпох (апноэ vs без апноэ)",
                        variable=self.analysis_mode, value="compare_epochs").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Тонические признаки vs AHI (хронический эффект)",
                        variable=self.analysis_mode, value="tonic_vs_ahi").pack(anchor=tk.W)



        cache_frame = ttk.LabelFrame(left_frame, text="Кэширование", padding=5)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cache_frame, text="Использовать кэш API (ускоряет повторные запуски)",
                        variable=self.use_cache).pack(anchor=tk.W)
        ttk.Button(cache_frame, text="Очистить кэш API", command=self.clear_api_cache).pack(anchor=tk.W, pady=2)

        # ----- Проверка нормальности (без изменений) -----
        norm_frame = ttk.LabelFrame(left_frame, text="Проверка нормальности", padding=5)
        norm_frame.pack(fill=tk.X, padx=5, pady=5)
        row1 = ttk.Frame(norm_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Канал:").pack(side=tk.LEFT, padx=2)
        self.norm_channel = tk.StringVar(value='C3')
        ttk.Combobox(row1, textvariable=self.norm_channel, values=self.all_channels, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="Признак:").pack(side=tk.LEFT, padx=2)
        self.norm_feature = tk.StringVar(value='mean')
        ttk.Combobox(row1, textvariable=self.norm_feature, values=self.all_features, width=12).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(norm_frame)
        row2.pack(fill=tk.X, pady=2)
        self.norm_btn = ttk.Button(row2, text="Проверить нормальность", command=self.check_normality, state=tk.DISABLED)
        self.norm_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Сохранить нормальность (CSV)", command=self.save_normality_csv).pack(side=tk.LEFT, padx=2)
        self.save_norm_plot_btn = ttk.Button(row2, text="Сохранить график (PNG)", command=self.save_norm_plot, state=tk.DISABLED)
        self.save_norm_plot_btn.pack(side=tk.LEFT, padx=2)

        # ----- Диагностика модели (без изменений) -----
        diag_frame = ttk.LabelFrame(left_frame, text="Диагностика модели (2.5.3)", padding=5)
        diag_frame.pack(fill=tk.X, padx=5, pady=5)
        row1d = ttk.Frame(diag_frame)
        row1d.pack(fill=tk.X, pady=2)
        ttk.Label(row1d, text="Канал:").pack(side=tk.LEFT, padx=2)
        self.diag_channel = tk.StringVar()
        ttk.Combobox(row1d, textvariable=self.diag_channel, values=self.all_channels, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1d, text="Признак:").pack(side=tk.LEFT, padx=2)
        self.diag_feature = tk.StringVar()
        ttk.Combobox(row1d, textvariable=self.diag_feature, values=self.all_features, width=12).pack(side=tk.LEFT, padx=2)
        row2d = ttk.Frame(diag_frame)
        row2d.pack(fill=tk.X, pady=2)
        self.diag_btn = ttk.Button(row2d, text="Диагностика модели", command=self.run_diagnostics, state=tk.DISABLED)
        self.diag_btn.pack(side=tk.LEFT, padx=2)
        self.save_diag_btn = ttk.Button(row2d, text="Сохранить диагностику (CSV)", command=self.save_diagnostics_csv, state=tk.DISABLED)
        self.save_diag_btn.pack(side=tk.LEFT, padx=2)
        self.save_diag_plot_btn = ttk.Button(row2d, text="Сохранить график (PNG)", command=self.save_diag_plot, state=tk.DISABLED)
        self.save_diag_plot_btn.pack(side=tk.LEFT, padx=2)
        row3d = ttk.Frame(diag_frame)
        row3d.pack(fill=tk.X, pady=2)
        self.bootstrap_btn = ttk.Button(row3d, text="Бутстрап (1000 итераций)", command=self.run_bootstrap, state=tk.DISABLED)
        self.bootstrap_btn.pack(side=tk.LEFT, padx=2)
        self.save_bootstrap_btn = ttk.Button(row3d, text="Сохранить бутстрап (CSV)", command=self.save_bootstrap_csv, state=tk.DISABLED)
        self.save_bootstrap_btn.pack(side=tk.LEFT, padx=2)
        self.fast_diag_btn = ttk.Button(row3d, text="Быстрая диагностика топ-10", command=self.fast_diagnostics, state=tk.DISABLED)
        self.fast_diag_btn.pack(side=tk.LEFT, padx=2)

        # ---- Основные кнопки управления ----
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Запустить LMM", command=self.run_lmm)
        self.run_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить CSV", command=self.save_results_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=2)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=2)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать отчёт", command=self.generate_report, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=2)

        # ---------- Правая панель (внутренний Notebook) ----------
        self.right_notebook = ttk.Notebook(right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_table = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_table, text="Результаты LMM")
        self.tree_frame = ttk.Frame(self.tab_table)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(self.tree_frame)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tab_plots = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_plots, text="Графики LMM")
        self.plot_frame = ttk.Frame(self.tab_plots)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_norm = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_norm, text="Проверка нормальности")
        self.norm_plot_frame = ttk.Frame(self.tab_norm)
        self.norm_plot_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_diag = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_diag, text="Диагностика модели")
        self.diag_plot_frame = ttk.Frame(self.tab_diag)
        self.diag_plot_frame.pack(fill=tk.BOTH, expand=True)

    # --------------------------------------------------------------
    # Вспомогательные методы (без изменений)
    # --------------------------------------------------------------
    def log(self, msg):
        self.main_app.log(msg)

    def stop_analysis(self):
        self.stop_flag = True
        self.log("Остановка LMM запрошена...")

    def clear_api_cache(self):
        if os.path.exists(CACHE_API_DIR):
            try:
                shutil.rmtree(CACHE_API_DIR)
                os.makedirs(CACHE_API_DIR, exist_ok=True)
                self.log("Кэш API полностью очищен.")
                messagebox.showinfo("Кэш очищен", "Все файлы кэша API удалены.")
            except Exception as e:
                self.log(f"Ошибка очистки кэша: {e}")
                messagebox.showerror("Ошибка", f"Не удалось очистить кэш: {e}")
        else:
            os.makedirs(CACHE_API_DIR, exist_ok=True)

    def show_instructions(self):
        msg = (
            "ИНСТРУКЦИЯ ПО LMM АНАЛИЗУ\n"
            "================================\n"
            "1. Сначала загрузите и отфильтруйте данные на вкладке 'Загрузка и фильтры'.\n"
            "2. Выберите режим LMM:\n"
            "   - 'Сравнение эпох' – проверяет острый эффект апноэ (has_apnea).\n"
            "   - 'Тонические признаки vs AHI' – оценивает хроническое влияние тяжести ОАС на фоновую ЭЭГ.\n"
            "3. Нажмите 'Запустить LMM'.\n"
            "4. После завершения появится таблица, forest plot и volcano plot.\n"
            "5. Диагностика и бутстрап доступны для выбранного признака.\n"
            "6. Отчёт включает интерпретацию в соответствии с выбранным режимом.\n"
        )
        messagebox.showinfo("Инструкция", msg)

    # --------------------------------------------------------------
    # Загрузка эпох (добавлен параметр data_type)
    # --------------------------------------------------------------
    def _load_epochs(self, study_ids, force_reload=False, data_type_override=None):
        if not force_reload and self.epochs_df is not None:
            return self.epochs_df

        data_type = data_type_override if data_type_override is not None else self.data_type.get()
        def update_progress(page, total, _):
            if total > 0:
                self.main_app.set_progress(int(page / total * 100))

        self.main_app.set_progress(0)
        self.log(f"Загрузка эпох из API (data_type={data_type})...")
        epochs = get_epochs(
            self.main_app.tabs['load'].api_url.get(),
            self.main_app.tabs['load'].token.get(),
            study_ids=study_ids,
            data_type=data_type,
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
        self.norm_btn.config(state=tk.NORMAL)
        return df

    # --------------------------------------------------------------
    # Проверка нормальности (без изменений)
    # --------------------------------------------------------------
    def check_normality(self):
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите и отфильтруйте данные.")
            return
        study_ids = filtered_df['study_id'].unique().tolist()
        if self.epochs_df is None:
            self.log("Загрузка эпох для проверки нормальности...")
            def load_and_check():
                df = self._load_epochs(study_ids)
                if df is not None:
                    self.main_app.root.after(0, self._perform_normality_check)
            threading.Thread(target=load_and_check, daemon=True).start()
        else:
            self._perform_normality_check()

    def _perform_normality_check(self):
        # тот же код, что и раньше (без изменений)
        if self.epochs_df is None:
            return
        channel = self.norm_channel.get()
        feature = self.norm_feature.get()
        col = f"{channel}_{feature}"
        if col not in self.epochs_df.columns:
            messagebox.showerror("Ошибка", f"Столбец {col} не найден.")
            return
        data = self.epochs_df[col].dropna()
        if len(data) < 3:
            messagebox.showwarning("Недостаточно данных", f"n={len(data)}")
            return
        if len(data) <= 5000:
            stat, p = shapiro(data)
            normal = p > 0.05
            self.normality_results[f"{channel}_{feature}"] = {'p_value': p, 'n': len(data), 'normal': normal}
            res = f"Шапиро-Уилк: W={stat:.4f}, p={p:.4e}\n"
            res += "Нормальное" if normal else "Ненормальное"
            messagebox.showinfo("Нормальность", res)
        else:
            self.normality_results[f"{channel}_{feature}"] = {'p_value': None, 'n': len(data), 'normal': None}
            messagebox.showinfo("Нормальность", "Выборка >5000, тест не применялся.\nСмотрите Q-Q plot.")
        for widget in self.norm_plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"Q-Q plot: {channel} {feature}")
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.norm_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.norm_figure = fig
        self.save_norm_plot_btn.config(state=tk.NORMAL)
        self.right_notebook.select(self.tab_norm)
        self.log(f"Проверка нормальности для {channel}_{feature} завершена.")

    def save_normality_csv(self):
        if not self.normality_results:
            messagebox.showwarning("Нет данных", "Сначала выполните хотя бы одну проверку нормальности.")
            return
        df = pd.DataFrame([
            {'feature': k, 'p_value': v['p_value'], 'n': v['n'], 'normal': v['normal']}
            for k, v in self.normality_results.items()
        ])
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            df.to_csv(path, index=False)
            self.log(f"Результаты нормальности сохранены в {path}")

    # --------------------------------------------------------------
    # Основной метод run_lmm (маршрутизация по режиму)
    # --------------------------------------------------------------
    def run_lmm(self):
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            messagebox.showerror("Ошибка", "Нет отфильтрованных данных.")
            return
        # Исключение центрального/смешанного апноэ применяется только для режима сравнения эпох
        if self.analysis_mode.get() == "compare_epochs" and self.exclude_central_mixed.get():
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
        self.fast_diag_btn.config(state=tk.DISABLED)
        self.save_diag_btn.config(state=tk.DISABLED)
        self.save_bootstrap_btn.config(state=tk.DISABLED)

        if self.analysis_mode.get() == "compare_epochs":
            threading.Thread(target=self._run_lmm_thread, args=(study_ids,), daemon=True).start()
        else:  # tonic_vs_ahi
            threading.Thread(target=self._run_tonic_vs_ahi_thread, args=(study_ids,), daemon=True).start()

    # --------------------------------------------------------------
    # Режим 1: сравнение эпох (апноэ vs без) - исходный код
    # --------------------------------------------------------------
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

            # Добавляем ковариаты
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

            # Проверка VIF
            vif_res = {}
            try:
                vif_res = self._check_vif(df)
                if vif_res:
                    high_vif = {k: v for k, v in vif_res.items() if v > 5}
                    if high_vif:
                        self.log(f"Предупреждение: мультиколлинеарность (VIF>5): {high_vif}")
            except Exception as e:
                self.log(f"Ошибка при вычислении VIF: {e}")
            self.last_vif = vif_res

            tasks = []
            for ch in self.all_channels:
                for feat in self.all_features:
                    col = f"{ch}_{feat}"
                    if col in df.columns:
                        tasks.append((ch, feat, col))
            for (a,b) in self.coh_pairs:
                pair = f"{a}{b}"
                for band in self.coh_bands:
                    col = f"{pair}_coh_{band}"
                    if col in df.columns:
                        tasks.append((f"{a}-{b}", band, col))

            if not tasks:
                self.log("Нет доступных признаков.")
                return
            self.log(f"Всего моделей: {len(tasks)}. Начинаем расчёт...")
            results = []
            total = len(tasks)
            for i, (ch, feat, col) in enumerate(tasks):
                if self.stop_flag:
                    break
                self.main_app.set_progress(int(i / total * 100))
                res = self._fit_lmm_model(df, col, ch, feat,
                                          self.include_stage.get(),
                                          self.include_covariates.get())
                if res:
                    results.append(res)
                if (i+1) % 30 == 0:
                    self.log(f"Обработано {i+1} из {total}")
            if not results:
                self.log("Нет результатов (модели не сошлись).")
                return
            res_df = pd.DataFrame(results)
            if res_df.empty:
                self.log("Нет результатов (модели не сошлись).")
                return

            res_df = res_df.dropna(subset=['p_value'])
            res_df = res_df[(res_df['p_value'] >= 0) & (res_df['p_value'] <= 1)]

            if res_df.empty:
                self.log("Нет корректных p-значений для FDR-коррекции.")
                return

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
            self.fast_diag_btn.config(state=tk.NORMAL)
            channels = sorted(res_df['channel'].unique())
            features = sorted(res_df['feature'].unique())
            if channels:
                self.diag_channel.set(channels[0])
            if features:
                self.diag_feature.set(features[0])
            n_sign = res_df['significant'].sum()
            self.log(f"LMM завершён. Значимых признаков: {n_sign} (q<{self.fdr_threshold.get()})")
            if n_sign > 0:
                pos = len(res_df[(res_df['significant']) & (res_df['beta'] > 0)])
                neg = n_sign - pos
                messagebox.showinfo("Результаты", f"Значимых признаков: {n_sign}\nИз них beta>0: {pos}, beta<0: {neg}")
        except Exception as e:
            self.log(f"Ошибка: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.main_app.set_progress(0)

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

    @staticmethod
    def _fit_lmm_model(df, col, channel, feature, include_stage, include_cov):
        cols = [col, 'has_apnea', 'patient_id']
        if include_stage and 'epoch_stage' in df.columns:
            cols.append('epoch_stage')
        if include_cov:
            for c in ['age_at_study', 'gender_code', 'bmi']:
                if c in df.columns:
                    cols.append(c)
        sub = df[cols].dropna()
        if len(sub) < 30:
            return None
        formula = f"{col} ~ has_apnea"
        if include_stage and 'epoch_stage' in sub.columns:
            formula += " + epoch_stage"
        if include_cov:
            for c in ['age_at_study', 'gender_code', 'bmi']:
                if c in sub.columns:
                    formula += f" + {c}"
        try:
            model = smf.mixedlm(formula, sub, groups=sub['patient_id'])
            result = model.fit(reml=True, method='lbfgs', maxiter=1000)
            beta = result.params.get('has_apnea[T.True]', result.params.get('has_apnea', None))
            pval = result.pvalues.get('has_apnea[T.True]', result.pvalues.get('has_apnea', None))
            if beta is None or pval is None or np.isnan(pval):
                return None
            ci = result.conf_int().loc['has_apnea[T.True]' if 'has_apnea[T.True]' in result.params else 'has_apnea']
            return {
                'channel': channel,
                'feature': feature,
                'beta': beta,
                'p_value': pval,
                'ci_low': ci[0],
                'ci_high': ci[1],
                'n_obs': len(sub)
            }
        except Exception:
            return None

    # --------------------------------------------------------------
    # НОВЫЙ РЕЖИМ: Тонические признаки vs AHI
    # --------------------------------------------------------------
    def _run_tonic_vs_ahi_thread(self, study_ids):
        try:
            # Принудительно загружаем тонические эпохи (data_type=1)
            df = self._load_epochs(study_ids, force_reload=True, data_type_override=1)
            if df is None or self.stop_flag:
                self.log("Нет тонических эпох для анализа.")
                return
            df = df.copy()
            # Убедимся, что нет эпох с событиями (по определению data_type=1 их нет)
            # Добавляем AHI из таблицы sleep_statistics (через filtered_df)
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is None:
                self.log("Нет отфильтрованных данных для извлечения AHI.")
                return
            # Собираем AHI по study_id
            ahi_df = filtered_df[['study_id', 'ahi']].drop_duplicates(subset=['study_id'])
            if 'ahi' not in ahi_df.columns:
                self.log("В отфильтрованных данных отсутствует колонка AHI.")
                return
            df = df.merge(ahi_df, on='study_id', how='left')
            if df['ahi'].isna().all():
                self.log("AHI отсутствует для всех исследований.")
                return
            # Подготовка данных
            df['patient_id'] = df['patient_id'].astype(int)
            if 'epoch_stage' in df.columns:
                df['epoch_stage'] = df['epoch_stage'].astype('category')
            if 'gender' in df.columns and 'gender_code' not in df.columns:
                df['gender_code'] = (df['gender'] == 'M').astype(int)

            # Добавляем ковариаты (возраст, пол, ИМТ)
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

            # Список задач (канал, признак, колонка)
            tasks = []
            for ch in self.all_channels:
                for feat in self.all_features:
                    col = f"{ch}_{feat}"
                    if col in df.columns:
                        tasks.append((ch, feat, col))
            for (a,b) in self.coh_pairs:
                pair = f"{a}{b}"
                for band in self.coh_bands:
                    col = f"{pair}_coh_{band}"
                    if col in df.columns:
                        tasks.append((f"{a}-{b}", band, col))

            if not tasks:
                self.log("Нет доступных признаков.")
                return
            self.log(f"Всего моделей (тонические vs AHI): {len(tasks)}. Начинаем расчёт...")
            results = []
            total = len(tasks)
            for i, (ch, feat, col) in enumerate(tasks):
                if self.stop_flag:
                    break
                self.main_app.set_progress(int(i / total * 100))
                res = self._fit_tonic_vs_ahi_model(df, col, ch, feat)
                if res:
                    results.append(res)
                if (i+1) % 30 == 0:
                    self.log(f"Обработано {i+1} из {total}")
            if not results:
                self.log("Нет результатов (модели не сошлись).")
                return
            res_df = pd.DataFrame(results)
            if res_df.empty:
                self.log("Нет результатов (модели не сошлись).")
                return

            # Удаляем строки с некорректными p-значениями
            res_df = res_df.dropna(subset=['p_value'])
            res_df = res_df[(res_df['p_value'] >= 0) & (res_df['p_value'] <= 1)]

            if res_df.empty:
                self.log("Нет корректных p-значений для FDR-коррекции.")
                return

            # Переименуем колонку beta в beta_ahi
            res_df.rename(columns={'beta': 'beta_ahi'}, inplace=True)

            pvals = res_df['p_value'].values
            res_df['q_value'] = false_discovery_control(pvals, method='bh')
            res_df['significant'] = res_df['q_value'] < self.fdr_threshold.get()
            self.results_df = res_df
            self.main_app.root.after(0, self._display_results_table, res_df)
            self.main_app.root.after(0, self._plot_lmm_results, res_df, is_ahi_mode=True)
            self.save_csv_btn.config(state=tk.NORMAL)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.diag_btn.config(state=tk.NORMAL)
            self.bootstrap_btn.config(state=tk.NORMAL)
            self.fast_diag_btn.config(state=tk.NORMAL)
            channels = sorted(res_df['channel'].unique())
            features = sorted(res_df['feature'].unique())
            if channels:
                self.diag_channel.set(channels[0])
            if features:
                self.diag_feature.set(features[0])
            n_sign = res_df['significant'].sum()
            self.log(f"Анализ тонических vs AHI завершён. Значимых признаков: {n_sign} (q<{self.fdr_threshold.get()})")
            if n_sign > 0:
                pos = len(res_df[(res_df['significant']) & (res_df['beta_ahi'] > 0)])
                neg = n_sign - pos
                messagebox.showinfo("Результаты", f"Значимых признаков (связь с AHI): {n_sign}\nИз них beta_ahi>0: {pos}, beta_ahi<0: {neg}")
        except Exception as e:
            self.log(f"Ошибка в режиме tonic_vs_ahi: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.main_app.set_progress(0)

    def _fit_tonic_vs_ahi_model(self, df, col, channel, feature):
        """Модель: признак ~ AHI + стадия + возраст + пол + ИМТ + (1|patient_id)"""
        cols = [col, 'ahi', 'patient_id']
        if self.include_stage.get() and 'epoch_stage' in df.columns:
            cols.append('epoch_stage')
        if self.include_covariates.get():
            for c in ['age_at_study', 'gender_code', 'bmi']:
                if c in df.columns:
                    cols.append(c)
        sub = df[cols].dropna()
        # Минимальное количество наблюдений
        if len(sub) < 30:
            return None
        # Удаляем выбросы по AHI (опционально, оставляем как есть)
        formula = f"{col} ~ ahi"
        if self.include_stage.get() and 'epoch_stage' in sub.columns:
            formula += " + epoch_stage"
        if self.include_covariates.get():
            for c in ['age_at_study', 'gender_code', 'bmi']:
                if c in sub.columns:
                    formula += f" + {c}"
        try:
            model = smf.mixedlm(formula, sub, groups=sub['patient_id'])
            result = model.fit(reml=True, method='lbfgs', maxiter=1000)
            beta = result.params.get('ahi', None)
            pval = result.pvalues.get('ahi', None)
            if beta is None or pval is None or np.isnan(pval):
                return None
            ci = result.conf_int().loc['ahi']
            return {
                'channel': channel,
                'feature': feature,
                'beta': beta,
                'p_value': pval,
                'ci_low': ci[0],
                'ci_high': ci[1],
                'n_obs': len(sub)
            }
        except Exception as e:
            # self.log(f"Ошибка для {col}: {e}")
            return None

    # --------------------------------------------------------------
    # Отображение таблицы и графиков (адаптировано под режимы)
    # --------------------------------------------------------------
    def _display_results_table(self, df):
        for row in self.tree.get_children():
            self.tree.delete(row)
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

    def _plot_lmm_results(self, results_df, is_ahi_mode=False):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        if results_df.empty:
            return

        beta_col = 'beta_ahi' if is_ahi_mode else 'beta'
        xlabel = 'Beta (коэффициент для AHI)' if is_ahi_mode else 'Beta (эффект апноэ)'
        title_prefix = 'Тонические признаки vs AHI' if is_ahi_mode else 'Сравнение эпох'

        fig = Figure(figsize=(12, 8), dpi=100)

        # Forest plot (топ-10/20 значимых)
        ax1 = fig.add_subplot(2, 1, 1)
        sign = results_df[results_df['significant']].copy()
        if sign.empty:
            ax1.text(0.5, 0.5, f"Нет значимых результатов (q < {self.fdr_threshold.get()})",
                     transform=ax1.transAxes, ha='center')
        else:
            top10 = sign.nsmallest(10, 'q_value')
            y_pos = np.arange(len(top10))
            labels = top10['feature'] + ' (' + top10['channel'] + ')'
            betas = top10[beta_col].values
            ci_low = top10['ci_low'].values
            ci_high = top10['ci_high'].values
            ax1.errorbar(betas, y_pos,
                         xerr=[betas - ci_low, ci_high - betas],
                         fmt='o', capsize=5, color='blue', ecolor='gray')
            ax1.axvline(x=0, linestyle='--', color='gray')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.set_xlabel(xlabel)
            ax1.set_title(f'{title_prefix} – топ-10 наиболее значимых признаков (FDR < {self.fdr_threshold.get()})')
            ax1.grid(True, alpha=0.3)

        # Volcano plot
        ax2 = fig.add_subplot(2, 1, 2)
        colors = np.where(results_df['significant'], 'red', 'gray')
        ax2.scatter(results_df[beta_col], -np.log10(results_df['p_value']), c=colors, alpha=0.6, s=20)
        if not sign.empty:
            top10_comb = sign.nsmallest(10, 'q_value')
            for _, row in top10_comb.iterrows():
                label = f"{row['feature']} ({row['channel']})"
                ax2.annotate(label, (row[beta_col], -np.log10(row['p_value'])),
                             textcoords="offset points", xytext=(5,5), ha='left', fontsize=7,
                             alpha=0.7, bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3))
        ax2.axhline(y=-np.log10(0.05), linestyle='--', color='blue', label='p=0.05 (номинальный)')
        ax2.axhline(y=-np.log10(self.fdr_threshold.get()), linestyle='--', color='green', label='FDR threshold')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('-log10(p-value)')
        ax2.set_title(f'{title_prefix} – Volcano plot (красные – значимые)')
        ax2.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig

    # --------------------------------------------------------------
    # Диагностика остатков (адаптирована под оба режима)
    # --------------------------------------------------------------
    def run_diagnostics(self):
        if self.results_df is None:
            messagebox.showwarning("Нет модели", "Сначала выполните LMM анализ.")
            return
        channel = self.diag_channel.get().strip()
        feature = self.diag_feature.get().strip()
        if not channel or not feature:
            messagebox.showwarning("Выбор признака", "Выберите канал и признак.")
            return
        row = self.results_df[(self.results_df['channel'] == channel) & (self.results_df['feature'] == feature)]
        if row.empty:
            messagebox.showerror("Ошибка", f"Признак {channel} {feature} не найден в результатах.")
            return
        self.log(f"Запуск диагностики для {channel} {feature}...")
        threading.Thread(target=self._diagnostics_thread, args=(channel, feature), daemon=True).start()

    def _diagnostics_thread(self, channel, feature):
        # Определяем режим по наличию колонки 'beta_ahi' в results_df (или по флагу)
        is_ahi_mode = 'beta_ahi' in self.results_df.columns if self.results_df is not None else False
        if self.epochs_df is None:
            self.log("Нет загруженных эпох.")
            return
        col = f"{channel}_{feature}"
        if col not in self.epochs_df.columns:
            self.log(f"Столбец {col} отсутствует.")
            return
        df = self.epochs_df.copy()
        if not is_ahi_mode:
            # Режим сравнения эпох: используем has_apnea
            df['has_apnea'] = df['has_apnea'].astype(bool)
        else:
            # Режим tonic_vs_ahi: нужен AHI
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is None:
                self.log("Нет данных для извлечения AHI.")
                return
            ahi_df = filtered_df[['study_id', 'ahi']].drop_duplicates(subset=['study_id'])
            if 'ahi' not in ahi_df.columns:
                self.log("AHI отсутствует.")
                return
            df = df.merge(ahi_df, on='study_id', how='left')
            if df['ahi'].isna().all():
                self.log("Нет AHI для исследований.")
                return
            # Для tonic_vs_ahi оставляем только тонические эпохи (data_type=1)
            if 'data_type' in df.columns:
                df = df[df['data_type'] == 1].copy()
        df['patient_id'] = df['patient_id'].astype(int)
        if 'epoch_stage' in df.columns:
            df['epoch_stage'] = df['epoch_stage'].astype('category')
        if 'gender' in df.columns and 'gender_code' not in df.columns:
            df['gender_code'] = (df['gender'] == 'M').astype(int)

        if self.include_covariates.get():
            filtered_df2 = self.main_app.get_filtered_data()
            if filtered_df2 is not None:
                cov = filtered_df2[['study_id', 'age_at_study', 'gender', 'bmi']].drop_duplicates(subset=['study_id'])
                cov['gender_code'] = (cov['gender'] == 'M').astype(int)
                df = df.merge(cov[['study_id', 'age_at_study', 'gender_code', 'bmi']], on='study_id', how='left')

        # Формула в зависимости от режима
        if not is_ahi_mode:
            formula = f"{col} ~ has_apnea"
            if self.include_stage.get() and 'epoch_stage' in df.columns:
                formula += " + epoch_stage"
            if self.include_covariates.get():
                for c in ['age_at_study', 'gender_code', 'bmi']:
                    if c in df.columns:
                        formula += f" + {c}"
            sub = df[[col, 'has_apnea', 'patient_id'] + [c for c in ['epoch_stage', 'age_at_study', 'gender_code', 'bmi'] if c in df.columns]].dropna()
        else:
            formula = f"{col} ~ ahi"
            if self.include_stage.get() and 'epoch_stage' in df.columns:
                formula += " + epoch_stage"
            if self.include_covariates.get():
                for c in ['age_at_study', 'gender_code', 'bmi']:
                    if c in df.columns:
                        formula += f" + {c}"
            sub = df[[col, 'ahi', 'patient_id'] + [c for c in ['epoch_stage', 'age_at_study', 'gender_code', 'bmi'] if c in df.columns]].dropna()

        if len(sub) < 30:
            self.log("Недостаточно наблюдений для диагностики.")
            return
        try:
            model = smf.mixedlm(formula, sub, groups=sub['patient_id'])
            result = model.fit(reml=False, method='lbfgs', maxiter=1000)
            if hasattr(result, 'cov_re') and np.linalg.matrix_rank(result.cov_re) < result.cov_re.shape[0]:
                self.log("Модель имеет сингулярную ковариацию случайных эффектов – диагностика пропущена.")
                return
            fitted = result.fittedvalues
            resid = result.resid
            shapiro_p = None
            if len(resid) <= 5000:
                _, shapiro_p = shapiro(resid)
            resid2 = resid ** 2
            import statsmodels.api as sm
            X = sm.add_constant(fitted)
            bp_model = sm.OLS(resid2, X).fit()
            bp_stat = bp_model.rsquared * len(resid)
            bp_p = 1 - chi2.cdf(bp_stat, df=1)
            self.last_diagnostics = {
                'fitted': fitted,
                'residuals': resid,
                'shapiro_p': shapiro_p,
                'bp_p': bp_p,
                'n': len(resid),
                'channel': channel,
                'feature': feature,
                'col': col
            }
            fig = Figure(figsize=(10,8))
            ax1 = fig.add_subplot(2,2,1)
            ax1.scatter(fitted, resid, alpha=0.5)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Предсказанные значения')
            ax1.set_ylabel('Остатки')
            ax1.set_title('Остатки vs Предсказанные')
            ax2 = fig.add_subplot(2,2,2)
            stats.probplot(resid, dist="norm", plot=ax2)
            ax2.set_title('Q-Q plot остатков')
            ax3 = fig.add_subplot(2,2,3)
            ax3.hist(resid, bins=30, edgecolor='black')
            ax3.set_xlabel('Остатки')
            ax3.set_ylabel('Частота')
            ax3.set_title('Гистограмма остатков')
            ax4 = fig.add_subplot(2,2,4)
            ax4.text(0.1, 0.9, f"n = {len(resid)}", fontsize=10)
            ax4.text(0.1, 0.8, f"Shapiro-Wilk p = {shapiro_p:.4f}" if shapiro_p else "Shapiro-Wilk: n>5000", fontsize=10)
            ax4.text(0.1, 0.7, f"Breusch-Pagan p = {bp_p:.4f}", fontsize=10)
            if shapiro_p and shapiro_p < 0.05:
                ax4.text(0.1, 0.6, "[!] Остатки не нормальны", color='red', fontsize=10)
            else:
                ax4.text(0.1, 0.6, "[OK] Нормальность не отвергается", color='green', fontsize=10)
            if bp_p < 0.05:
                ax4.text(0.1, 0.5, "[!] Гетероскедастичность", color='red', fontsize=10)
            else:
                ax4.text(0.1, 0.5, "[OK] Гомоскедастичность", color='green', fontsize=10)
            ax4.axis('off')
            fig.tight_layout()
            self.main_app.root.after(0, self._show_diagnostic_plot, fig)
            self.diag_figure = fig
            self.diag_model = result
            self.diag_feature_name = f"{channel}_{feature}"
            self.save_diag_btn.config(state=tk.NORMAL)
            self.log(f"Диагностика завершена. Shapiro p={shapiro_p}, BP p={bp_p:.4f}")
            self.all_diagnostics.append({
                'channel': channel,
                'feature': feature,
                'shapiro_p': shapiro_p,
                'bp_p': bp_p,
                'n': len(resid)
            })
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

    def save_diagnostics_csv(self):
        if not hasattr(self, 'last_diagnostics'):
            messagebox.showwarning("Нет данных", "Сначала выполните диагностику для выбранного признака.")
            return
        diag = self.last_diagnostics
        df = pd.DataFrame({
            'fitted': diag['fitted'],
            'residuals': diag['residuals']
        })
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            df.to_csv(path, index=False)
            self.log(f"Диагностика сохранена в {path}")

    def fast_diagnostics(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Сначала выполните LMM анализ.")
            return
        sign = self.results_df[self.results_df['significant']].copy()
        if len(sign) == 0:
            messagebox.showinfo("Нет значимых", "Нет значимых признаков для диагностики.")
            return
        top10 = sign.nsmallest(10, 'q_value')
        self.log(f"Быстрая диагностика для {len(top10)} наиболее значимых признаков...")
        def run_sequential():
            for idx, (_, row) in enumerate(top10.iterrows()):
                if self.stop_flag:
                    break
                self.log(f"[{idx+1}/{len(top10)}] Диагностика: {row['channel']} {row['feature']}")
                self._diagnostics_thread(row['channel'], row['feature'])
                import time
                time.sleep(2)
        threading.Thread(target=run_sequential, daemon=True).start()

    # --------------------------------------------------------------
    # Бутстрап (адаптирован под режимы)
    # --------------------------------------------------------------
    def run_bootstrap(self):
        if self.diag_model is None:
            messagebox.showwarning("Нет модели", "Сначала выполните диагностику для выбранного признака.")
            return
        messagebox.showinfo("Бутстрап", "Будет выполнено 1000 итераций (блок-бутстрап по пациентам).\nЭто может занять 1-2 минуты.")
        threading.Thread(target=self._bootstrap_thread, daemon=True).start()

    def _bootstrap_thread(self):
        is_ahi_mode = 'beta_ahi' in self.results_df.columns if self.results_df is not None else False
        channel, feature = self.diag_feature_name.split('_', 1)
        col = f"{channel}_{feature}"
        df = self.epochs_df.copy()
        if not is_ahi_mode:
            df['has_apnea'] = df['has_apnea'].astype(bool)
        else:
            # Режим AHI: добавляем AHI и оставляем тонические эпохи
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is None:
                self.log("Нет данных для извлечения AHI.")
                return
            ahi_df = filtered_df[['study_id', 'ahi']].drop_duplicates(subset=['study_id'])
            df = df.merge(ahi_df, on='study_id', how='left')
            if 'data_type' in df.columns:
                df = df[df['data_type'] == 1].copy()
        df['patient_id'] = df['patient_id'].astype(int)
        if 'epoch_stage' in df.columns:
            df['epoch_stage'] = df['epoch_stage'].astype('category')
        if self.include_covariates.get():
            filtered_df2 = self.main_app.get_filtered_data()
            if filtered_df2 is not None:
                cov = filtered_df2[['study_id', 'age_at_study', 'gender', 'bmi']].drop_duplicates(subset=['study_id'])
                cov['gender_code'] = (cov['gender'] == 'M').astype(int)
                df = df.merge(cov[['study_id', 'age_at_study', 'gender_code', 'bmi']], on='study_id', how='left')
        # Формула
        if not is_ahi_mode:
            formula = f"{col} ~ has_apnea"
            if self.include_stage.get() and 'epoch_stage' in df.columns:
                formula += " + epoch_stage"
            if self.include_covariates.get():
                for c in ['age_at_study', 'gender_code', 'bmi']:
                    if c in df.columns:
                        formula += f" + {c}"
            use_cols = [col, 'has_apnea', 'patient_id']
        else:
            formula = f"{col} ~ ahi"
            if self.include_stage.get() and 'epoch_stage' in df.columns:
                formula += " + epoch_stage"
            if self.include_covariates.get():
                for c in ['age_at_study', 'gender_code', 'bmi']:
                    if c in df.columns:
                        formula += f" + {c}"
            use_cols = [col, 'ahi', 'patient_id']
        # Добавляем остальные колонки
        if self.include_stage.get() and 'epoch_stage' in df.columns:
            use_cols.append('epoch_stage')
        if self.include_covariates.get():
            for c in ['age_at_study', 'gender_code', 'bmi']:
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
                if not is_ahi_mode:
                    beta = result_boot.params.get('has_apnea[T.True]', result_boot.params.get('has_apnea', None))
                else:
                    beta = result_boot.params.get('ahi', None)
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
                'feature': f"{channel}_{feature}",
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_bootstrap': p_bootstrap,
                'n_iter': len(betas)
            }
            self.log(f"Бутстрап сохранён для {channel}_{feature}")
            self.bootstrap_results = {'betas': betas, 'ci_low': ci_low, 'ci_high': ci_high, 'p': p_bootstrap}
            self.log(f"Бутстрап CI: [{ci_low:.4f}, {ci_high:.4f}], p≈{p_bootstrap:.4f}")
            fig = Figure(figsize=(8,5))
            ax = fig.add_subplot(111)
            ax.hist(betas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=ci_low, color='red', linestyle='--', label=f'2.5%: {ci_low:.2f}')
            ax.axvline(x=ci_high, color='red', linestyle='--', label=f'97.5%: {ci_high:.2f}')
            ax.axvline(x=np.mean(betas), color='green', linestyle='-', label=f'Mean: {np.mean(betas):.2f}')
            ax.set_xlabel('Beta coefficient')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Bootstrap distribution of {"AHI coefficient" if is_ahi_mode else "beta (apnea effect)"} for {channel} {feature}')
            ax.legend()
            self.main_app.root.after(0, self._show_diagnostic_plot, fig)
            self.save_bootstrap_btn.config(state=tk.NORMAL)
            self.right_notebook.select(self.tab_diag)
        else:
            self.log("Бутстрап не дал результатов.")

    def save_bootstrap_csv(self):
        if not self.bootstrap_results:
            messagebox.showwarning("Нет данных", "Сначала выполните бутстрап.")
            return
        df = pd.DataFrame({'beta': self.bootstrap_results['betas']})
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            df.to_csv(path, index=False)
            self.log(f"Бутстрап-распределение сохранено в {path}")

    # --------------------------------------------------------------
    # Генерация отчёта (добавлена информация о режиме)
    # --------------------------------------------------------------
    def generate_report(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Сначала выполните LMM анализ.")
            return

        # Определяем режим
        is_ahi_mode = 'beta_ahi' in self.results_df.columns
        beta_col = 'beta_ahi' if is_ahi_mode else 'beta'
        mode_name = "Тонические признаки vs AHI (хронический эффект)" if is_ahi_mode else "Сравнение эпох (апноэ vs без апноэ)"

        # Получаем основные данные
        filtered_df = self.main_app.get_filtered_data()
        n_patients = filtered_df['patient_id'].nunique() if filtered_df is not None else 0
        n_epochs = len(self.epochs_df) if self.epochs_df is not None else 0
        sign = self.results_df[self.results_df['significant']].copy()
        top10 = sign.nsmallest(10, 'q_value')

        # Информация о наборе эпох
        if not is_ahi_mode:
            data_type_label = {1: "Тонический (чистые эпохи N2/N3)", 2: "Все эпохи N2/N3",
                               3: "Фильтр по положению (первые 10 с)"}
            data_type_text = data_type_label.get(self.data_type.get(), "Неизвестно")
        else:
            data_type_text = "Тонические эпохи (data_type=1) – только фоновые эпохи без событий"

        # VIF (если есть)
        vif_html = ""
        if hasattr(self, 'last_vif') and self.last_vif:
            high_vif = {k: v for k, v in self.last_vif.items() if v > 5}
            if high_vif:
                vif_html = "<h2>Проверка мультиколлинеарности (VIF)</h2>"
                vif_html += "<p>Обнаружены признаки с Variance Inflation Factor > 5, что указывает на мультиколлинеарность.</p>"
                vif_html += "<table border='1'><tr><th>Признак</th><th>VIF</th></tr>"
                for var, vif_val in high_vif.items():
                    vif_html += f"<tr><td>{var}</td><td>{vif_val:.2f}</td></tr>"
                vif_html += "</table><p>Следует интерпретировать коэффициенты с осторожностью.</p>"

        # График LMM (если есть)
        plot_html = ""
        if self.current_figure is not None:
            buf = io.BytesIO()
            self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'

        # ----- Раздел проверки гипотез (адаптирован под режим) -----
        hypotheses_html = "<h2>Проверка гипотез исследования</h2>"
        if not is_ahi_mode:
            # Режим сравнения эпох (острый эффект)
            delta_sign = sign[sign['feature'].isin(['abs_delta', 'rel_delta'])]
            if not delta_sign.empty:
                h1_status = "✅ ПОДТВЕРЖДЕНА (острая реакция)"
                h1_detail = f"Дельта-мощность значимо различается между эпохами с апноэ и без в {delta_sign['channel'].nunique()} каналах."
            else:
                h1_status = "❌ НЕ ПОДТВЕРЖДЕНА"
                h1_detail = "Нет значимых различий дельта-мощности."
            hypotheses_html += f"<h3>H1 (тоническая спектральная) – острая реакция</h3><p><strong>{h1_status}</strong> — {h1_detail}</p>"

            tbr_sign = sign[sign['feature'] == 'tbr']
            if not tbr_sign.empty:
                h2_status = "✅ ПОДТВЕРЖДЕНА"
                h2_detail = f"TBR значимо {'снижается' if all(tbr_sign[beta_col] < 0) else 'изменяется'} в {tbr_sign['channel'].nunique()} каналах."
            else:
                h2_status = "❌ НЕ ПОДТВЕРЖДЕНА"
                h2_detail = "TBR не показал значимых различий."
            hypotheses_html += f"<h3>H2 (тоническая спектральная)</h3><p><strong>{h2_status}</strong> — {h2_detail}</p>"
            hypotheses_html += "<h3>H3 (фазическая)</h3><p>❌ НЕ ПРОВЕРЯЕТСЯ В ДАННОМ АНАЛИЗЕ (требуется фазический анализ событий)</p>"

            sampen_sign = sign[sign['feature'] == 'sampen']
            if not sampen_sign.empty:
                h4_status = "✅ ПОДТВЕРЖДЕНА"
                h4_detail = f"Sample Entropy значимо изменяется в {sampen_sign['channel'].nunique()} каналах."
            else:
                h4_status = "❌ НЕ ПОДТВЕРЖДЕНА"
                h4_detail = "Нет значимых различий SampEn."
            hypotheses_html += f"<h3>H4 (нелинейная)</h3><p><strong>{h4_status}</strong> — {h4_detail}</p>"
        else:
            # Режим тонические vs AHI (хронический эффект)
            delta_sign = sign[sign['feature'].isin(['abs_delta', 'rel_delta'])]
            if not delta_sign.empty:
                pos_delta = delta_sign[delta_sign[beta_col] > 0]
                h1_status = "✅ ПОДТВЕРЖДЕНА"
                if len(pos_delta) > 0:
                    h1_detail = f"Дельта-мощность положительно связана с AHI в {len(pos_delta)} комбинациях (каналы: {', '.join(pos_delta['channel'].unique())}) — подтверждает H1."
                else:
                    h1_detail = "Дельта-мощность связана с AHI, но коэффициент отрицательный (уменьшается с ростом AHI) – противоречит H1."
            else:
                h1_status = "❌ НЕ ПОДТВЕРЖДЕНА"
                h1_detail = "Нет значимой связи дельта-мощности с AHI в тонических эпохах."
            hypotheses_html += f"<h3>H1 (хронический эффект)</h3><p><strong>{h1_status}</strong> — {h1_detail}</p>"
            hypotheses_html += "<p><em>Гипотеза H1:</em> относительная мощность дельта-ритма в тонических эпохах N2/N3 положительно коррелирует с тяжестью ОАС.</p>"

            tbr_sign = sign[sign['feature'] == 'tbr']
            if not tbr_sign.empty:
                neg_tbr = tbr_sign[tbr_sign[beta_col] < 0]
                if len(neg_tbr) > 0:
                    h2_status = "✅ ПОДТВЕРЖДЕНА"
                    h2_detail = f"TBR отрицательно связан с AHI в {len(neg_tbr)} каналах (снижается с тяжестью)."
                else:
                    h2_status = "⚠️ ЧАСТИЧНО"
                    h2_detail = "TBR связан с AHI, но положительно (противоречит H2)."
            else:
                h2_status = "❌ НЕ ПОДТВЕРЖДЕНА"
                h2_detail = "Нет значимой связи TBR с AHI."
            hypotheses_html += f"<h3>H2 (хронический эффект)</h3><p><strong>{h2_status}</strong> — {h2_detail}</p>"

            sampen_sign = sign[sign['feature'] == 'sampen']
            if not sampen_sign.empty:
                h4_status = "✅ ПОДТВЕРЖДЕНА (хронический эффект)"
                h4_detail = f"SampEn значимо связан с AHI в {sampen_sign['channel'].nunique()} каналах."
            else:
                h4_status = "❌ НЕ ПОДТВЕРЖДЕНА"
                h4_detail = "Нет значимой связи SampEn с AHI."
            hypotheses_html += f"<h3>H4 (нелинейная, хроническая)</h3><p><strong>{h4_status}</strong> — {h4_detail}</p>"
            hypotheses_html += "<p><em>Примечание:</em> Для проверки H3 (фазическая) требуется отдельный анализ событий (не входит в данный LMM).</p>"

        # ----- Диагностика остатков -----
        diag_html = ""
        if self.all_diagnostics:
            top_diag = self.all_diagnostics[:5]
            diag_html = "<h2>Диагностика остатков LMM</h2>"
            diag_html += "<p>Для наиболее значимых признаков выполнена проверка нормальности остатков (Шапиро-Уилк) и гомоскедастичности (Бройш-Паган).</p>"
            diag_html += "<table border='1'><tr><th>Канал</th><th>Признак</th><th>n</th><th>Shapiro-Wilk p</th><th>Breusch-Pagan p</th><th>Заключение</th></tr>"
            for d in top_diag:
                shapiro_str = f"{d['shapiro_p']:.4f}" if d['shapiro_p'] is not None else ">5000 (тест не прим.)"
                shapiro_ok = d['shapiro_p'] > 0.05 if d['shapiro_p'] is not None else None
                bp_ok = d['bp_p'] > 0.05
                conclusion = []
                if shapiro_ok is True:
                    conclusion.append("✅ нормальность")
                elif shapiro_ok is False:
                    conclusion.append("❗ остатки не нормальны")
                else:
                    conclusion.append("⚠️ выборка >5000, тест не применялся")
                conclusion.append("✅ гомоскедастичность" if bp_ok else "❗ гетероскедастичность")
                diag_html += f"<tr><td>{d['channel']}</td><td>{d['feature']}</td><td>{d['n']}</td><td>{shapiro_str}</td><td>{d['bp_p']:.4f}</td><td>{', '.join(conclusion)}</td></tr>"
            diag_html += "</table><p>При нарушениях предположений рекомендуется использовать бутстрап (см. ниже).</p>"

        # ----- Бутстрап -----
        bootstrap_html = ""
        if self.bootstrap_info:
            bi = self.bootstrap_info
            bootstrap_html = f"<h2>Бутстрап-проверка (блок-бутстрап по пациентам)</h2>"
            bootstrap_html += f"<p>Для признака <strong>{bi['feature']}</strong> выполнено {bi['n_iter']} успешных итераций.</p>"
            bootstrap_html += f"<p><strong>95% доверительный интервал для {'коэффициента AHI' if is_ahi_mode else 'beta (эффект апноэ)'}:</strong> [{bi['ci_low']:.4f}, {bi['ci_high']:.4f}]</p>"
            bootstrap_html += f"<p><strong>Бутстрап p-value (двусторонний):</strong> {bi['p_bootstrap']:.4f}</p>"
            bootstrap_html += "<p><em>Интерпретация:</em> если CI не пересекает 0, эффект статистически значим с учётом возможных нарушений допущений.</p>"

        # ----- Основная таблица значимых признаков -----
        if is_ahi_mode:
            interpretation_text = "Положительный коэффициент = признак увеличивается с ростом AHI (тяжести ОАС). Отрицательный = уменьшается."
        else:
            interpretation_text = "Положительный коэффициент = признак выше в эпохах с апноэ, отрицательный = ниже."

        # Формирование полного HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>LMM отчёт – {mode_name}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; line-height: 1.4; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: left; vertical-align: top; }}
            th {{ background-color: #ecf0f1; }}
            .sign {{ background-color: #f9e79f; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .footnote {{ font-size: 0.9em; color: #7f8c8d; margin-top: 20px; }}
        </style>
        </head>
        <body>
        <h1>Отчёт о линейных смешанных моделях (LMM)</h1>
        <p><strong>Дата генерации:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Режим анализа:</strong> {mode_name}</p>
        <p><strong>Пациентов (исследований):</strong> {n_patients}</p>
        <p><strong>Эпох (всего в загруженном наборе):</strong> {n_epochs}</p>
        <p><strong>Модель:</strong> признак ~ {'AHI' if is_ahi_mode else 'has_apnea'} {'+ стадия сна' if self.include_stage.get() else ''} {'+ возраст + пол + ИМТ' if self.include_covariates.get() else ''} + (1 | patient_id)</p>
        <p><strong>Тип набора эпох:</strong> {data_type_text}</p>
        <p><strong>FDR порог:</strong> q = {self.fdr_threshold.get()}</p>
        {vif_html}
        {hypotheses_html}
        {diag_html}
        {bootstrap_html}
        <h2>Значимые признаки (всего {len(sign)})</h2>
        <table>
            <thead>
                <tr><th>Канал</th><th>Признак</th><th>{'β<sub>AHI</sub>' if is_ahi_mode else 'β<sub>apnea</sub>'}</th><th>p-value</th><th>q-value (FDR)</th><th>95% CI</th><th>Интерпретация</th></tr>
            </thead>
            <tbody>
        """ + "".join(
            f"<tr class='sign'><td>{row['channel']}</td><td>{row['feature']}</td>"
            f"<td>{row[beta_col]:.4f}</td><td>{row['p_value']:.2e}</td><td>{row['q_value']:.4f}</td>"
            f"<td>[{row['ci_low']:.2f}, {row['ci_high']:.2f}]</td><td>{interpretation_text}</td></tr>"
            for _, row in sign.iterrows()
        ) + """
            </tbody>
        </table>
        <h2>Топ-10 наиболее значимых признаков</h2>
        <table>
            <thead><tr><th>Канал</th><th>Признак</th><th>Коэффициент</th><th>q-value</th><th>Направление</th></tr></thead>
            <tbody>
        """ + "".join(
            f"<tr><td>{row['channel']}</td><td>{row['feature']}</td><td>{row[beta_col]:.4f}</td><td>{row['q_value']:.4f}</td>"
            f"<td>{'Положительная связь с AHI' if row[beta_col] > 0 else 'Отрицательная связь с AHI' if is_ahi_mode else ('Выше при апноэ' if row[beta_col] > 0 else 'Ниже при апноэ')}</td></tr>"
            for _, row in top10.iterrows()
        ) + """
            </tbody>
        </table>
        <h2>Графики LMM</h2>
        """ + plot_html + """
        <div class="footnote">
            <p><strong>Примечание:</strong> При нарушениях нормальности остатков или гетероскедастичности доверительные интервалы и p‑значения следует уточнять с помощью бутстрапа (вкладка "Диагностика модели").</p>
            <p>Полный код и данные доступны в репозитории проекта.</p>
        </div>
        </body>
        </html>
        """

        # Сохраняем во временный файл и открываем в браузере
        fd, path = tempfile.mkstemp(suffix='.html', prefix='lmm_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт в браузере: {path}")

    def _generate_hypotheses_html(self, sign):
        # оригинальная проверка гипотез из старого кода
        hypotheses_html = "<h2>Проверка гипотез исследования</h2>"
        delta_sign = sign[sign['feature'].isin(['abs_delta', 'rel_delta'])]
        if not delta_sign.empty:
            h1_status = "✅ ПОДТВЕРЖДЕНА"
            h1_detail = f"Значимые изменения дельта-мощности в {delta_sign['channel'].nunique()} каналах."
        else:
            h1_status = "❌ НЕ ПОДТВЕРЖДЕНА"
            h1_detail = "Нет значимых различий дельта-мощности."
        hypotheses_html += f"<h3>H1 (тоническая спектральная)</h3><p><strong>{h1_status}</strong> — {h1_detail}</p>"
        tbr_sign = sign[sign['feature'] == 'tbr']
        if not tbr_sign.empty:
            h2_status = "✅ ПОДТВЕРЖДЕНА"
            h2_detail = f"TBR значимо снижается в {tbr_sign['channel'].nunique()} каналах."
        else:
            h2_status = "❌ НЕ ПОДТВЕРЖДЕНА"
            h2_detail = "TBR не показал значимых различий."
        hypotheses_html += f"<h3>H2 (тоническая спектральная)</h3><p><strong>{h2_status}</strong> — {h2_detail}</p>"
        hypotheses_html += "<h3>H3 (фазическая)</h3><p>❌ НЕ ПРОВЕРЯЕТСЯ В ТОНИЧЕСКОМ LMM</p>"
        sampen_sign = sign[sign['feature'] == 'sampen']
        if not sampen_sign.empty:
            h4_status = "✅ ПОДТВЕРЖДЕНА"
            h4_detail = f"Sample Entropy значимо изменяется в {sampen_sign['channel'].nunique()} каналах."
        else:
            h4_status = "❌ НЕ ПОДТВЕРЖДЕНА"
            h4_detail = "Нет значимых различий SampEn."
        hypotheses_html += f"<h3>H4 (нелинейная)</h3><p><strong>{h4_status}</strong> — {h4_detail}</p>"
        return hypotheses_html

    # --------------------------------------------------------------
    # Сохранение графиков и CSV
    # --------------------------------------------------------------
    def save_norm_plot(self):
        if self.norm_figure is None:
            messagebox.showwarning("Нет графика", "Сначала выполните проверку нормальности.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if path:
            self.norm_figure.savefig(path, dpi=150, bbox_inches='tight')
            self.log(f"График нормальности сохранён в {path}")

    def save_diag_plot(self):
        if self.diag_figure is None:
            messagebox.showwarning("Нет графика", "Сначала выполните диагностику модели.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if path:
            self.diag_figure.savefig(path, dpi=150, bbox_inches='tight')
            self.log(f"График диагностики сохранён в {path}")

    def save_results_csv(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Нет результатов для сохранения.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.results_df.to_csv(path, index=False)
            self.log(f"Сохранено в {path}")

    def save_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "График не построен.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if path:
            self.current_figure.savefig(path, dpi=150, bbox_inches='tight')
            self.log(f"График сохранён в {path}")
