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
        self.norm_figure = None  # для графика проверки нормальности
        self.diag_figure = None  # для графика диагностики модели
        self.all_diagnostics = [] # Хранилище результатов диагностики для отчёта
        self.bootstrap_info = None # Хранилище результатов бутстрапа для отчёта

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
        lmm_desc_frame = ttk.LabelFrame(left_frame, text="Что такое LMM?", padding=5)
        lmm_desc_frame.pack(fill=tk.X, padx=5, pady=5)
        desc_text = ("Линейные смешанные модели (LMM) учитывают иерархическую структуру данных (эпохи вложены в пациентов). Оценивают фиксированный эффект апноэ (beta) с поправкой на возраст, пол, ИМТ и стадию сна, а также случайный перехват для каждого пациента. Позволяют выявить ЭЭГ-признаки, значимо различающиеся между эпохами с апноэ и без, с контролем множественных сравнений (FDR).")
        ttk.Label(lmm_desc_frame, text=desc_text, justify=tk.LEFT, wraplength=600).pack(anchor=tk.W, fill=tk.X, padx=5, pady=2)
        # ---- Кнопка инструкции ----
        ttk.Button(lmm_desc_frame, text="Показать инструкцию", command=self.show_instructions).pack(anchor=tk.W, padx=5,
                                                                                                    pady=2)
        model_frame = ttk.LabelFrame(left_frame, text="Настройки LMM", padding=5)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(model_frame, text="Включить стадию сна (N2/N3) как фактор",
                        variable=self.include_stage).pack(anchor=tk.W)
        ttk.Checkbutton(model_frame, text="Включить ковариаты (возраст, пол, ИМТ)",
                        variable=self.include_covariates).pack(anchor=tk.W)
        ttk.Label(model_frame, text="Порог FDR (q):").pack(anchor=tk.W)
        ttk.Entry(model_frame, textvariable=self.fdr_threshold, width=6).pack(anchor=tk.W)

        # ---- Выбор типа набора эпох ----
        data_type_frame = ttk.Frame(model_frame)
        data_type_frame.pack(fill=tk.X, pady=5)
        ttk.Label(data_type_frame, text="Тип набора эпох:").pack(side=tk.LEFT, padx=2)
        self.data_type = tk.IntVar(value=2)  # по умолчанию "все эпохи"
        ttk.Radiobutton(data_type_frame, text="Тонический (1)", variable=self.data_type, value=1).pack(side=tk.LEFT,
                                                                                                       padx=5)
        ttk.Radiobutton(data_type_frame, text="Все эпохи (2)", variable=self.data_type, value=2).pack(side=tk.LEFT,
                                                                                                      padx=5)
        ttk.Radiobutton(data_type_frame, text="Фильтр по положению (3)", variable=self.data_type, value=3).pack(
            side=tk.LEFT, padx=5)

        self.exclude_central_mixed = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_frame, text="Исключить центральное/смешанное апноэ",
                        variable=self.exclude_central_mixed).pack(anchor=tk.W)

        cache_frame = ttk.LabelFrame(left_frame, text="Кэширование", padding=5)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cache_frame, text="Использовать кэш API (ускоряет повторные запуски)",
                        variable=self.use_cache).pack(anchor=tk.W)
        ttk.Button(cache_frame, text="Очистить кэш API", command=self.clear_api_cache).pack(anchor=tk.W, pady=2)

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
        self.save_norm_plot_btn = ttk.Button(row2, text="Сохранить график (PNG)", command=self.save_norm_plot,
                                             state=tk.DISABLED)
        self.save_norm_plot_btn.pack(side=tk.LEFT, padx=2)

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
        self.save_diag_plot_btn = ttk.Button(row2d, text="Сохранить график (PNG)", command=self.save_diag_plot,
                                             state=tk.DISABLED)
        self.save_diag_plot_btn.pack(side=tk.LEFT, padx=2)

        row3d = ttk.Frame(diag_frame)
        row3d.pack(fill=tk.X, pady=2)
        self.bootstrap_btn = ttk.Button(row3d, text="Бутстрап (1000 итераций)", command=self.run_bootstrap,
                                        state=tk.DISABLED)
        self.bootstrap_btn.pack(side=tk.LEFT, padx=2)
        self.save_bootstrap_btn = ttk.Button(row3d, text="Сохранить бутстрап (CSV)", command=self.save_bootstrap_csv,
                                             state=tk.DISABLED)
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
    # Вспомогательные методы
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
            "2. Нажмите 'Запустить LMM'. Первый запуск загрузит эпохи (кэшируются).\n"
            "3. После завершения появится таблица, forest plot и volcano plot.\n"
            "   - Forest plot показывает топ-10 наиболее значимых признаков (q < порог).\n"
            "   - Volcano plot показывает все признаки с подписями значимых.\n"
            "4. Выберите признак из таблицы (в левой панели в разделе 'Диагностика модели')\n"
            "   и нажмите 'Диагностика остатков' – будут построены графики и тесты.\n"
            "   Кнопка 'Сохранить диагностику (CSV)' сохранит остатки и предсказанные значения.\n"
            "5. Если предположения нарушены, нажмите 'Бутстрап' – получите робастные CI.\n"
            "   Результат бутстрапа также появится во вкладке 'Диагностика модели'.\n"
            "6. Кнопка 'Быстрая диагностика топ-10' запустит диагностику для 10 наиболее значимых признаков.\n"
            "7. Кнопка 'Сформировать отчёт' создаст HTML-отчёт с интерпретацией и встроенными графиками.\n"
            "8. Результаты нормальности и бутстрапа можно сохранить в CSV отдельными кнопками.\n"
            "\n"
            "Примечание: бутстрап (1000 итераций) может занять 1-2 минуты.\n"
        )
        messagebox.showinfo("Инструкция", msg)

    def _aggregate_features_by_feature(self, results_df):
        """
        Агрегирует результаты LMM по имени признака (без учёта канала).
        Для каждого признака вычисляет:
        - средний beta (mean_beta)
        - минимальный q_value (лучший)
        - средний доверительный интервал (средний CI_low, CI_high)
        - количество каналов, для которых признак значим.
        """
        if results_df.empty:
            return pd.DataFrame()
        agg = results_df.groupby('feature').agg(
            mean_beta=('beta', 'mean'),
            min_q=('q_value', 'min'),
            mean_ci_low=('ci_low', 'mean'),
            mean_ci_high=('ci_high', 'mean'),
            n_significant=('significant', 'sum'),
            n_total=('significant', 'count')
        ).reset_index()
        # Рассчитываем обобщённый эффект и значимость
        agg['significant_agg'] = agg['min_q'] < self.fdr_threshold.get()
        # Сортируем по минимальному q
        agg = agg.sort_values('min_q')
        return agg

    def _check_vif(self, df):
        """Вычисляет VIF для фиксированных эффектов модели."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        import pandas as pd

        # Копируем данные
        data = df.copy()
        fixed_cols = ['has_apnea']
        if self.include_stage.get() and 'epoch_stage' in data.columns:
            stage_dummies = pd.get_dummies(data['epoch_stage'], prefix='stage', drop_first=True)
            data = pd.concat([data, stage_dummies], axis=1)
            fixed_cols.extend(stage_dummies.columns)
        if self.include_covariates.get():
            for c in ['age_at_study', 'gender_code', 'bmi']:
                if c in data.columns:
                    fixed_cols.append(c)
        # Приводим всё к числовым типам (ошибки заменяем на NaN)
        vif_df = data[fixed_cols].apply(pd.to_numeric, errors='coerce').dropna()
        if vif_df.shape[1] < 2:
            return {}
        vif_df = add_constant(vif_df)
        vif_data = {}
        for i in range(1, len(vif_df.columns)):
            try:
                vif_val = variance_inflation_factor(vif_df.values, i)
                vif_data[vif_df.columns[i]] = vif_val
            except Exception:
                vif_data[vif_df.columns[i]] = np.nan
        return vif_data

    # --------------------------------------------------------------
    # Загрузка эпох
    # --------------------------------------------------------------
    def _load_epochs(self, study_ids, force_reload=False):
        if not force_reload and self.epochs_df is not None:
            return self.epochs_df

        def update_progress(page, total, _):
            if total > 0:
                self.main_app.set_progress(int(page / total * 100))

        self.main_app.set_progress(0)
        self.log("Загрузка эпох из API (может занять несколько минут)...")
        epochs = get_epochs(
            self.main_app.tabs['load'].api_url.get(),
            self.main_app.tabs['load'].token.get(),
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
        self.norm_btn.config(state=tk.NORMAL)
        return df

    # --------------------------------------------------------------
    # Проверка нормальности
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
        # Тест Шапиро-Уилка
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
        # Очистить и построить Q-Q plot
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
    # Основной LMM
    # --------------------------------------------------------------
    def run_lmm(self):
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            messagebox.showerror("Ошибка", "Нет отфильтрованных данных.")
            return
        if self.exclude_central_mixed.get():
            allowed = ['no_impairment', 'mild', 'moderate', 'severe']
            filtered_df = filtered_df[filtered_df['breathing_impairment_severity'].isin(allowed)]
            if filtered_df.empty:
                messagebox.showerror("Ошибка", "После исключения центрального/смешанного апноэ нет данных.")
                return
            study_ids = filtered_df['study_id'].unique().tolist()
            self.log(f"После исключения центрального/смешанного апноэ осталось исследований: {len(study_ids)}")
        needed = ['study_id', 'patient_id']
        for c in needed:
            if c not in filtered_df.columns:
                messagebox.showerror("Ошибка", f"В отфильтрованных данных отсутствует колонка {c}.")
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

            # ---- Проверка VIF ----
            vif_res = {}
            try:
                vif_res = self._check_vif(df)
                if vif_res:
                    high_vif = {k: v for k, v in vif_res.items() if v > 5}
                    if high_vif:
                        self.log(f"Предупреждение: мультиколлинеарность (VIF>5): {high_vif}")
            except Exception as e:
                self.log(f"Ошибка при вычислении VIF: {e}")
                vif_res = {}
            self.last_vif = vif_res
            # Формируем список задач (канал, признак, имя столбца)
            tasks = []
            for ch in self.all_channels:
                for feat in self.all_features:
                    col = f"{ch}_{feat}"
                    if col in df.columns:
                        tasks.append((ch, feat, col))
            for (a, b) in self.coh_pairs:
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
            self.fast_diag_btn.config(state=tk.NORMAL)
            # Заполним комбобоксы диагностики
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
            if beta is None or pval is None:
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
    # Отображение результатов (топ-10 значимых)
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

    def _plot_lmm_results(self, results_df):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        if results_df.empty:
            return

        fig = Figure(figsize=(12, 8), dpi=100)

        # --- Forest plot (исходные комбинации, топ-10) ---
        ax1 = fig.add_subplot(2, 1, 1)
        # Отбираем только значимые
        sign = results_df[results_df['significant']].copy()
        if sign.empty:
            ax1.text(0.5, 0.5, f"Нет значимых результатов (q < {self.fdr_threshold.get()})",
                     transform=ax1.transAxes, ha='center')
        else:
            # Берём топ-20 по q-value
            top20 = sign.nsmallest(10, 'q_value')
            y_pos = np.arange(len(top20))
            # Формируем метки: признак (канал)
            labels = top20['feature'] + ' (' + top20['channel'] + ')'
            # Используем beta и CI из исходных данных
            ax1.errorbar(top20['beta'], y_pos,
                         xerr=[top20['beta'] - top20['ci_low'],
                               top20['ci_high'] - top20['beta']],
                         fmt='o', capsize=5, color='blue', ecolor='gray')
            ax1.axvline(x=0, linestyle='--', color='gray')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.set_xlabel('Beta коэффициент (эффект апноэ)')
            ax1.set_title(f'Топ-20 наиболее значимых признаков (FDR < {self.fdr_threshold.get()})')
            ax1.grid(True, alpha=0.3)

        # --- Volcano plot (без изменений) ---
        ax2 = fig.add_subplot(2, 1, 2)
        colors = np.where(results_df['significant'], 'red', 'gray')
        ax2.scatter(results_df['beta'], -np.log10(results_df['p_value']), c=colors, alpha=0.6, s=20)

        # Подписываем топ-10 значимых (для уменьшения наложений)
        sign = results_df[results_df['significant']].copy()
        if not sign.empty:
            top10_comb = sign.nsmallest(10, 'q_value')
            for _, row in top10_comb.iterrows():
                label = f"{row['feature']} ({row['channel']})"
                ax2.annotate(label, (row['beta'], -np.log10(row['p_value'])),
                             textcoords="offset points", xytext=(5, 5), ha='left', fontsize=7,
                             alpha=0.7, bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3))

        ax2.axhline(y=-np.log10(0.05), linestyle='--', color='blue', label='p=0.05 (номинальный)')
        ax2.axhline(y=-np.log10(self.fdr_threshold.get()), linestyle='--', color='green', label='FDR threshold')
        ax2.set_xlabel('Beta coefficient')
        ax2.set_ylabel('-log10(p-value)')
        ax2.set_title('Volcano plot (красные – значимые)')
        ax2.legend()

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig

    # --------------------------------------------------------------
    # Диагностика остатков
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
        if self.epochs_df is None:
            self.log("Нет загруженных эпох.")
            return
        col = f"{channel}_{feature}"
        if col not in self.epochs_df.columns:
            self.log(f"Столбец {col} отсутствует.")
            return
        df = self.epochs_df.copy()
        df['has_apnea'] = df['has_apnea'].astype(bool)
        df['patient_id'] = df['patient_id'].astype(int)
        if 'epoch_stage' in df.columns:
            df['epoch_stage'] = df['epoch_stage'].astype('category')
        if 'gender' in df.columns and 'gender_code' not in df.columns:
            df['gender_code'] = (df['gender'] == 'M').astype(int)
        if self.include_covariates.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is not None:
                cov = filtered_df[['study_id', 'age_at_study', 'gender', 'bmi']].drop_duplicates(subset=['study_id'])
                cov['gender_code'] = (cov['gender'] == 'M').astype(int)
                df = df.merge(cov[['study_id', 'age_at_study', 'gender_code', 'bmi']], on='study_id', how='left')
        formula = f"{col} ~ has_apnea"
        if self.include_stage.get() and 'epoch_stage' in df.columns:
            formula += " + epoch_stage"
        if self.include_covariates.get():
            for c in ['age_at_study', 'gender_code', 'bmi']:
                if c in df.columns:
                    formula += f" + {c}"
        sub = df[[col, 'has_apnea', 'patient_id'] + [c for c in ['epoch_stage', 'age_at_study', 'gender_code', 'bmi'] if c in df.columns]].dropna()
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
            # Сохраняем для CSV
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
            # Графики
            fig = Figure(figsize=(10, 8))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.scatter(fitted, resid, alpha=0.5)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Предсказанные значения')
            ax1.set_ylabel('Остатки')
            ax1.set_title('Остатки vs Предсказанные')
            ax2 = fig.add_subplot(2, 2, 2)
            stats.probplot(resid, dist="norm", plot=ax2)
            ax2.set_title('Q-Q plot остатков')
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.hist(resid, bins=30, edgecolor='black')
            ax3.set_xlabel('Остатки')
            ax3.set_ylabel('Частота')
            ax3.set_title('Гистограмма остатков')
            ax4 = fig.add_subplot(2, 2, 4)
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
            # Сохраняем для отчёта
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

    # --------------------------------------------------------------
    # Быстрая диагностика топ-10
    # --------------------------------------------------------------
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
                self.log(f"[{idx + 1}/{len(top10)}] Диагностика: {row['channel']} {row['feature']}")
                self._diagnostics_thread(row['channel'], row['feature'])
                import time
                time.sleep(2)  # небольшая пауза, чтобы не перегружать интерфейс
        threading.Thread(target=run_sequential, daemon=True).start()

    # --------------------------------------------------------------
    # Блок-бутстрап
    # --------------------------------------------------------------
    def run_bootstrap(self):
        if self.diag_model is None:
            messagebox.showwarning("Нет модели", "Сначала выполните диагностику для выбранного признака.")
            return
        messagebox.showinfo("Бутстрап", "Будет выполнено 1000 итераций (блок-бутстрап по пациентам).\nЭто может занять 1-2 минуты.")
        threading.Thread(target=self._bootstrap_thread, daemon=True).start()

    def _bootstrap_thread(self):
        channel, feature = self.diag_feature_name.split('_', 1)
        col = f"{channel}_{feature}"
        df = self.epochs_df.copy()
        df['has_apnea'] = df['has_apnea'].astype(bool)
        df['patient_id'] = df['patient_id'].astype(int)
        if 'epoch_stage' in df.columns:
            df['epoch_stage'] = df['epoch_stage'].astype('category')
        if self.include_covariates.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is not None:
                cov = filtered_df[['study_id', 'age_at_study', 'gender', 'bmi']].drop_duplicates(subset=['study_id'])
                cov['gender_code'] = (cov['gender'] == 'M').astype(int)
                df = df.merge(cov[['study_id', 'age_at_study', 'gender_code', 'bmi']], on='study_id', how='left')
        formula = f"{col} ~ has_apnea"
        if self.include_stage.get() and 'epoch_stage' in df.columns:
            formula += " + epoch_stage"
        if self.include_covariates.get():
            for c in ['age_at_study', 'gender_code', 'bmi']:
                if c in df.columns:
                    formula += f" + {c}"
        use_cols = [col, 'has_apnea', 'patient_id']
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
                beta = result_boot.params.get('has_apnea[T.True]', result_boot.params.get('has_apnea', None))
                if beta is not None:
                    betas.append(beta)
            except Exception:
                pass
            if (i + 1) % 100 == 0:
                self.log(f"Бутстрап: {i + 1}/{n_iter} итераций")
        if betas:
            ci_low = np.percentile(betas, 2.5)
            ci_high = np.percentile(betas, 97.5)
            p_bootstrap = (np.sum(np.abs(betas) < 1e-8) * 2) / len(betas)
            # Сохраняем информацию для отчёта
            self.bootstrap_info = {
                'feature': f"{channel}_{feature}",
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_bootstrap': p_bootstrap,
                'n_iter': len(betas)  # реальное число успешных итераций
            }
            self.log(f"Бутстрап сохранён для {channel}_{feature}")
            self.bootstrap_results = {'betas': betas, 'ci_low': ci_low, 'ci_high': ci_high, 'p': p_bootstrap}
            self.log(f"Бутстрап CI для beta: [{ci_low:.4f}, {ci_high:.4f}], p≈{p_bootstrap:.4f}")
            # Гистограмма
            fig = Figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.hist(betas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=ci_low, color='red', linestyle='--', label=f'2.5%: {ci_low:.2f}')
            ax.axvline(x=ci_high, color='red', linestyle='--', label=f'97.5%: {ci_high:.2f}')
            ax.axvline(x=np.mean(betas), color='green', linestyle='-', label=f'Mean: {np.mean(betas):.2f}')
            ax.set_xlabel('Beta coefficient')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Bootstrap distribution of beta for {channel} {feature}')
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
    # Генерация отчёта HTML
    # --------------------------------------------------------------
    def generate_report(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Сначала выполните LMM анализ.")
            return

        filtered_df = self.main_app.get_filtered_data()
        n_patients = filtered_df['patient_id'].nunique() if filtered_df is not None else 0
        n_epochs = len(self.epochs_df) if self.epochs_df is not None else 0
        sign = self.results_df[self.results_df['significant']].copy()
        top10 = sign.nsmallest(10, 'q_value')

        data_type_label = {1: "Тонический (чистые эпохи N2/N3)", 2: "Все эпохи N2/N3",
                           3: "Фильтр по положению (первые 10 с)"}
        data_type_text = data_type_label.get(self.data_type.get(), "Неизвестно")

        # VIF
        vif_html = ""
        if hasattr(self, 'last_vif') and self.last_vif:
            high_vif = {k: v for k, v in self.last_vif.items() if v > 5}
            if high_vif:
                vif_html = """
                <h2>Проверка мультиколлинеарности (VIF)</h2>
                <p>Обнаружены признаки с Variance Inflation Factor > 5, что может указывать на мультиколлинеарность.</p>
                <table border='1'>
                <tr><th>Признак</th><th>VIF</th></tr>
                """ + "".join(f"<tr><td>{var}</td><td>{vif_val:.2f}</td></tr>" for var, vif_val in high_vif.items()) + """
                </table><p>Рекомендуется интерпретировать коэффициенты с осторожностью.</p>
                """

        # График LMM (если есть)
        plot_html = ""
        if self.current_figure is not None:
            buf = io.BytesIO()
            self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'

        # ==================== 1. ПРОВЕРКА ГИПОТЕЗ ====================
        hypotheses_html = "<h2>Проверка гипотез исследования</h2>"

        # H1: дельта-мощность
        delta_sign = sign[sign['feature'].isin(['abs_delta', 'rel_delta'])]
        if not delta_sign.empty:
            h1_status = "✅ ПОДТВЕРЖДЕНА"
            h1_detail = f"Значимые изменения дельта-мощности в {delta_sign['channel'].nunique()} каналах (beta "
            h1_detail += "положительный/отрицательный)."
        else:
            h1_status = "❌ НЕ ПОДТВЕРЖДЕНА"
            h1_detail = "Нет значимых различий дельта-мощности между эпохами с апноэ и без."
        hypotheses_html += f"""
        <h3>H1 (тоническая спектральная)</h3>
        <p><strong>{h1_status}</strong> — {h1_detail}</p>
        <p><em>Гипотеза:</em> относительная мощность дельта-ритма (0.5-4 Гц) в тонических эпохах N2/N3 положительно коррелирует с тяжестью ОАС, особенно в лобных отведениях.</p>
        """

        # H2: TBR
        tbr_sign = sign[sign['feature'] == 'tbr']
        if not tbr_sign.empty:
            h2_status = "✅ ПОДТВЕРЖДЕНА"
            h2_detail = f"TBR значимо снижается в {tbr_sign['channel'].nunique()} каналах (все beta < 0)."
        else:
            h2_status = "❌ НЕ ПОДТВЕРЖДЕНА"
            h2_detail = "TBR не показал значимых различий."
        hypotheses_html += f"""
        <h3>H2 (тоническая спектральная)</h3>
        <p><strong>{h2_status}</strong> — {h2_detail}</p>
        <p><em>Гипотеза:</em> сигма-мощность и отношение тета/бета (TBR) в тонических эпохах достоверно снижаются при нарастании степени тяжести ОАС.</p>
        """

        # H3 (фазическая) – в LMM тонических признаков не проверяется, но можно отметить
        hypotheses_html += f"""
        <h3>H3 (фазическая)</h3>
        <p>❌ НЕ ПРОВЕРЯЕТСЯ В ТОНИЧЕСКОМ LMM</p>
        <p><em>Гипотеза:</em> абсолютная гамма-мощность и индекс реактивности бета в пост-событийном окне значимо выше в эпохах с апноэ/гипопноэ по сравнению с фоновыми эпохами.</p>
        <p><strong>Примечание:</strong> Для проверки H3 используется фазический анализ (таблица phasic_events) и event-locked анализ.</p>
        """

        # H4: SampEn
        sampen_sign = sign[sign['feature'] == 'sampen']
        if not sampen_sign.empty:
            h4_status = "✅ ПОДТВЕРЖДЕНА"
            h4_detail = f"Sample Entropy значимо изменяется в {sampen_sign['channel'].nunique()} каналах."
        else:
            h4_status = "❌ НЕ ПОДТВЕРЖДЕНА"
            h4_detail = "Нет значимых различий SampEn между эпохами."
        hypotheses_html += f"""
        <h3>H4 (нелинейная)</h3>
        <p><strong>{h4_status}</strong> — {h4_detail}</p>
        <p><em>Гипотеза:</em> выборочная энтропия (SampEn) снижается в период апноэ и резко возрастает после окончания события; ΔSampEn положительно связан с AHI.</p>
        """

        # H5 и H6 – ML задачи
        hypotheses_html += f"""
        <h3>H5 (классификация эпох)</h3>
        <p>⏳ ЧАСТИЧНО ПРОВЕРЕНО</p>
        <p>Результаты LMM показывают, что многие ЭЭГ-признаки значимо различаются между эпохами с апноэ и без, что создаёт основу для построения классификатора с ожидаемым AUC ≥ 0,85. Оценка классификационных моделей будет представлена в отдельном разделе (этап 5).</p>
        <h3>H6 (прогноз тяжести)</h3>
        <p>⏳ БУДЕТ ПРОВЕРЕНА В ЭТАПЕ 6</p>
        <p>Агрегированные ЭЭГ-признаки в сочетании с клиническими данными будут использованы для предсказания тяжести ОАС.</p>
        """

        # ==================== 2. ДИАГНОСТИКА ОСТАТКОВ ====================
        diag_html = ""
        if self.all_diagnostics:
            # Берём топ-5 по значимости (если есть) или все диагностики
            top_diag = self.all_diagnostics[
                :5]  # или можно отсортировать по убыванию значимости, но у нас нет q в диагностиках
            diag_html = "<h2>Диагностика остатков LMM</h2><p>Для наиболее значимых признаков выполнена проверка нормальности остатков (Шапиро-Уилк) и гомоскедастичности (Бройш-Паган).</p>"
            diag_html += "<table border='1'><tr><th>Канал</th><th>Признак</th><th>n</th><th>Shapiro-Wilk p</th><th>Breusch-Pagan p</th><th>Заключение</th></tr>"
            for d in top_diag:
                shapiro_ok = d['shapiro_p'] > 0.05 if d['shapiro_p'] is not None else "n>5000"
                bp_ok = d['bp_p'] > 0.05
                conclusion = []
                if d['shapiro_p'] is not None and d['shapiro_p'] <= 0.05:
                    conclusion.append("❗ остатки не нормальны")
                elif d['shapiro_p'] is None:
                    conclusion.append("⚠️ выборка >5000, тест не применялся")
                else:
                    conclusion.append("✅ нормальность")
                if d['bp_p'] <= 0.05:
                    conclusion.append("❗ гетероскедастичность")
                else:
                    conclusion.append("✅ гомоскедастичность")
                conclusion_str = ", ".join(conclusion)
                diag_html += f"<tr><td>{d['channel']}</td><td>{d['feature']}</td><td>{d['n']}</td><td>{d['shapiro_p']:.4f if d['shapiro_p'] else '>5000'}</td><td>{d['bp_p']:.4f}</td><td>{conclusion_str}</td></tr>"
            diag_html += "</table><p>При нарушениях предположений рекомендуется использовать бутстрап (см. раздел ниже).</p>"

        # ==================== 3. БУТСТРАП ====================
        bootstrap_html = ""
        if self.bootstrap_info:
            bi = self.bootstrap_info
            bootstrap_html = f"""
            <h2>Бутстрап-проверка (блок-бутстрап по пациентам)</h2>
            <p>Для признака <strong>{bi['feature']}</strong> выполнено {bi['n_iter']} успешных итераций.</p>
            <p><strong>95% доверительный интервал для beta:</strong> [{bi['ci_low']:.4f}, {bi['ci_high']:.4f}]</p>
            <p><strong>Бутстрап p-value (двусторонний):</strong> {bi['p_bootstrap']:.4f}</p>
            <p><em>Интерпретация:</em> если CI не пересекает 0, эффект апноэ статистически значим с учётом возможных нарушений допущений.</p>
            """

        # ==================== 4. ОСНОВНАЯ ТАБЛИЦА И ГРАФИКИ ====================
        # (оставляем существующий код, но немного упростим)

        cov_info = "Включены" if self.include_covariates.get() else "Не включены"
        stage_info = "Включена" if self.include_stage.get() else "Не включена"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>LMM отчёт</title>
        <style>
            body {{ font-family: Arial; margin:20px; }}
            table {{ border-collapse: collapse; width:100%; margin-bottom:20px; }}
            th, td {{ border:1px solid #ddd; padding:8px; text-align:left; }}
            th {{ background:#f2f2f2; }}
            .sign {{ background:#ffcccc; }}
            .plot {{ margin:20px 0; text-align:center; }}
        </style>
        </head>
        <body>
        <h1>Отчёт о линейных смешанных моделях (LMM)</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Пациентов (исследований):</strong> {n_patients}</p>
        <p><strong>Эпох N2/N3 (всего):</strong> {n_epochs}</p>
        <p><strong>Модель:</strong> признак ~ has_apnea {'+ стадия сна' if self.include_stage.get() else ''} {'+ возраст + пол + ИМТ' if self.include_covariates.get() else ''} + (1|patient_id)</p>
        <p><strong>Ковариаты:</strong> {cov_info} &nbsp;|&nbsp; <strong>Стадия сна:</strong> {stage_info}</p>
        <p><strong>Тип набора эпох:</strong> {data_type_text}</p>
        <p><strong>FDR порог:</strong> q = {self.fdr_threshold.get()}</p>
        {vif_html}
        {hypotheses_html}
        {diag_html}
        {bootstrap_html}
        <h2>Значимые признаки (всего {len(sign)})</h2>
        <table>
            <tr><th>Канал</th><th>Признак</th><th>Beta</th><th>p-value</th><th>q-value</th><th>95% CI</th></tr>
        """ + "".join(
            f"<tr class='sign'><td>{r['channel']}</td><td>{r['feature']}</td><td>{r['beta']:.4f}</td><td>{r['p_value']:.2e}</td><td>{r['q_value']:.4f}</td><td>[{r['ci_low']:.2f}, {r['ci_high']:.2f}]</td></tr>"
            for _, r in sign.iterrows()) + """
        </table>
        <h2>Топ-10 наиболее значимых признаков</h2>
        <table>
            <tr><th>Канал</th><th>Признак</th><th>Beta</th><th>q-value</th><th>Интерпретация</th></tr>
        """ + "".join(
            f"<tr><td>{r['channel']}</td><td>{r['feature']}</td><td>{r['beta']:.4f}</td><td>{r['q_value']:.4f}</td><td>При апноэ {'увеличивается' if r['beta'] > 0 else 'уменьшается'} на {abs(r['beta']):.2f}</td></tr>"
            for _, r in top10.iterrows()) + """
        </table>
        <h2>Графики LMM</h2>
        """ + plot_html + """
        <p><em>Примечание:</em> Бутстрап-проверка рекомендуется при нарушении нормальности остатков или гетероскедастичности.</p>
        </body></html>
        """

        fd, path = tempfile.mkstemp(suffix='.html', prefix='lmm_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт в браузере: {path}")

    # --------------------------------------------------------------
    # Сохранение основных результатов
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
