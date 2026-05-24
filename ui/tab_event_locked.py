# ui/tab_event_locked.py
"""
Event‑locked анализ гамма‑активности (30–45 Гц) в соответствии с главой 2, п. 2.4.3.
- Выравнивание событий по окончанию (offset), интервал -60 … +30 с.
- Скользящие окна 5 с с шагом 1 с (перекрытие 4 с).
- Нормализация относительно фона (-60 … -30 с): γ_norm(%) = 100*(P - P_bg)/P_bg.
- Построение средних кривых для групп тяжести ОАС (норма, лёгкая, умеренная, тяжёлая).
- Вычисление метрик: пик γ в окне 0-5 с, латентность пика, AUC 0-10 с, AUC 10-30 с.
- Агрегация метрик на уровне пациентов (среднее, SEM, 95% CI).
- Бутстрап (1000 итераций) для сравнения групп.
"""

import threading
import tkinter as tk
from tkinter import messagebox, ttk
import io
import base64
import webbrowser
import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.stats import shapiro
from ui.base_tab import BaseTab
from core.api_client import get_event_time_series, get_studies

# Размер страницы для загрузки (10 000 записей)
EVENT_PAGE_SIZE = 100000


class EventLockedTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        self.channel = tk.StringVar(value='C3')
        self.time_min = tk.DoubleVar(value=-60.0)
        self.time_max = tk.DoubleVar(value=30.0)
        self.use_filtered = tk.BooleanVar(value=True)      # по умолчанию True
        self.confidence_level = tk.DoubleVar(value=0.95)
        self.use_cache = tk.BooleanVar(value=True)
        self.stop_flag = False

        # Результаты
        self.results_metrics = None          # DataFrame с метриками по группам (mean, sem, ci, n)
        self.summary_curves = None           # список словарей с кривыми (severity, time, mean, ci, n)
        self.common_time = None
        self.current_figure = None
        self.patient_curves = None           # список кривых по пациентам (для бутстрапа)
        self.patient_metrics_df = None       # DataFrame per‑patient метрик (для нормальности и бутстрапа)

        # Хранилища для диагностики
        self.normality_results = {}
        self.bootstrap_results = {}
        self.diag_figure = None
        self.bootstrap_figure = None

        self._create_widgets()

    def _create_widgets(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned)
        right_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        paned.add(right_frame, weight=1)

        # ========== ЛЕВАЯ ПАНЕЛЬ ==========
        # Важно: все внутренние контейнеры должны растягиваться по горизонтали
        info_frame = ttk.LabelFrame(left_frame, text="Источник данных", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, text="Используются отфильтрованные данные из вкладки 'Загрузка'",
                  foreground="blue").pack(anchor=tk.W)
        ttk.Label(info_frame, text="(токен и URL не требуются)").pack(anchor=tk.W)

        desc_frame = ttk.LabelFrame(left_frame, text="Event‑locked анализ (Глава 2, п. 2.4.3)", padding=5)
        desc_frame.pack(fill=tk.X, padx=5, pady=5)
        desc_text = ("Динамика γ‑активности (30–45 Гц) относительно окончания респираторного события. "
                     "Окна: 5 с, шаг 1 с. Нормировка на фон -60…-30 с. Строятся средние кривые для групп "
                     "тяжести ОАС с 95% ДИ. Метрики: пик (0-5 с), латентность, AUC (0-10 и 10-30 с).")
        ttk.Label(desc_frame, text=desc_text, justify=tk.LEFT, wraplength=600).pack(anchor=tk.W, fill=tk.X, padx=5,
                                                                                    pady=2)
        ttk.Button(desc_frame, text="Показать инструкцию", command=self.show_instructions).pack(anchor=tk.W, padx=5,
                                                                                                pady=2)

        param_frame = ttk.LabelFrame(left_frame, text="Параметры анализа", padding=5)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        # Растягиваем колонки с виджетами
        param_frame.columnconfigure(1, weight=1)
        param_frame.columnconfigure(3, weight=1)

        ttk.Label(param_frame, text="Канал:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(param_frame, textvariable=self.channel, values=['C3', 'C4'], state='readonly').grid(row=0,
                                                                                                         column=1,
                                                                                                         padx=5,
                                                                                                         sticky=tk.W)
        ttk.Label(param_frame, text="Интервал времени (сек):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.time_min, width=8).grid(row=1, column=1, padx=2, sticky=tk.W)
        ttk.Label(param_frame, text="до").grid(row=1, column=2, padx=2)
        ttk.Entry(param_frame, textvariable=self.time_max, width=8).grid(row=1, column=3, padx=2, sticky=tk.W)
        ttk.Label(param_frame, text="Доверительный интервал:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.confidence_level, width=6).grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Checkbutton(param_frame, text="Использовать отфильтрованные исследования",
                        variable=self.use_filtered).grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=5)

        cache_frame = ttk.LabelFrame(left_frame, text="Кэширование", padding=5)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cache_frame, text="Использовать кэш (ускоряет повторные запуски)",
                        variable=self.use_cache).pack(anchor=tk.W)
        ttk.Label(cache_frame, text=f"Страница загрузки: {EVENT_PAGE_SIZE} записей", foreground="gray").pack(
            anchor=tk.W)

        norm_frame = ttk.LabelFrame(left_frame, text="Проверка нормальности метрик", padding=5)
        norm_frame.pack(fill=tk.X, padx=5, pady=5)
        norm_frame.columnconfigure(1, weight=1)

        ttk.Label(norm_frame, text="Группа:").grid(row=0, column=0, sticky=tk.W)
        self.norm_group = tk.StringVar()
        self.norm_group_combo = ttk.Combobox(norm_frame, textvariable=self.norm_group, state='readonly', width=15)
        self.norm_group_combo.grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Label(norm_frame, text="Метрика:").grid(row=1, column=0, sticky=tk.W)
        self.norm_metric = tk.StringVar()
        self.norm_metric_combo = ttk.Combobox(norm_frame, textvariable=self.norm_metric, state='readonly',
                                              values=['peak_amplitude', 'peak_latency', 'auc_0_10', 'auc_10_30'],
                                              width=150)
        self.norm_metric_combo.grid(row=1, column=1, padx=5, sticky=tk.W)
        self.norm_btn = ttk.Button(norm_frame, text="Проверить нормальность", command=self.check_normality_metrics,
                                   state=tk.DISABLED)
        self.norm_btn.grid(row=2, column=0, columnspan=2, pady=2)

        bootstrap_frame = ttk.LabelFrame(left_frame, text="Бутстрап сравнения групп", padding=5)
        bootstrap_frame.pack(fill=tk.X, padx=5, pady=5)
        bootstrap_frame.columnconfigure(1, weight=1)

        ttk.Label(bootstrap_frame, text="Группа 1:").grid(row=0, column=0, sticky=tk.W)
        self.boot_group1 = tk.StringVar()
        self.boot_group1_combo = ttk.Combobox(bootstrap_frame, textvariable=self.boot_group1, state='readonly',
                                              width=15)
        self.boot_group1_combo.grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Label(bootstrap_frame, text="Группа 2:").grid(row=1, column=0, sticky=tk.W)
        self.boot_group2 = tk.StringVar()
        self.boot_group2_combo = ttk.Combobox(bootstrap_frame, textvariable=self.boot_group2, state='readonly',
                                              width=15)
        self.boot_group2_combo.grid(row=1, column=1, padx=5, sticky=tk.W)
        ttk.Label(bootstrap_frame, text="Метрика:").grid(row=2, column=0, sticky=tk.W)
        self.boot_metric = tk.StringVar()
        self.boot_metric_combo = ttk.Combobox(bootstrap_frame, textvariable=self.boot_metric, state='readonly',
                                              values=['peak_amplitude', 'peak_latency', 'auc_0_10', 'auc_10_30'],
                                              width=15)
        self.boot_metric_combo.grid(row=2, column=1, padx=5, sticky=tk.W)
        self.bootstrap_btn = ttk.Button(bootstrap_frame, text="Запустить бутстрап (1000 итераций)",
                                        command=self.run_bootstrap, state=tk.DISABLED)
        self.bootstrap_btn.grid(row=3, column=0, columnspan=2, pady=5)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Запустить анализ", command=self.run_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить метрики (CSV)", command=self.save_results_csv,
                                       state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график (PNG)", command=self.save_plot,
                                        state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать отчёт", command=self.generate_report,
                                     state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=5)

        ttk.Frame(left_frame, height=1).pack(fill=tk.BOTH, expand=True)

        # ========== ПРАВАЯ ПАНЕЛЬ: Notebook ==========
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="График")
        self.plot_frame = ttk.Frame(self.tab_plot)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="Метрики")
        self.metrics_tree = ttk.Treeview(self.tab_metrics,
                                         columns=('severity', 'peak_amp', 'peak_latency', 'auc_0_10', 'auc_10_30',
                                                  'n_patients'),
                                         show='headings')
        self.metrics_tree.heading('severity', text='Группа')
        self.metrics_tree.heading('peak_amp', text='Пик γ (%)\nmean ± SEM [95% CI]')
        self.metrics_tree.heading('peak_latency', text='Латентность (с)\nmean ± SEM [95% CI]')
        self.metrics_tree.heading('auc_0_10', text='AUC 0-10 с\nmean ± SEM [95% CI]')
        self.metrics_tree.heading('auc_10_30', text='AUC 10-30 с\nmean ± SEM [95% CI]')
        self.metrics_tree.heading('n_patients', text='n')
        self.metrics_tree.pack(fill=tk.BOTH, expand=True)

        self.tab_diag = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diag, text="Диагностика")
        self.diag_frame = ttk.Frame(self.tab_diag)
        self.diag_frame.pack(fill=tk.BOTH, expand=True)
        self.diag_text = tk.Text(self.diag_frame, wrap=tk.WORD, font=("Courier New", 9), height=10)
        self.diag_text.pack(fill=tk.X, padx=5, pady=5)
        self.diag_plot_frame = ttk.Frame(self.tab_diag)
        self.diag_plot_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_bootstrap = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bootstrap, text="Бутстрап")
        self.bootstrap_tree_frame = ttk.Frame(self.tab_bootstrap)
        self.bootstrap_tree_frame.pack(fill=tk.BOTH, expand=True)
        self.bootstrap_tree = ttk.Treeview(self.bootstrap_tree_frame,
                                           columns=('comparison', 'metric', 'diff', 'ci_low', 'ci_high', 'p_value',
                                                    'significant'),
                                           show='headings')
        for col in ('comparison', 'metric', 'diff', 'ci_low', 'ci_high', 'p_value', 'significant'):
            self.bootstrap_tree.heading(col, text=col)
            self.bootstrap_tree.column(col, width=100)
        self.bootstrap_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(self.bootstrap_tree_frame, orient=tk.VERTICAL, command=self.bootstrap_tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.bootstrap_tree.configure(yscrollcommand=sb.set)

        self.tab_log = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_log, text="Лог")
        self.log_text = tk.Text(self.tab_log, wrap=tk.WORD, font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.main_app.log(msg)

    def stop_analysis(self):
        self.stop_flag = True
        self.log("Остановка запрошена...")

    def show_instructions(self):
        msg = (
            "ИНСТРУКЦИЯ ПО EVENT‑LOCKED АНАЛИЗУ (Глава 2, п. 2.4.3)\n"
            "===================================================\n"
            "1. Загрузите и отфильтруйте данные на вкладке 'Загрузка и фильтры'.\n"
            "2. Выберите канал (C3 или C4), временной интервал (по умолчанию -60…+30 с).\n"
            "3. Нажмите 'Запустить анализ'. Строятся средние кривые γ‑мощности для групп тяжести ОАС.\n"
            "4. После завершения откроется вкладка 'График' и таблица метрик.\n"
            "5. Для проверки нормальности метрик выберите группу и метрику → 'Проверить нормальность'.\n"
            "6. Для сравнения групп выберите две группы, метрику и нажмите 'Бутстрап' (1000 итераций).\n"
            "7. Кнопка 'Сформировать отчёт' создаст HTML-отчёт со всеми графиками и таблицами.\n"
            "8. Результаты можно сохранить в CSV (метрики) и PNG (график).\n"
            "Примечание: нормализация γ-мощности выполняется API относительно интервала -60…-30 с."
        )
        messagebox.showinfo("Инструкция", msg)

    def run_analysis(self):
        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_csv_btn.config(state=tk.DISABLED)
        self.save_plot_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        self.norm_btn.config(state=tk.DISABLED)
        self.bootstrap_btn.config(state=tk.DISABLED)
        self.log("Загрузка данных...")
        thread = threading.Thread(target=self._run_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_analysis_thread(self):
        try:
            load_tab = self.main_app.tabs['load']
            api_url = load_tab.api_url.get().rstrip('/')
            token = load_tab.token.get().strip()
            if not api_url or not token:
                self.log("Ошибка: не указаны URL или токен API. Перейдите на вкладку 'Загрузка и фильтры'.")
                return

            study_ids = None
            if self.use_filtered.get():
                filtered_df = self.main_app.get_filtered_data()
                if filtered_df is None or filtered_df.empty:
                    self.log("Нет отфильтрованных данных. Сначала загрузите и отфильтруйте исследования.")
                    return
                study_ids = filtered_df['study_id'].unique().tolist()
                if not study_ids:
                    self.log("В отфильтрованных данных нет study_id.")
                    return
                self.log(f"Используем исследования: {len(study_ids)}")

            def update_progress(page, total, _):
                if total > 0:
                    self.main_app.set_progress(int(page / total * 100))

            self.main_app.set_progress(0)
            ts_data = get_event_time_series(
                api_url, token,
                study_ids=study_ids,
                channel=self.channel.get(),
                time_from_offset_min=self.time_min.get(),
                time_from_offset_max=self.time_max.get(),
                stop_check=lambda: self.stop_flag,
                progress_callback=update_progress,
                use_cache=self.use_cache.get(),
                page_size=EVENT_PAGE_SIZE
            )
            self.main_app.set_progress(100)
            if not ts_data:
                self.log("Нет данных для выбранного канала и интервала.")
                return
            df = pd.DataFrame(ts_data)
            self.log(f"Загружено {len(df)} временных точек")

            # ---- ОЧИСТКА ДАННЫХ (преобразование типов и удаление NaN) ----
            df['gamma_power_norm_pct'] = pd.to_numeric(df['gamma_power_norm_pct'], errors='coerce')
            df['time_from_offset'] = pd.to_numeric(df['time_from_offset'], errors='coerce')
            df = df.dropna(subset=['gamma_power_norm_pct', 'time_from_offset'])
            # Опциональная фильтрация выбросов (можно раскомментировать)
            # df = df[(df['gamma_power_norm_pct'] > -200) & (df['gamma_power_norm_pct'] < 500)]
            self.log(f"После очистки осталось {len(df)} временных точек")
            # ------------------------------------------------------------

            # Получаем severity исследований
            studies = get_studies(api_url, token, stop_check=lambda: self.stop_flag)
            study_severity = {s['study_id']: s.get('breathing_impairment_severity', 'unknown') for s in studies}
            df['severity'] = df['study_id'].map(study_severity)

            # Оставляем только основные группы тяжести (норма, лёгкая, умеренная, тяжёлая)
            valid_severities = ['no_impairment', 'mild', 'moderate', 'severe']
            df = df[df['severity'].isin(valid_severities)]
            if df.empty:
                self.log("Нет событий с допустимой тяжестью ОАС (no_impairment, mild, moderate, severe).")
                return
            self.log(f"Группы: {df['severity'].unique()}")

            # Общий временной массив с шагом 1 секунда (соответствует скользящим окнам 5 с, шаг 1 с)
            common_time = np.arange(self.time_min.get(), self.time_max.get() + 1, 1)
            patient_curves = []
            for (pid, sev), group in df.groupby(['patient_id', 'severity']):
                group = group.sort_values('time_from_offset')
                group = group.drop_duplicates(subset=['time_from_offset'])
                if len(group) < 2:
                    continue
                # Интерполяция (линейная) на общую сетку
                f = interp1d(group['time_from_offset'], group['gamma_power_norm_pct'],
                             kind='linear', fill_value='extrapolate')
                y_interp = f(common_time)
                # Ограничиваем экстраполированные значения физиологическим диапазоном
                y_interp = np.clip(y_interp, -200, 500)
                patient_curves.append({
                    'patient_id': pid,
                    'severity': sev,
                    'time': common_time,
                    'gamma': y_interp
                })
            if not patient_curves:
                self.log("Не удалось построить кривые для пациентов (недостаточно точек).")
                return
            self.patient_curves = patient_curves
            self.common_time = common_time

            # --- Построение групповых кривых (для визуализации) ---
            grouped_by_sev = {}
            for rec in patient_curves:
                sev = rec['severity']
                grouped_by_sev.setdefault(sev, []).append(rec['gamma'])
            ci_z = stats.norm.ppf(1 - (1 - self.confidence_level.get()) / 2)
            summary_curves = []
            for sev, curves in grouped_by_sev.items():
                curves = np.array(curves)
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0, ddof=1)
                n_patients = len(curves)
                ci = ci_z * std_curve / np.sqrt(n_patients)
                summary_curves.append({
                    'severity': sev,
                    'time': common_time,
                    'mean': mean_curve,
                    'ci': ci,
                    'n': n_patients
                })
            self.summary_curves = summary_curves
            self._plot_curves(summary_curves, common_time)

            # --- Вычисление метрик на уровне пациентов ---
            # Сначала per‑patient метрики
            patient_metrics = self._compute_patient_metrics_from_curves(patient_curves, common_time)
            self.patient_metrics_df = patient_metrics  # для нормальности и бутстрапа

            # Агрегация по группам
            agg_metrics = []
            for sev in valid_severities:
                sub = patient_metrics[patient_metrics['severity'] == sev]
                if sub.empty:
                    continue
                n = len(sub)
                # Для каждой метрики: mean, sem, ci_low, ci_high
                peak_amp_mean = sub['peak_amplitude'].mean()
                peak_amp_sem = sub['peak_amplitude'].sem(ddof=1) if n > 1 else np.nan
                peak_amp_ci = ci_z * peak_amp_sem if n > 1 else np.nan

                peak_lat_mean = sub['peak_latency'].mean()
                peak_lat_sem = sub['peak_latency'].sem(ddof=1) if n > 1 else np.nan
                peak_lat_ci = ci_z * peak_lat_sem if n > 1 else np.nan

                auc1_mean = sub['auc_0_10'].mean()
                auc1_sem = sub['auc_0_10'].sem(ddof=1) if n > 1 else np.nan
                auc1_ci = ci_z * auc1_sem if n > 1 else np.nan

                auc2_mean = sub['auc_10_30'].mean()
                auc2_sem = sub['auc_10_30'].sem(ddof=1) if n > 1 else np.nan
                auc2_ci = ci_z * auc2_sem if n > 1 else np.nan

                agg_metrics.append({
                    'severity': sev,
                    'peak_amp_mean': peak_amp_mean,
                    'peak_amp_sem': peak_amp_sem,
                    'peak_amp_ci_low': peak_amp_mean - peak_amp_ci if n > 1 else np.nan,
                    'peak_amp_ci_high': peak_amp_mean + peak_amp_ci if n > 1 else np.nan,
                    'peak_lat_mean': peak_lat_mean,
                    'peak_lat_sem': peak_lat_sem,
                    'peak_lat_ci_low': peak_lat_mean - peak_lat_ci if n > 1 else np.nan,
                    'peak_lat_ci_high': peak_lat_mean + peak_lat_ci if n > 1 else np.nan,
                    'auc_0_10_mean': auc1_mean,
                    'auc_0_10_sem': auc1_sem,
                    'auc_0_10_ci_low': auc1_mean - auc1_ci if n > 1 else np.nan,
                    'auc_0_10_ci_high': auc1_mean + auc1_ci if n > 1 else np.nan,
                    'auc_10_30_mean': auc2_mean,
                    'auc_10_30_sem': auc2_sem,
                    'auc_10_30_ci_low': auc2_mean - auc2_ci if n > 1 else np.nan,
                    'auc_10_30_ci_high': auc2_mean + auc2_ci if n > 1 else np.nan,
                    'n_patients': n
                })
            self.results_metrics = pd.DataFrame(agg_metrics)
            self._display_metrics_table(self.results_metrics)

            self.save_csv_btn.config(state=tk.NORMAL)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.norm_btn.config(state=tk.NORMAL)
            self.bootstrap_btn.config(state=tk.NORMAL)

            groups = self.results_metrics['severity'].tolist()
            self.norm_group_combo['values'] = groups
            self.boot_group1_combo['values'] = groups
            self.boot_group2_combo['values'] = groups
            if groups:
                self.norm_group.set(groups[0])
                self.boot_group1.set(groups[0])
                self.boot_group2.set(groups[1] if len(groups) > 1 else groups[0])

            self.log("Event‑locked анализ завершён.")
        except Exception as e:
            self.log(f"Ошибка: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _compute_patient_metrics_from_curves(self, patient_curves, common_time):
        """Вычисляет per‑patient метрики (пик, латентность, AUC) по индивидуальным кривым."""
        records = []
        for rec in patient_curves:
            sev = rec['severity']
            y = rec['gamma']
            t = common_time
            # Пик в окне 0-5 с
            mask_peak = (t >= 0) & (t <= 5)
            if np.any(mask_peak):
                peak_amp = np.max(y[mask_peak])
                peak_lat = t[mask_peak][np.argmax(y[mask_peak])]
            else:
                peak_amp = np.nan
                peak_lat = np.nan
            # AUC 0-10 с
            mask_auc1 = (t >= 0) & (t <= 10)
            auc1 = trapezoid(y[mask_auc1], t[mask_auc1]) if np.any(mask_auc1) else np.nan
            # AUC 10-30 с
            mask_auc2 = (t >= 10) & (t <= 30)
            auc2 = trapezoid(y[mask_auc2], t[mask_auc2]) if np.any(mask_auc2) else np.nan
            records.append({
                'patient_id': rec['patient_id'],
                'severity': sev,
                'peak_amplitude': peak_amp,
                'peak_latency': peak_lat,
                'auc_0_10': auc1,
                'auc_10_30': auc2
            })
        return pd.DataFrame(records)

    def _plot_curves(self, summary_curves, common_time):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        # Порядок отображения (норма, лёгкая, умеренная, тяжёлая)
        severity_order = ['no_impairment', 'mild', 'moderate', 'severe']
        severity_labels = {
            'no_impairment': 'Норма',
            'mild': 'Лёгкая',
            'moderate': 'Умеренная',
            'severe': 'Тяжёлая'
        }
        for sev in severity_order:
            rec = next((r for r in summary_curves if r['severity'] == sev), None)
            if rec is not None:
                label = severity_labels.get(sev, sev)
                ax.plot(common_time, rec['mean'], label=label, linewidth=2)
                ax.fill_between(common_time, rec['mean'] - rec['ci'], rec['mean'] + rec['ci'], alpha=0.2)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='offset')
        ax.axvspan(-60, -30, alpha=0.1, color='gray', label='фон (нормировка)')
        ax.set_xlabel('Время относительно окончания события (сек)')
        ax.set_ylabel('Гамма-мощность, % от фона')
        ax.set_title('Динамика γ-активности (30–45 Гц) вокруг респираторных событий')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig

    def _display_metrics_table(self, metrics_df):
        severity_labels = {
            'no_impairment': 'Норма',
            'mild': 'Лёгкая',
            'moderate': 'Умеренная',
            'severe': 'Тяжёлая'
        }
        for row in self.metrics_tree.get_children():
            self.metrics_tree.delete(row)
        for _, row in metrics_df.iterrows():
            peak_amp_str = f"{row['peak_amp_mean']:.2f} ± {row['peak_amp_sem']:.2f} [{row['peak_amp_ci_low']:.2f}, {row['peak_amp_ci_high']:.2f}]"
            peak_lat_str = f"{row['peak_lat_mean']:.2f} ± {row['peak_lat_sem']:.2f} [{row['peak_lat_ci_low']:.2f}, {row['peak_lat_ci_high']:.2f}]"
            auc1_str = f"{row['auc_0_10_mean']:.2f} ± {row['auc_0_10_sem']:.2f} [{row['auc_0_10_ci_low']:.2f}, {row['auc_0_10_ci_high']:.2f}]"
            auc2_str = f"{row['auc_10_30_mean']:.2f} ± {row['auc_10_30_sem']:.2f} [{row['auc_10_30_ci_low']:.2f}, {row['auc_10_30_ci_high']:.2f}]"
            self.metrics_tree.insert('', 'end', values=(
                severity_labels.get(row['severity'], row['severity']),
                peak_amp_str, peak_lat_str, auc1_str, auc2_str,
                row['n_patients']
            ))

    def check_normality_metrics(self):
        if self.patient_metrics_df is None or self.patient_metrics_df.empty:
            messagebox.showwarning("Нет данных", "Сначала выполните анализ.")
            return
        group = self.norm_group.get()
        metric = self.norm_metric.get()
        if not group or not metric:
            messagebox.showwarning("Выбор", "Выберите группу и метрику.")
            return
        sub = self.patient_metrics_df[self.patient_metrics_df['severity'] == group]
        if sub.empty:
            self.log(f"Группа {group} не найдена.")
            return
        values = sub[metric].dropna().values
        if len(values) < 3:
            self.log(f"Недостаточно данных для группы {group} (n={len(values)}).")
            return
        if len(values) <= 5000:
            stat, p = shapiro(values)
            normal = p > 0.05
            self.normality_results[f"{group}_{metric}"] = {'p': p, 'n': len(values), 'normal': normal}
            res_text = f"Группа: {group}, метрика: {metric}\nШапиро-Уилк: W={stat:.4f}, p={p:.4e}\n"
            res_text += "Распределение нормальное" if normal else "Распределение не нормальное"
        else:
            self.normality_results[f"{group}_{metric}"] = {'p': None, 'n': len(values), 'normal': None}
            res_text = f"Группа: {group}, метрика: {metric}\nВыборка >5000, тест не применялся.\nОриентируйтесь на Q-Q plot."
        # Q-Q plot
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(f"Q-Q plot: {group} - {metric}")
        ax.grid(True)
        for widget in self.diag_plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.diag_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.diag_figure = fig
        self.diag_text.delete(1.0, tk.END)
        self.diag_text.insert(tk.END, res_text)
        self.notebook.select(self.tab_diag)
        self.log(f"Проверка нормальности для {group} {metric} завершена.")

    def run_bootstrap(self):
        if self.patient_metrics_df is None or self.patient_metrics_df.empty:
            messagebox.showwarning("Нет данных", "Сначала выполните анализ.")
            return
        group1 = self.boot_group1.get()
        group2 = self.boot_group2.get()
        metric = self.boot_metric.get()
        if not group1 or not group2 or not metric:
            messagebox.showwarning("Выбор", "Выберите две группы и метрику.")
            return
        if group1 == group2:
            messagebox.showwarning("Ошибка", "Группы должны различаться.")
            return
        sub1 = self.patient_metrics_df[self.patient_metrics_df['severity'] == group1]
        sub2 = self.patient_metrics_df[self.patient_metrics_df['severity'] == group2]
        if sub1.empty or sub2.empty:
            self.log(f"Одна из групп отсутствует: {group1}, {group2}")
            return
        vals1 = sub1[metric].dropna().values
        vals2 = sub2[metric].dropna().values
        if len(vals1) < 2 or len(vals2) < 2:
            self.log("Недостаточно пациентов в одной из групп для бутстрапа.")
            return
        n_iter = 1000
        diff_boot = []
        self.log(f"Бутстрап: {n_iter} итераций, сравнение {group1} vs {group2} по {metric}")
        for i in range(n_iter):
            if self.stop_flag:
                break
            boot1 = np.random.choice(vals1, size=len(vals1), replace=True)
            boot2 = np.random.choice(vals2, size=len(vals2), replace=True)
            diff = np.mean(boot1) - np.mean(boot2)
            diff_boot.append(diff)
        diff_orig = np.mean(vals1) - np.mean(vals2)
        ci_low = np.percentile(diff_boot, 2.5)
        ci_high = np.percentile(diff_boot, 97.5)
        p = (np.sum(np.abs(diff_boot) < 1e-8) * 2) / len(diff_boot)  # приближение двустороннего p
        significant = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
        key = f"{group1}_vs_{group2}"
        self.bootstrap_results[key] = self.bootstrap_results.get(key, {})
        self.bootstrap_results[key][metric] = {
            'diff': diff_orig,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p,
            'significant': significant,
            'n1': len(vals1),
            'n2': len(vals2)
        }
        self._update_bootstrap_table()
        # Гистограмма
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.hist(diff_boot, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=ci_low, color='red', linestyle='--', label=f'2.5%: {ci_low:.2f}')
        ax.axvline(x=ci_high, color='red', linestyle='--', label=f'97.5%: {ci_high:.2f}')
        ax.axvline(x=diff_orig, color='green', linestyle='-', label=f'Исходная разница: {diff_orig:.2f}')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel(f'Разница средних ({metric})')
        ax.set_ylabel('Частота')
        ax.set_title(f'Бутстрап распределение разницы: {group1} - {group2}')
        ax.legend()
        for widget in self.bootstrap_tree_frame.winfo_children():
            if widget not in (self.bootstrap_tree, self.bootstrap_tree_frame.children.get('!scrollbar2')):
                widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.bootstrap_tree_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.bootstrap_figure = fig
        self.notebook.select(self.tab_bootstrap)
        self.log(f"Бутстрап завершён. Разница средних: {diff_orig:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}], p≈{p:.4f}")

    def _update_bootstrap_table(self):
        for row in self.bootstrap_tree.get_children():
            self.bootstrap_tree.delete(row)
        for comp, metrics in self.bootstrap_results.items():
            for metric, res in metrics.items():
                self.bootstrap_tree.insert('', 'end', values=(
                    comp, metric,
                    f"{res['diff']:.4f}",
                    f"{res['ci_low']:.4f}",
                    f"{res['ci_high']:.4f}",
                    f"{res['p_value']:.4f}",
                    "Да" if res['significant'] else "Нет"
                ))

    def generate_report(self):
        if self.results_metrics is None or self.summary_curves is None:
            messagebox.showwarning("Нет данных", "Сначала выполните анализ.")
            return
        severity_labels = {
            'no_impairment': 'Норма', 'mild': 'Лёгкая', 'moderate': 'Умеренная', 'severe': 'Тяжёлая'
        }
        # Таблица метрик для отчёта (средние ± SEM)
        metrics_html = "<table border='1'><thead><tr><th>Группа</th><th>Пик γ (%)</th><th>Латентность (с)</th><th>AUC 0-10 с</th><th>AUC 10-30 с</th><th>n</th></tr></thead><tbody>"
        for _, row in self.results_metrics.iterrows():
            group = severity_labels.get(row['severity'], row['severity'])
            peak = f"{row['peak_amp_mean']:.2f} ± {row['peak_amp_sem']:.2f}"
            lat = f"{row['peak_lat_mean']:.2f} ± {row['peak_lat_sem']:.2f}"
            auc1 = f"{row['auc_0_10_mean']:.2f} ± {row['auc_0_10_sem']:.2f}"
            auc2 = f"{row['auc_10_30_mean']:.2f} ± {row['auc_10_30_sem']:.2f}"
            metrics_html += f"<tr><td>{group}</td><td>{peak}</td><td>{lat}</td><td>{auc1}</td><td>{auc2}</td><td>{row['n_patients']}</td></tr>"
        metrics_html += "</tbody></table>"

        # График
        buf = io.BytesIO()
        self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'

        # Нормальность
        norm_html = "<h3>Проверка нормальности метрик (Шапиро-Уилк)</h3>"
        if self.normality_results:
            norm_html += "<table border='1'><tr><th>Признак</th><th>p-value</th><th>n</th><th>Нормальное</th></tr>"
            for key, val in self.normality_results.items():
                p_str = f"{val['p']:.4e}" if val['p'] is not None else '>5000'
                norm_html += f"<tr><td>{key}</td><td>{p_str}</td><td>{val['n']}</td><td>{'Да' if val['normal'] else 'Нет'}</td></tr>"
            norm_html += "</table>"
        else:
            norm_html += "<p>Не выполнено.</p>"

        # Бутстрап
        boot_html = "<h3>Бутстрап сравнения групп (1000 итераций)</h3>"
        if self.bootstrap_results:
            boot_html += "<table border='1'><tr><th>Сравнение</th><th>Метрика</th><th>Разница</th><th>95% CI</th><th>p-value</th><th>Значимо</th></tr>"
            for comp, metrics in self.bootstrap_results.items():
                for metric, res in metrics.items():
                    boot_html += f"<tr><td>{comp}</td><td>{metric}</td><td>{res['diff']:.4f}</td><td>[{res['ci_low']:.4f}, {res['ci_high']:.4f}]</td><td>{res['p_value']:.4f}</td><td>{'Да' if res['significant'] else 'Нет'}</td></tr>"
            boot_html += "</table>"
        else:
            boot_html += "<p>Не выполнено.</p>"

        params = f"""
        <p><strong>Канал:</strong> {self.channel.get()}</p>
        <p><strong>Интервал времени:</strong> {self.time_min.get()} … {self.time_max.get()} с</p>
        <p><strong>Доверительный уровень:</strong> {self.confidence_level.get()}</p>
        <p><strong>Использованы фильтрованные исследования:</strong> {'Да' if self.use_filtered.get() else 'Нет'}</p>
        <p><strong>Метод:</strong> скользящие окна 5 с, шаг 1 с; нормализация относительно фона -60…-30 с.</p>
        """
        html = f"""<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>Event‑locked анализ γ-активности</title>
        <style>
            body {{ font-family: Arial; margin:20px; }}
            table {{ border-collapse: collapse; width:100%; margin-bottom:20px; }}
            th, td {{ border:1px solid #ddd; padding:8px; text-align:left; }}
            th {{ background:#f2f2f2; }}
            .plot {{ margin:20px 0; text-align:center; }}
        </style>
        </head>
        <body>
        <h1>Отчёт event‑locked анализа гамма‑активности (Глава 2, п. 2.4.3)</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {params}
        <h2>График динамики γ‑мощности</h2>
        {plot_html}
        <h2>Метрики по группам тяжести (mean ± SEM)</h2>
        {metrics_html}
        {norm_html}
        {boot_html}
        <p><em>Примечание:</em> AUC рассчитана методом трапеций. Для кривых показаны 95% ДИ (нормальное приближение). 
        Для робастных сравнений групп использован бутстрап.</p>
        </body></html>
        """
        fd, path = tempfile.mkstemp(suffix='.html', prefix='event_locked_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт в браузере: {path}")

    def save_results_csv(self):
        if self.results_metrics is None or self.results_metrics.empty:
            messagebox.showwarning("Нет данных", "Нет результатов для сохранения.")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.results_metrics.to_csv(file_path, index=False, encoding='utf-8-sig')
            self.log(f"Результаты сохранены в {file_path}")

    def save_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "График не построен.")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.current_figure.savefig(file_path, dpi=150, bbox_inches='tight')
            self.log(f"График сохранён в {file_path}")