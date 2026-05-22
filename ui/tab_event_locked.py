# ui/tab_event_locked.py
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
from scipy.stats import false_discovery_control, shapiro, chi2
import statsmodels.formula.api as smf
from ui.base_tab import BaseTab
from core.api_client import get_event_time_series, get_studies
from core.config import DEFAULT_API_URL, DEFAULT_TOKEN

class EventLockedTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        self.api_url = tk.StringVar(value=DEFAULT_API_URL)
        self.token = tk.StringVar(value=DEFAULT_TOKEN)
        self.channel = tk.StringVar(value='C3')
        self.time_min = tk.DoubleVar(value=-60.0)
        self.time_max = tk.DoubleVar(value=30.0)
        self.use_filtered = tk.BooleanVar(value=False)
        self.confidence_level = tk.DoubleVar(value=0.95)
        self.use_cache = tk.BooleanVar(value=True)
        self.stop_flag = False
        self.results_df = None          # метрики по группам
        self.summary = None             # список dict для кривых (mean, ci, time)
        self.common_time = None
        self.current_figure = None
        self.patient_curves = None      # сырые кривые для бутстрапа
        self.grouped_by_sev = None      # dict {severity: list of arrays}
        
        # Для диагностики и бутстрапа
        self.normality_results = {}     # {metric_group: shapiro p}
        self.bootstrap_results = {}     # {comparison: {metric: [ci_low, ci_high, p]}}
        self.diag_model = None
        self.diag_figure = None
        self.bootstrap_figure = None
        
        self._create_widgets()
        
    def _create_widgets(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(main_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        paned = ttk.PanedWindow(inner, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        # ========== ЛЕВАЯ ПАНЕЛЬ ==========
        api_frame = ttk.LabelFrame(left_frame, text="Подключение к API", padding=5)
        api_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(api_frame, text="URL:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(api_frame, textvariable=self.api_url, width=40).grid(row=0, column=1, padx=5)
        ttk.Label(api_frame, text="Токен:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(api_frame, textvariable=self.token, width=40, show="*").grid(row=1, column=1, padx=5)
        
        param_frame = ttk.LabelFrame(left_frame, text="Параметры", padding=5)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(param_frame, text="Канал:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(param_frame, textvariable=self.channel, values=['C3','C4'], state='readonly').grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Интервал времени (сек):").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(param_frame, text="от").grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.time_min, width=8).grid(row=1, column=2, padx=2)
        ttk.Label(param_frame, text="до").grid(row=1, column=3, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.time_max, width=8).grid(row=1, column=4, padx=2)
        ttk.Label(param_frame, text="Доверительный интервал:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.confidence_level, width=6).grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Checkbutton(param_frame, text="Использовать отфильтрованные исследования (из вкладки 'Загрузка')",
                        variable=self.use_filtered).grid(row=3, column=0, columnspan=5, sticky=tk.W, pady=5)
        
        cache_frame = ttk.LabelFrame(left_frame, text="Кэширование", padding=5)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cache_frame, text="Использовать кэш (ускоряет повторные запуски)",
                        variable=self.use_cache).pack(anchor=tk.W)
        
        # ---- Новые разделы: диагностика и бутстрап ----
        norm_frame = ttk.LabelFrame(left_frame, text="Проверка нормальности метрик", padding=5)
        norm_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(norm_frame, text="Группа:").grid(row=0, column=0, sticky=tk.W)
        self.norm_group = tk.StringVar()
        self.norm_group_combo = ttk.Combobox(norm_frame, textvariable=self.norm_group, state='readonly', width=15)
        self.norm_group_combo.grid(row=0, column=1, padx=5)
        ttk.Label(norm_frame, text="Метрика:").grid(row=1, column=0, sticky=tk.W)
        self.norm_metric = tk.StringVar()
        self.norm_metric_combo = ttk.Combobox(norm_frame, textvariable=self.norm_metric, state='readonly',
                                              values=['peak_amplitude','peak_latency','auc_0_10','auc_10_30'], width=15)
        self.norm_metric_combo.grid(row=1, column=1, padx=5)
        self.norm_btn = ttk.Button(norm_frame, text="Проверить нормальность", command=self.check_normality_metrics, state=tk.DISABLED)
        self.norm_btn.grid(row=2, column=0, columnspan=2, pady=2)
        
        bootstrap_frame = ttk.LabelFrame(left_frame, text="Бутстрап сравнения групп", padding=5)
        bootstrap_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(bootstrap_frame, text="Группа 1:").grid(row=0, column=0, sticky=tk.W)
        self.boot_group1 = tk.StringVar()
        self.boot_group1_combo = ttk.Combobox(bootstrap_frame, textvariable=self.boot_group1, state='readonly', width=15)
        self.boot_group1_combo.grid(row=0, column=1, padx=5)
        ttk.Label(bootstrap_frame, text="Группа 2:").grid(row=1, column=0, sticky=tk.W)
        self.boot_group2 = tk.StringVar()
        self.boot_group2_combo = ttk.Combobox(bootstrap_frame, textvariable=self.boot_group2, state='readonly', width=15)
        self.boot_group2_combo.grid(row=1, column=1, padx=5)
        ttk.Label(bootstrap_frame, text="Метрика:").grid(row=2, column=0, sticky=tk.W)
        self.boot_metric = tk.StringVar()
        self.boot_metric_combo = ttk.Combobox(bootstrap_frame, textvariable=self.boot_metric, state='readonly',
                                              values=['peak_amplitude','peak_latency','auc_0_10','auc_10_30'], width=15)
        self.boot_metric_combo.grid(row=2, column=1, padx=5)
        self.bootstrap_btn = ttk.Button(bootstrap_frame, text="Запустить бутстрап (1000 итераций)", command=self.run_bootstrap, state=tk.DISABLED)
        self.bootstrap_btn.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Кнопки управления
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.run_btn = ttk.Button(btn_frame, text="Запустить анализ", command=self.run_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить метрики (CSV)", command=self.save_results_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график (PNG)", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать отчёт", command=self.generate_report, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=5)
        
        # ========== ПРАВАЯ ПАНЕЛЬ: ноутбук с вкладками ==========
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="График")
        self.plot_frame = ttk.Frame(self.tab_plot)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="Метрики")
        self.metrics_tree = ttk.Treeview(self.tab_metrics,
                                         columns=('severity','peak_amp','peak_latency','auc_0_10','auc_10_30','n_patients'),
                                         show='headings')
        self.metrics_tree.heading('severity', text='Группа')
        self.metrics_tree.heading('peak_amp', text='Пик γ (%)')
        self.metrics_tree.heading('peak_latency', text='Латентность пика (с)')
        self.metrics_tree.heading('auc_0_10', text='AUC 0-10 с')
        self.metrics_tree.heading('auc_10_30', text='AUC 10-30 с')
        self.metrics_tree.heading('n_patients', text='n пациентов')
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
                                           columns=('comparison','metric','beta','ci_low','ci_high','p_value','significant'),
                                           show='headings')
        for col in ('comparison','metric','beta','ci_low','ci_high','p_value','significant'):
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
            study_ids = None
            if self.use_filtered.get():
                filtered_df = self.main_app.get_filtered_data()
                if filtered_df is None or filtered_df.empty:
                    self.log("Нет отфильтрованных данных.")
                    return
                study_ids = filtered_df['study_id'].unique().tolist()
                if not study_ids:
                    self.log("В отфильтрованных данных нет study_id.")
                    return
                self.log(f"Используем исследования: {study_ids}")
                
            def update_progress(page, total, _):
                if total > 0:
                    self.main_app.set_progress(int(page / total * 100))
                    
            self.main_app.set_progress(0)
            ts_data = get_event_time_series(
                self.api_url.get(), self.token.get(),
                study_ids=study_ids,
                channel=self.channel.get(),
                time_from_offset_min=self.time_min.get(),
                time_from_offset_max=self.time_max.get(),
                stop_check=lambda: self.stop_flag,
                progress_callback=update_progress,
                use_cache=self.use_cache.get()
            )
            self.main_app.set_progress(100)
            if not ts_data:
                self.log("Нет данных для выбранного канала и интервала.")
                return
            df = pd.DataFrame(ts_data)
            self.log(f"Загружено {len(df)} временных точек")
            
            studies = get_studies(self.api_url.get(), self.token.get(), stop_check=lambda: self.stop_flag)
            study_severity = {s['study_id']: s.get('breathing_impairment_severity', 'unknown') for s in studies}
            df['severity'] = df['study_id'].map(study_severity)
            df = df.dropna(subset=['severity'])
            if df.empty:
                self.log("Нет событий с известной тяжестью ОАС.")
                return
            self.log(f"Группы: {df['severity'].unique()}")
            
            common_time = np.arange(self.time_min.get(), self.time_max.get() + 0.5, 0.5)
            patient_curves = []
            for (pid, sev), group in df.groupby(['patient_id', 'severity']):
                group = group.sort_values('time_from_offset')
                group = group.drop_duplicates(subset=['time_from_offset'])
                if len(group) < 2:
                    continue
                f = interp1d(group['time_from_offset'], group['gamma_power_norm_pct'],
                             kind='linear', fill_value='extrapolate')
                y_interp = f(common_time)
                patient_curves.append({
                    'patient_id': pid,
                    'severity': sev,
                    'time': common_time,
                    'gamma': y_interp
                })
            if not patient_curves:
                self.log("Не удалось построить кривые для пациентов.")
                return
            self.patient_curves = patient_curves
            grouped_by_sev = {}
            for rec in patient_curves:
                sev = rec['severity']
                grouped_by_sev.setdefault(sev, []).append(rec['gamma'])
            self.grouped_by_sev = grouped_by_sev
            
            ci_z = stats.norm.ppf(1 - (1 - self.confidence_level.get()) / 2)
            summary = []
            for sev, curves in grouped_by_sev.items():
                curves = np.array(curves)
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0, ddof=1)
                n_patients = len(curves)
                ci = ci_z * std_curve / np.sqrt(n_patients)
                summary.append({
                    'severity': sev,
                    'time': common_time,
                    'mean': mean_curve,
                    'ci': ci,
                    'n': n_patients
                })
            self.summary = summary
            self.common_time = common_time
            self._plot_curves(summary, common_time)
            
            metrics = []
            for rec in summary:
                sev = rec['severity']
                t = rec['time']
                y = rec['mean']
                mask_peak = (t >= 0) & (t <= 5)
                if np.any(mask_peak):
                    peak_amp = np.max(y[mask_peak])
                    peak_latency = t[mask_peak][np.argmax(y[mask_peak])]
                else:
                    peak_amp = np.nan
                    peak_latency = np.nan
                mask_auc1 = (t >= 0) & (t <= 10)
                auc_0_10 = trapezoid(y[mask_auc1], t[mask_auc1]) if np.any(mask_auc1) else np.nan
                mask_auc2 = (t >= 10) & (t <= 30)
                auc_10_30 = trapezoid(y[mask_auc2], t[mask_auc2]) if np.any(mask_auc2) else np.nan
                metrics.append({
                    'severity': sev,
                    'peak_amplitude': peak_amp,
                    'peak_latency': peak_latency,
                    'auc_0_10': auc_0_10,
                    'auc_10_30': auc_10_30,
                    'n_patients': rec['n']
                })
            self.results_df = pd.DataFrame(metrics)
            self._display_metrics_table(self.results_df)
            self.save_csv_btn.config(state=tk.NORMAL)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.norm_btn.config(state=tk.NORMAL)
            self.bootstrap_btn.config(state=tk.NORMAL)
            # заполним комбобоксы
            groups = sorted(self.results_df['severity'].unique())
            self.norm_group_combo['values'] = groups
            self.boot_group1_combo['values'] = groups
            self.boot_group2_combo['values'] = groups
            if groups:
                self.norm_group.set(groups[0])
                self.boot_group1.set(groups[0])
                self.boot_group2.set(groups[1] if len(groups)>1 else groups[0])
            self.log("Event‑locked анализ завершён.")
            
        except Exception as e:
            self.log(f"Ошибка: {e}")
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
    def _plot_curves(self, summary, common_time):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        severity_order = ['no_impairment', 'mild', 'moderate', 'severe', 'central', 'mixed']
        severity_labels = {
            'no_impairment': 'Норма',
            'mild': 'Лёгкая',
            'moderate': 'Умеренная',
            'severe': 'Тяжёлая',
            'central': 'Центральное',
            'mixed': 'Смешанное'
        }
        for sev in severity_order:
            rec = next((r for r in summary if r['severity'] == sev), None)
            if rec is not None:
                label = severity_labels.get(sev, sev)
                ax.plot(common_time, rec['mean'], label=label, linewidth=2)
                ax.fill_between(common_time, rec['mean'] - rec['ci'], rec['mean'] + rec['ci'], alpha=0.2)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='offset')
        ax.axvspan(-60, -30, alpha=0.1, color='gray', label='фон (нормировка)')
        ax.set_xlabel('Время относительно окончания события (сек)')
        ax.set_ylabel('Гамма-мощность, % от фона')
        ax.set_title('Динамика гамма-активности вокруг респираторных событий')
        ax.legend()
        ax.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig
        
    def _display_metrics_table(self, df):
        for row in self.metrics_tree.get_children():
            self.metrics_tree.delete(row)
        for _, row in df.iterrows():
            self.metrics_tree.insert('', 'end', values=(
                row['severity'],
                f"{row['peak_amplitude']:.2f}",
                f"{row['peak_latency']:.2f}",
                f"{row['auc_0_10']:.2f}",
                f"{row['auc_10_30']:.2f}",
                row['n_patients']
            ))
            
    # ------------------------------------------------------------
    # Проверка нормальности метрик
    # ------------------------------------------------------------
    def check_normality_metrics(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Сначала выполните анализ.")
            return
        group = self.norm_group.get()
        metric = self.norm_metric.get()
        if not group or not metric:
            messagebox.showwarning("Выбор", "Выберите группу и метрику.")
            return
        # Получаем значения метрики для этой группы из сырых кривых пациентов
        # Для этого нужно иметь доступ к per‑patient значениям
        # Соберём per‑patient метрики
        patient_metrics = self._compute_patient_metrics()
        if patient_metrics is None:
            self.log("Не удалось собрать per‑patient метрики.")
            return
        if group not in patient_metrics:
            self.log(f"Группа {group} не найдена.")
            return
        values = patient_metrics[group][metric].dropna().values
        if len(values) < 3:
            self.log(f"Недостаточно данных для группы {group} (n={len(values)}).")
            return
        # Тест Шапиро-Уилка
        if len(values) <= 5000:
            stat, p = shapiro(values)
            normal = p > 0.05
            self.normality_results[f"{group}_{metric}"] = {'p': p, 'n': len(values), 'normal': normal}
            res_text = f"Группа: {group}, метрика: {metric}\n"
            res_text += f"Шапиро-Уилк: W={stat:.4f}, p={p:.4e}\n"
            res_text += "Распределение нормальное" if normal else "Распределение не нормальное"
        else:
            self.normality_results[f"{group}_{metric}"] = {'p': None, 'n': len(values), 'normal': None}
            res_text = f"Группа: {group}, метрика: {metric}\nВыборка >5000, тест не применялся.\nОриентируйтесь на Q-Q plot."
        # Q-Q plot
        fig = Figure(figsize=(6,5))
        ax = fig.add_subplot(111)
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(f"Q-Q plot: {group} - {metric}")
        ax.grid(True)
        # Отображаем в правой панели диагностики
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
        
    def _compute_patient_metrics(self):
        """Возвращает dict: severity -> DataFrame с per-patient метриками"""
        if self.patient_curves is None or self.common_time is None:
            return None
        records = []
        t = self.common_time
        for rec in self.patient_curves:
            sev = rec['severity']
            y = rec['gamma']
            mask_peak = (t >= 0) & (t <= 5)
            if np.any(mask_peak):
                peak_amp = np.max(y[mask_peak])
                peak_lat = t[mask_peak][np.argmax(y[mask_peak])]
            else:
                peak_amp = np.nan
                peak_lat = np.nan
            mask_auc1 = (t >= 0) & (t <= 10)
            auc1 = trapezoid(y[mask_auc1], t[mask_auc1]) if np.any(mask_auc1) else np.nan
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
        df = pd.DataFrame(records)
        result = {}
        for sev, sub in df.groupby('severity'):
            result[sev] = sub[['peak_amplitude','peak_latency','auc_0_10','auc_10_30']].copy()
        return result
        
    # ------------------------------------------------------------
    # Блок-бутстрап для сравнения групп
    # ------------------------------------------------------------
    def run_bootstrap(self):
        if self.patient_curves is None:
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
        # Получим per-patient метрики
        patient_metrics = self._compute_patient_metrics()
        if patient_metrics is None:
            return
        if group1 not in patient_metrics or group2 not in patient_metrics:
            self.log(f"Одна из групп отсутствует: {group1}, {group2}")
            return
        vals1 = patient_metrics[group1][metric].dropna().values
        vals2 = patient_metrics[group2][metric].dropna().values
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
        p = (np.sum(np.abs(diff_boot) < 1e-8) * 2) / len(diff_boot)
        significant = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
        # Сохраним результат
        key = f"{group1}_vs_{group2}"
        self.bootstrap_results[key] = self.bootstrap_results.get(key, {})
        self.bootstrap_results[key][metric] = {
            'beta': diff_orig,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p,
            'significant': significant,
            'n1': len(vals1),
            'n2': len(vals2)
        }
        self._update_bootstrap_table()
        # Построим гистограмму
        fig = Figure(figsize=(8,5))
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
        # Показать на вкладке бутстрап
        for widget in self.bootstrap_tree_frame.winfo_children():
            if widget != self.bootstrap_tree and widget != sb:
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
                    f"{res['beta']:.4f}",
                    f"{res['ci_low']:.4f}",
                    f"{res['ci_high']:.4f}",
                    f"{res['p_value']:.4f}",
                    "Да" if res['significant'] else "Нет"
                ))
                
    # ------------------------------------------------------------
    # Генерация HTML-отчёта
    # ------------------------------------------------------------
    def generate_report(self):
        if self.results_df is None or self.summary is None:
            messagebox.showwarning("Нет данных", "Сначала выполните анализ.")
            return
        # Подготовим данные
        severity_labels = {
            'no_impairment': 'Норма', 'mild': 'Лёгкая', 'moderate': 'Умеренная',
            'severe': 'Тяжёлая', 'central': 'Центральное', 'mixed': 'Смешанное'
        }
        # Таблица метрик
        metrics_html = self.results_df.to_html(index=False, float_format="%.2f")
        # График в base64
        buf = io.BytesIO()
        self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
        # Нормальность
        norm_html = "<h3>Проверка нормальности метрик</h3>"
        if self.normality_results:
            norm_html += "<table border='1'><tr><th>Признак</th><th>p-value</th><th>n</th><th>Нормальное</th></tr>"
            for key, val in self.normality_results.items():
                norm_html += f"<tr><td>{key}</td><td>{val['p']:.4e if val['p'] else '>5000'}</td><td>{val['n']}</td><td>{'Да' if val['normal'] else 'Нет'}</td></tr>"
            norm_html += "</table>"
        else:
            norm_html += "<p>Не выполнено.</p>"
        # Бутстрап
        boot_html = "<h3>Бутстрап сравнения групп</h3>"
        if self.bootstrap_results:
            boot_html += "<table border='1'><tr><th>Сравнение</th><th>Метрика</th><th>Разница</th><th>95% CI</th><th>p-value</th><th>Значимо</th></tr>"
            for comp, metrics in self.bootstrap_results.items():
                for metric, res in metrics.items():
                    boot_html += f"<tr><td>{comp}</td><td>{metric}</td><td>{res['beta']:.4f}</td><td>[{res['ci_low']:.4f}, {res['ci_high']:.4f}]</td><td>{res['p_value']:.4f}</td><td>{'Да' if res['significant'] else 'Нет'}</td></tr>"
            boot_html += "</table>"
        else:
            boot_html += "<p>Не выполнено.</p>"
        # Параметры
        params = f"""
        <p><strong>Канал:</strong> {self.channel.get()}</p>
        <p><strong>Интервал времени:</strong> {self.time_min.get()} … {self.time_max.get()} сек</p>
        <p><strong>Доверительный уровень:</strong> {self.confidence_level.get()}</p>
        <p><strong>Использованы фильтрованные исследования:</strong> {'Да' if self.use_filtered.get() else 'Нет'}</p>
        """
        html = f"""<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>Event‑locked анализ</title>
        <style>
            body {{ font-family: Arial; margin:20px; }}
            table {{ border-collapse: collapse; width:100%; margin-bottom:20px; }}
            th, td {{ border:1px solid #ddd; padding:8px; text-align:left; }}
            th {{ background:#f2f2f2; }}
            .plot {{ margin:20px 0; text-align:center; }}
        </style>
        </head>
        <body>
        <h1>Отчёт event‑locked анализа гамма‑активности</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {params}
        <h2>График динамики γ‑мощности</h2>
        {plot_html}
        <h2>Метрики по группам тяжести</h2>
        {metrics_html}
        {norm_html}
        {boot_html}
        <p><em>Примечание:</em> AUC рассчитана методом трапеций, доверительные интервалы для кривых – нормальное приближение. Для робастных сравнений рекомендуется бутстрап.</p>
        </body></html>
        """
        fd, path = tempfile.mkstemp(suffix='.html', prefix='event_locked_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт в браузере: {path}")
        
    # ------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------
    def save_results_csv(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Нет результатов для сохранения.")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.results_df.to_csv(file_path, index=False, encoding='utf-8-sig')
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
