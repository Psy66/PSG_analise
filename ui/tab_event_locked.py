# ui/tab_event_locked.py
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
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
        self.use_cache = tk.BooleanVar(value=True)      # ДОБАВЛЕНО
        self.stop_flag = False
        self.results_df = None
        self.current_figure = None
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
        ttk.Combobox(param_frame, textvariable=self.channel, values=['C3','C4'],
                     state='readonly').grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Интервал времени (сек):").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(param_frame, text="от").grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.time_min, width=8).grid(row=1, column=2, padx=2)
        ttk.Label(param_frame, text="до").grid(row=1, column=3, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.time_max, width=8).grid(row=1, column=4, padx=2)
        ttk.Label(param_frame, text="Доверительный интервал:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.confidence_level, width=6).grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Checkbutton(param_frame, text="Использовать отфильтрованные исследования (из вкладки 'Загрузка')",
                        variable=self.use_filtered).grid(row=3, column=0, columnspan=5, sticky=tk.W, pady=5)

        # Кэширование
        cache_frame = ttk.LabelFrame(left_frame, text="Кэширование", padding=5)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cache_frame, text="Использовать кэш (ускоряет повторные запуски)",
                        variable=self.use_cache).pack(anchor=tk.W)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.run_btn = ttk.Button(btn_frame, text="Запустить анализ", command=self.run_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить результаты (CSV)",
                                       command=self.save_results_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график (PNG)",
                                        command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)

        # ========== ПРАВАЯ ПАНЕЛЬ ==========
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="График")
        self.plot_frame = ttk.Frame(self.tab_plot)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="Метрики")
        self.metrics_tree = ttk.Treeview(self.tab_metrics,
                                         columns=('severity','peak_amp','peak_latency','auc_0_10','auc_10_30'),
                                         show='headings')
        self.metrics_tree.heading('severity', text='Группа')
        self.metrics_tree.heading('peak_amp', text='Пик γ (%) 0-5 с')
        self.metrics_tree.heading('peak_latency', text='Латентность пика (с)')
        self.metrics_tree.heading('auc_0_10', text='AUC 0-10 с')
        self.metrics_tree.heading('auc_10_30', text='AUC 10-30 с')
        self.metrics_tree.pack(fill=tk.BOTH, expand=True)

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

            # Прогресс-колбэк для get_event_time_series
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

            # get_studies без кэша (небольшой объём)
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

            grouped_by_sev = {}
            for rec in patient_curves:
                sev = rec['severity']
                grouped_by_sev.setdefault(sev, []).append(rec['gamma'])

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
                f"{row['auc_10_30']:.2f}"
            ))

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