# ui/tab_gam.py
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ui.base_tab import BaseTab
from core.api_client import get_event_time_series
from core.config import DEFAULT_API_URL, DEFAULT_TOKEN, OUTPUT_DIR

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    from rpy2.robjects import Formula
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    ro = None

class GAMTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        self.api_url = tk.StringVar(value=DEFAULT_API_URL)
        self.token = tk.StringVar(value=DEFAULT_TOKEN)
        self.channel = tk.StringVar(value='C3')
        self.time_min = tk.DoubleVar(value=-60.0)
        self.time_max = tk.DoubleVar(value=30.0)
        self.use_filtered = tk.BooleanVar(value=False)
        self.use_cache = tk.BooleanVar(value=True)   # ДОБАВЛЕНО
        self.stop_flag = False
        self.model = None
        self.gam_summary = None
        self.pred_df = None
        self.current_figure = None
        self._create_widgets()
        if not R_AVAILABLE:
            self.log("rpy2 не установлен. GAM-моделирование недоступно. Установите rpy2 и R с пакетом mgcv.")

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

        param_frame = ttk.LabelFrame(left_frame, text="Параметры модели", padding=5)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(param_frame, text="Канал:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(param_frame, textvariable=self.channel, values=['C3','C4'],
                     state='readonly').grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Интервал времени (сек):").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(param_frame, text="от").grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.time_min, width=8).grid(row=1, column=2, padx=2)
        ttk.Label(param_frame, text="до").grid(row=1, column=3, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.time_max, width=8).grid(row=1, column=4, padx=2)
        ttk.Checkbutton(param_frame, text="Использовать отфильтрованные исследования (из вкладки 'Загрузка')",
                        variable=self.use_filtered).grid(row=2, column=0, columnspan=5, sticky=tk.W, pady=5)

        cache_frame = ttk.LabelFrame(left_frame, text="Кэширование", padding=5)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cache_frame, text="Использовать кэш (ускоряет повторные запуски)",
                        variable=self.use_cache).pack(anchor=tk.W)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.run_btn = ttk.Button(btn_frame, text="Запустить GAM", command=self.run_gam)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить предсказания (CSV)",
                                       command=self.save_predictions_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график (PNG)",
                                        command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)
        self.save_summary_btn = ttk.Button(btn_frame, text="Сохранить summary (TXT)",
                                           command=self.save_summary_txt, state=tk.DISABLED)
        self.save_summary_btn.pack(side=tk.LEFT, padx=5)

        # ========== ПРАВАЯ ПАНЕЛЬ ==========
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_summary = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_summary, text="Summary модели")
        self.summary_text = tk.Text(self.tab_summary, wrap=tk.WORD, font=("Courier New", 10))
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        self.tab_plots = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plots, text="Графики")
        self.plot_frame = ttk.Frame(self.tab_plots)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

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

    def run_gam(self):
        if not R_AVAILABLE:
            messagebox.showerror("Ошибка", "rpy2 не установлен. Установите rpy2 и R с пакетом mgcv.")
            return
        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_csv_btn.config(state=tk.DISABLED)
        self.save_plot_btn.config(state=tk.DISABLED)
        self.save_summary_btn.config(state=tk.DISABLED)
        self.summary_text.delete(1.0, tk.END)
        self.log("Загрузка данных...")
        thread = threading.Thread(target=self._run_gam_thread)
        thread.daemon = True
        thread.start()

    def _run_gam_thread(self):
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
            df = df[['patient_id', 'event_duration', 'time_from_offset', 'gamma_power_norm_pct']].dropna()
            if df.empty:
                self.log("Нет данных после удаления пропусков.")
                return
            df['patient_id'] = df['patient_id'].astype('category')
            self.log(f"Данные: {len(df)} строк, пациентов: {df['patient_id'].nunique()}")

            pandas2ri.activate()
            mgcv = importr('mgcv')
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(df)
            formula = Formula("gamma_power_norm_pct ~ s(event_duration, bs='tp', k=8) + "
                              "s(time_from_offset, bs='tp', k=15) + "
                              "ti(event_duration, time_from_offset, bs='tp', k=5) + "
                              "s(patient_id, bs='re')")
            self.log("Оценка GAM-модели (REML)...")
            model = mgcv.gam(formula, data=r_df, method="REML")
            self.model = model
            summary_rs = ro.r.summary(model)
            summary_str = self._capture_r_output(ro.r.capture_output(ro.r.print(summary_rs)))
            self.summary_text.insert(tk.END, summary_str)
            self.log("Модель обучена.")
            pred = mgcv.predict_gam(model)
            self.pred_df = df.copy()
            with localconverter(ro.default_converter + pandas2ri.converter):
                self.pred_df['predicted'] = ro.conversion.rpy2py(pred)
            self.save_csv_btn.config(state=tk.NORMAL)
            self._plot_gam(model)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.save_summary_btn.config(state=tk.NORMAL)
            self.log("GAM анализ завершён.")
        except Exception as e:
            self.log(f"Ошибка: {e}")
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _capture_r_output(self, capture_obj):
        lines = [str(line) for line in capture_obj]
        return "\n".join(lines)

    def _plot_gam(self, model):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        import matplotlib.pyplot as plt
        from rpy2.robjects import r
        tmp_file = os.path.join(OUTPUT_DIR, 'temp_gam_plot.png')
        r('png')(filename=tmp_file, width=800, height=600)
        r('plot')(model, pages=1, scheme=2, seWithMean=True)
        r('dev.off')()
        img = plt.imread(tmp_file)
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    def save_predictions_csv(self):
        if self.pred_df is None:
            messagebox.showwarning("Нет данных", "Сначала выполните GAM.")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.pred_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            self.log(f"Предсказания сохранены в {file_path}")

    def save_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "График не построен.")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.current_figure.savefig(file_path, dpi=150, bbox_inches='tight')
            self.log(f"График сохранён в {file_path}")

    def save_summary_txt(self):
        if self.summary_text.get(1.0, tk.END).strip() == "":
            messagebox.showwarning("Нет данных", "Нет summary для сохранения.")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.summary_text.get(1.0, tk.END))
            self.log(f"Summary сохранён в {file_path}")