# ui/tab_gam.py
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import io
import base64
import webbrowser
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import shapiro, chi2
import statsmodels.api as sm

from ui.base_tab import BaseTab
from core.api_client import get_event_time_series
from core.config import DEFAULT_API_URL, DEFAULT_TOKEN, OUTPUT_DIR

# Попытка импорта rpy2 (необязательно, но без него GAM не работает)
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

# Константы для загрузки
GAM_PAGE_SIZE = 5000  # увеличенный размер страницы для ускорения


class GAMTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        # Параметры
        self.channel = tk.StringVar(value='C3')
        self.time_min = tk.DoubleVar(value=-60.0)
        self.time_max = tk.DoubleVar(value=30.0)
        self.use_filtered = tk.BooleanVar(value=True)
        self.use_cache = tk.BooleanVar(value=True)
        self.k_duration = tk.IntVar(value=8)   # число базисных функций для duration
        self.k_time = tk.IntVar(value=15)      # число базисных функций для time
        self.k_ti = tk.IntVar(value=5)         # число базисных функций для тензорного взаимодействия
        self.include_patient_re = tk.BooleanVar(value=True)  # случайный перехват для пациента
        
        self.stop_flag = False
        self.model = None          # R объект модели
        self.gam_summary = None    # текстовое представление summary
        self.pred_df = None        # предсказания для обучающих данных
        self.current_figure = None
        self.contour_figure = None
        self.partial_figure = None
        
        # Для диагностики и бутстрапа
        self.diagnostics = {}      # остатки, fitted, тесты
        self.bootstrap_results = None  # DataFrame с результатами бутстрапа
        self.all_diagnostics = []  # список для отчёта
        self.smooth_summary = None # таблица гладких членов
        
        self._create_widgets()
        if not R_AVAILABLE:
            self.log("ВНИМАНИЕ: rpy2 не установлен. GAM-моделирование недоступно.\n"
                     "Установите rpy2: pip install rpy2\n"
                     "Также требуется R с пакетом mgcv: install.packages('mgcv')")
                     
    def _create_widgets(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Прокрутка левой панели
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
        right_frame.config(width=600)
        
        # ========== ЛЕВАЯ ПАНЕЛЬ ==========
        info_frame = ttk.LabelFrame(left_frame, text="Источник данных", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, text="Используются отфильтрованные данные из вкладки 'Загрузка'",
                  foreground="blue").pack(anchor=tk.W)
        ttk.Label(info_frame, text="(токен и URL не требуются)").pack(anchor=tk.W)
        
        desc_frame = ttk.LabelFrame(left_frame, text="Что такое GAM? (Глава 2, п. 2.5.6)", padding=5)
        desc_frame.pack(fill=tk.X, padx=5, pady=5)
        desc_text = (
            "Генерализованные аддитивные модели (GAM) выявляют нелинейные зависимости между характеристиками "
            "респираторного события (длительность, время относительно offset) и нормализованной гамма‑мощностью. "
            "Модель включает гладкие члены s(duration), s(time), тензорное взаимодействие ti(duration, time) "
            "и случайный перехват для пациента. Оценка REML, проверка значимости нелинейности (p < 0.05, edf > 1.5)."
        )
        ttk.Label(desc_frame, text=desc_text, justify=tk.LEFT, wraplength=600).pack(anchor=tk.W, fill=tk.X, padx=5, pady=2)
        ttk.Button(desc_frame, text="Показать инструкцию", command=self.show_instructions).pack(anchor=tk.W, padx=5, pady=2)
        
        # Параметры модели
        param_frame = ttk.LabelFrame(left_frame, text="Параметры GAM", padding=5)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(param_frame, text="Канал:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(param_frame, textvariable=self.channel, values=['C3','C4'], state='readonly').grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Интервал времени (сек):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.time_min, width=8).grid(row=1, column=1, padx=2)
        ttk.Label(param_frame, text="до").grid(row=1, column=2, padx=2)
        ttk.Entry(param_frame, textvariable=self.time_max, width=8).grid(row=1, column=3, padx=2)
        ttk.Label(param_frame, text="k (duration):").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=4, to=20, textvariable=self.k_duration, width=5).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(param_frame, text="k (time):").grid(row=3, column=0, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=5, to=30, textvariable=self.k_time, width=5).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(param_frame, text="k (ti):").grid(row=4, column=0, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=3, to=15, textvariable=self.k_ti, width=5).grid(row=4, column=1, sticky=tk.W)
        ttk.Checkbutton(param_frame, text="Случайный перехват для пациента (re)", variable=self.include_patient_re).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Checkbutton(param_frame, text="Использовать отфильтрованные исследования", variable=self.use_filtered).grid(row=6, column=0, columnspan=2, sticky=tk.W)
        
        # Кэширование
        cache_frame = ttk.LabelFrame(left_frame, text="Кэширование", padding=5)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cache_frame, text="Использовать кэш (ускоряет повторные запуски)", variable=self.use_cache).pack(anchor=tk.W)
        ttk.Label(cache_frame, text="Страница загрузки: 5000 записей", foreground="gray").pack(anchor=tk.W)
        
        # Диагностика и бутстрап
        diag_frame = ttk.LabelFrame(left_frame, text="Диагностика и бутстрап", padding=5)
        diag_frame.pack(fill=tk.X, padx=5, pady=5)
        self.diag_btn = ttk.Button(diag_frame, text="Диагностика остатков", command=self.run_diagnostics, state=tk.DISABLED)
        self.diag_btn.pack(anchor=tk.W, pady=2)
        self.bootstrap_btn = ttk.Button(diag_frame, text="Бутстрап (100 итераций)", command=self.run_bootstrap, state=tk.DISABLED)
        self.bootstrap_btn.pack(anchor=tk.W, pady=2)
        self.save_diag_btn = ttk.Button(diag_frame, text="Сохранить диагностику (CSV)", command=self.save_diagnostics_csv, state=tk.DISABLED)
        self.save_diag_btn.pack(anchor=tk.W, pady=2)
        
        # Кнопки управления
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Запустить GAM", command=self.run_gam)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.save_pred_btn = ttk.Button(btn_frame, text="Сохранить предсказания (CSV)", command=self.save_predictions_csv, state=tk.DISABLED)
        self.save_pred_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график (PNG)", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать отчёт", command=self.generate_report, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=5)
        
        # ========== ПРАВАЯ ПАНЕЛЬ: вкладки ==========
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_summary = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_summary, text="Summary модели")
        self.summary_text = tk.Text(self.tab_summary, wrap=tk.WORD, font=("Courier New", 10))
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        self.tab_plots = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plots, text="Графики GAM")
        self.plot_frame = ttk.Frame(self.tab_plots)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tab_diag = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diag, text="Диагностика")
        self.diag_text = tk.Text(self.tab_diag, wrap=tk.WORD, font=("Courier New", 9), height=8)
        self.diag_text.pack(fill=tk.X, padx=5, pady=5)
        self.diag_plot_frame = ttk.Frame(self.tab_diag)
        self.diag_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tab_bootstrap = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bootstrap, text="Бутстрап")
        self.bootstrap_tree = ttk.Treeview(self.tab_bootstrap, columns=('параметр','beta','ci_low','ci_high','p_bootstrap'), show='headings')
        for col in ('параметр','beta','ci_low','ci_high','p_bootstrap'):
            self.bootstrap_tree.heading(col, text=col)
            self.bootstrap_tree.column(col, width=120)
        self.bootstrap_tree.pack(fill=tk.BOTH, expand=True)
        
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
            "ИНСТРУКЦИЯ ПО GAM АНАЛИЗУ (Глава 2, п. 2.5.6)\n"
            "============================================\n"
            "1. Убедитесь, что установлены R и пакет mgcv, а также rpy2.\n"
            "2. Загрузите и отфильтруйте данные на вкладке 'Загрузка'.\n"
            "3. Выберите канал (C3 или C4) и временной интервал.\n"
            "4. Нажмите 'Запустить GAM'. Модель оценивается методом REML.\n"
            "5. После завершения откроются summary модели и графики:\n"
            "   - Контурный график взаимодействия duration × time\n"
            "   - Графики частичных зависимостей для s(duration) и s(time)\n"
            "6. Выполните диагностику остатков (нормальность, гомоскедастичность).\n"
            "7. Запустите бутстрап (100 итераций) для робастных доверительных интервалов.\n"
            "8. Кнопка 'Сформировать отчёт' создаст HTML-отчёт с интерпретацией.\n"
            "9. Результаты можно сохранить в CSV (предсказания) и PNG (графики)."
        )
        messagebox.showinfo("Инструкция", msg)
        
    def run_gam(self):
        if not R_AVAILABLE:
            messagebox.showerror("Ошибка", "rpy2 не установлен или R недоступен.\nУстановите rpy2 и пакет mgcv.")
            return
        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_pred_btn.config(state=tk.DISABLED)
        self.save_plot_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        self.diag_btn.config(state=tk.DISABLED)
        self.bootstrap_btn.config(state=tk.DISABLED)
        self.save_diag_btn.config(state=tk.DISABLED)
        self.summary_text.delete(1.0, tk.END)
        self.log("Загрузка данных...")
        thread = threading.Thread(target=self._run_gam_thread)
        thread.daemon = True
        thread.start()
        
    def _run_gam_thread(self):
        try:
            load_tab = self.main_app.tabs['load']
            api_url = load_tab.api_url.get().rstrip('/')
            token = load_tab.token.get().strip()
            if not api_url or not token:
                self.log("Ошибка: не указаны URL или токен API.")
                return
                
            study_ids = None
            if self.use_filtered.get():
                filtered_df = self.main_app.get_filtered_data()
                if filtered_df is None or filtered_df.empty:
                    self.log("Нет отфильтрованных данных.")
                    return
                study_ids = filtered_df['study_id'].unique().tolist()
                self.log(f"Используем исследования: {study_ids}")
                
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
                page_size=GAM_PAGE_SIZE
            )
            self.main_app.set_progress(100)
            if not ts_data:
                self.log("Нет данных для выбранного канала.")
                return
            df = pd.DataFrame(ts_data)
            self.log(f"Загружено {len(df)} временных точек")
            # Агрегация: для каждого события и момента времени оставляем gamma_norm
            df = df[['patient_id', 'event_duration', 'time_from_offset', 'gamma_power_norm_pct']].dropna()
            if df.empty:
                self.log("Нет данных после удаления пропусков.")
                return
            df['patient_id'] = df['patient_id'].astype('category')
            self.log(f"Данные: {len(df)} строк, пациентов: {df['patient_id'].nunique()}")
            
            # Подготовка R
            pandas2ri.activate()
            mgcv = importr('mgcv')
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(df)
                
            # Формула модели
            formula_str = "gamma_power_norm_pct ~ s(event_duration, bs='tp', k={}) + s(time_from_offset, bs='tp', k={}) + ti(event_duration, time_from_offset, bs='tp', k={})".format(
                self.k_duration.get(), self.k_time.get(), self.k_ti.get())
            if self.include_patient_re.get():
                formula_str += " + s(patient_id, bs='re')"
            formula = Formula(formula_str)
            self.log(f"Оценка GAM-модели: {formula_str}")
            model = mgcv.gam(formula, data=r_df, method="REML")
            self.model = model
            
            # Получение summary
            summary_rs = ro.r.summary(model)
            summary_str = self._capture_r_output(ro.r.capture_output(ro.r.print(summary_rs)))
            self.summary_text.insert(tk.END, summary_str)
            
            # Извлечение таблицы гладких членов
            self._extract_smooth_summary(model)
            
            # Предсказания
            pred = mgcv.predict_gam(model)
            self.pred_df = df.copy()
            with localconverter(ro.default_converter + pandas2ri.converter):
                self.pred_df['predicted'] = ro.conversion.rpy2py(pred)
            self.save_pred_btn.config(state=tk.NORMAL)
            
            # Графики
            self._plot_gam(model)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.diag_btn.config(state=tk.NORMAL)
            self.bootstrap_btn.config(state=tk.NORMAL)
            self.log("GAM анализ завершён.")
            
        except Exception as e:
            self.log(f"Ошибка: {e}")
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
    def _capture_r_output(self, capture_obj):
        lines = [str(line) for line in capture_obj]
        return "\n".join(lines)
        
    def _extract_smooth_summary(self, model):
        """Извлекает информацию о гладких членах из R-модели."""
        try:
            summary_rs = ro.r.summary(model)
            # В R: summary_rs$s.table
            s_table = ro.r['$'](summary_rs, 's.table')
            if s_table is not None:
                df_smooth = pd.DataFrame({
                    'smooth_term': ['s(event_duration)', 's(time_from_offset)', 'ti(duration,time)'],
                    'edf': [s_table[0,0], s_table[1,0], s_table[2,0]],
                    'Ref.df': [s_table[0,1], s_table[1,1], s_table[2,1]],
                    'F': [s_table[0,2], s_table[1,2], s_table[2,2]],
                    'p_value': [s_table[0,3], s_table[1,3], s_table[2,3]]
                })
                self.smooth_summary = df_smooth
                self.log("Таблица гладких членов извлечена.")
        except Exception as e:
            self.log(f"Не удалось извлечь таблицу гладких членов: {e}")
            
    def _plot_gam(self, model):
        """Строит контурный график и графики частичных зависимостей, сохраняя в файлы."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Создаём временные PNG-файлы через R
        tmp_dir = tempfile.mkdtemp()
        contour_file = os.path.join(tmp_dir, 'contour.png')
        partial_file = os.path.join(tmp_dir, 'partial.png')
        
        try:
            # Контурный график (взаимодействие)
            ro.r('png')(filename=contour_file, width=800, height=600)
            ro.r('vis.gam')(model, view=ro.StrVector(['event_duration','time_from_offset']), 
                            theta=30, color='topo', plot_type='contour')
            ro.r('dev.off')()
            
            # Графики частичных зависимостей
            ro.r('png')(filename=partial_file, width=800, height=600)
            ro.r('par')(mfrow=ro.IntVector([2,2]))
            ro.r('plot')(model, page=1, scheme=2, seWithMean=True)
            ro.r('dev.off')()
            
            # Загрузка в matplotlib
            import matplotlib.pyplot as plt
            img_contour = plt.imread(contour_file)
            img_partial = plt.imread(partial_file)
            
            fig = Figure(figsize=(12, 10), dpi=100)
            ax1 = fig.add_subplot(2,1,1)
            ax1.imshow(img_contour)
            ax1.axis('off')
            ax1.set_title('Contour plot: interaction duration × time')
            ax2 = fig.add_subplot(2,1,2)
            ax2.imshow(img_partial)
            ax2.axis('off')
            ax2.set_title('Partial effects (s(duration), s(time), ti)')
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.current_figure = fig
            
        except Exception as e:
            self.log(f"Ошибка при построении графиков GAM: {e}")
            label = ttk.Label(self.plot_frame, text="Не удалось построить графики.\nПроверьте установку R и mgcv.")
            label.pack()
        finally:
            # Очистка временной директории
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            
    def run_diagnostics(self):
        if self.model is None or self.pred_df is None:
            messagebox.showwarning("Нет модели", "Сначала выполните GAM.")
            return
        try:
            # Остатки и предсказанные значения
            fitted = self.pred_df['predicted'].values
            observed = self.pred_df['gamma_power_norm_pct'].values
            resid = observed - fitted
            n = len(resid)
            # Тест Шапиро-Уилка
            shapiro_p = None
            if n <= 5000:
                _, shapiro_p = shapiro(resid)
            # Тест Бройша-Пагана
            resid2 = resid ** 2
            X = sm.add_constant(fitted)
            bp_model = sm.OLS(resid2, X).fit()
            bp_stat = bp_model.rsquared * n
            bp_p = 1 - chi2.cdf(bp_stat, df=1)
            
            self.diagnostics = {
                'fitted': fitted,
                'residuals': resid,
                'shapiro_p': shapiro_p,
                'bp_p': bp_p,
                'n': n
            }
            # Графики
            fig = Figure(figsize=(10, 8))
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
            ax4.text(0.1, 0.9, f"n = {n}", fontsize=10)
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
            
            for widget in self.diag_plot_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.diag_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.diag_text.delete(1.0, tk.END)
            self.diag_text.insert(tk.END, f"Диагностика остатков GAM:\nShapiro-Wilk p={shapiro_p:.4e}\nBreusch-Pagan p={bp_p:.4f}\n")
            self.notebook.select(self.tab_diag)
            self.save_diag_btn.config(state=tk.NORMAL)
            # Сохраняем для отчёта
            self.all_diagnostics.append({'shapiro_p': shapiro_p, 'bp_p': bp_p, 'n': n})
            self.log("Диагностика завершена.")
        except Exception as e:
            self.log(f"Ошибка диагностики: {e}")
            
    def save_diagnostics_csv(self):
        if not self.diagnostics:
            messagebox.showwarning("Нет данных", "Сначала выполните диагностику.")
            return
        df = pd.DataFrame({'fitted': self.diagnostics['fitted'], 'residuals': self.diagnostics['residuals']})
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            df.to_csv(path, index=False)
            self.log(f"Диагностика сохранена в {path}")
            
    def run_bootstrap(self):
        if self.model is None:
            messagebox.showwarning("Нет модели", "Сначала выполните GAM.")
            return
        # Упрощённый бутстрап: ресэмплируем наблюдения с возвращением,
        # переоцениваем модель (только фиксированные гладкие члены, без RE для скорости)
        # и сохраняем коэффициенты (edf для гладких членов)
        self.log("Запуск бутстрапа (100 итераций)...")
        n_iter = 100
        results = []
        data = self.pred_df[['event_duration', 'time_from_offset', 'gamma_power_norm_pct']].dropna()
        if len(data) < 100:
            self.log("Недостаточно данных для бутстрапа.")
            return
        mgcv = importr('mgcv')
        for i in range(n_iter):
            if self.stop_flag:
                break
            boot_idx = np.random.choice(len(data), size=len(data), replace=True)
            boot_data = data.iloc[boot_idx].copy()
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_boot = ro.conversion.py2rpy(boot_data)
            formula_str = "gamma_power_norm_pct ~ s(event_duration, bs='tp', k={}) + s(time_from_offset, bs='tp', k={}) + ti(event_duration, time_from_offset, bs='tp', k={})".format(
                self.k_duration.get(), self.k_time.get(), self.k_ti.get())
            try:
                model_boot = mgcv.gam(Formula(formula_str), data=r_boot, method="REML")
                summary_rs = ro.r.summary(model_boot)
                s_table = ro.r['$'](summary_rs, 's.table')
                if s_table is not None and s_table.nrow >= 3:
                    results.append({
                        'edf_dur': s_table[0,0],
                        'edf_time': s_table[1,0],
                        'edf_ti': s_table[2,0],
                        'p_dur': s_table[0,3],
                        'p_time': s_table[1,3],
                        'p_ti': s_table[2,3]
                    })
            except:
                pass
            if (i+1) % 20 == 0:
                self.log(f"Бутстрап: {i+1}/{n_iter} итераций")
        if results:
            df_res = pd.DataFrame(results)
            ci = {}
            for param in ['edf_dur', 'edf_time', 'edf_ti', 'p_dur', 'p_time', 'p_ti']:
                ci[param] = (np.percentile(df_res[param], 2.5), np.percentile(df_res[param], 97.5))
            self.bootstrap_results = ci
            # Отобразить в таблице
            for row in self.bootstrap_tree.get_children():
                self.bootstrap_tree.delete(row)
            for param, (low, high) in ci.items():
                self.bootstrap_tree.insert('', 'end', values=(param, f"{np.mean(df_res[param]):.4f}", f"{low:.4f}", f"{high:.4f}", ""))
            self.notebook.select(self.tab_bootstrap)
            self.log("Бутстрап завершён.")
        else:
            self.log("Бутстрап не дал результатов.")
            
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
            
    def generate_report(self):
        if self.model is None:
            messagebox.showwarning("Нет модели", "Сначала выполните GAM.")
            return
        # Собираем отчёт
        # Таблица гладких членов
        smooth_html = ""
        if self.smooth_summary is not None:
            smooth_html = self.smooth_summary.to_html(index=False, float_format="%.4f")
        else:
            smooth_html = "<p>Не удалось извлечь таблицу гладких членов.</p>"
            
        # Графики в base64
        buf = io.BytesIO()
        self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
        
        # Диагностика
        diag_html = "<h3>Диагностика остатков</h3>"
        if self.all_diagnostics:
            d = self.all_diagnostics[-1]
            diag_html += f"<p>Shapiro-Wilk p = {d['shapiro_p']:.4e}<br>Breusch-Pagan p = {d['bp_p']:.4f}<br>n = {d['n']}</p>"
        else:
            diag_html += "<p>Диагностика не выполнялась.</p>"
            
        # Бутстрап
        boot_html = "<h3>Бутстрап доверительные интервалы (95%)</h3>"
        if self.bootstrap_results:
            boot_html += "<table border='1'><tr><th>Параметр</th><th>2.5%</th><th>97.5%</th></tr>"
            for param, (low, high) in self.bootstrap_results.items():
                boot_html += f"<tr><td>{param}</td><td>{low:.4f}</td><td>{high:.4f}</td></tr>"
            boot_html += "</table>"
        else:
            boot_html += "<p>Бутстрап не выполнялся.</p>"
            
        params = f"""
        <p><strong>Канал:</strong> {self.channel.get()}</p>
        <p><strong>Интервал времени:</strong> {self.time_min.get()} … {self.time_max.get()} сек</p>
        <p><strong>k (duration):</strong> {self.k_duration.get()}, <strong>k (time):</strong> {self.k_time.get()}, <strong>k (ti):</strong> {self.k_ti.get()}</p>
        <p><strong>Случайный перехват пациента:</strong> {'Да' if self.include_patient_re.get() else 'Нет'}</p>
        """
        
        html = f"""<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>GAM анализ γ-активности</title>
        <style>
            body {{ font-family: Arial; margin:20px; }}
            table {{ border-collapse: collapse; width:100%; margin-bottom:20px; }}
            th, td {{ border:1px solid #ddd; padding:8px; text-align:left; }}
            th {{ background:#f2f2f2; }}
            .plot {{ margin:20px 0; text-align:center; }}
        </style>
        </head>
        <body>
        <h1>Отчёт о генерализованной аддитивной модели (GAM) – Глава 2, п. 2.5.6</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {params}
        <h2>Графики GAM</h2>
        {plot_html}
        <h2>Гладкие члены модели (summary)</h2>
        {smooth_html}
        {diag_html}
        {boot_html}
        <p><em>Интерпретация:</em> Значимые нелинейные эффекты (p < 0.05) и эффективные степени свободы (edf > 1.5) указывают на наличие нелинейной зависимости. Тензорное взаимодействие показывает, как длительность события и время относительно offset совместно влияют на γ-мощность.</p>
        </body></html>
        """
        fd, path = tempfile.mkstemp(suffix='.html', prefix='gam_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт в браузере: {path}")        self.use_cache = tk.BooleanVar(value=True)   # ДОБАВЛЕНО
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
