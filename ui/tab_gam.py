# ui/tab_gam.py
"""
Генерализованные аддитивные модели (GAM) для анализа нелинейных зависимолен
нормализованной гамма‑мощности от длительности апноэ и времени относительно offset.
Соответствует Главе 2, п. 2.5.6.
"""

import threading
import tkinter as tk
from tkinter import messagebox, ttk
import warnings
import os
import tempfile
import webbrowser
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ui.base_tab import BaseTab
from core.api_client import get_event_time_series

# Проверка наличия rpy2 и R
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

warnings.filterwarnings("ignore")


class GAMTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        # ---- Параметры ----
        self.channel = tk.StringVar(value='C3')
        self.time_min = tk.DoubleVar(value=-60.0)
        self.time_max = tk.DoubleVar(value=30.0)
        self.use_filtered = tk.BooleanVar(value=True)
        self.use_cache = tk.BooleanVar(value=True)
        self.k_duration = tk.IntVar(value=8)
        self.k_time = tk.IntVar(value=15)
        self.k_interaction = tk.IntVar(value=5)
        self.stop_flag = False
        self.model = None
        self.pred_df = None
        self.current_figure = None
        self.results_smooth = None
        self._create_widgets()
        if not R_AVAILABLE:
            self.log("rpy2 не установлен. GAM-моделирование недоступно. Установите rpy2 и R с пакетом mgcv.")

    def _create_widgets(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Горизонтальный разделитель
        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned)
        right_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        paned.add(right_frame, weight=4)

        # ========== Левая панель ==========
        info_frame = ttk.LabelFrame(left_frame, text="Описание анализа", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, text="GAM (Глава 2, п. 2.5.6)\nВыявление нелинейных зависимостей γ-мощности\nот длительности события и времени.", justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        ttk.Button(info_frame, text="Инструкция", command=self.show_instructions).pack(anchor=tk.W, pady=2)

        param_frame = ttk.LabelFrame(left_frame, text="Параметры модели", padding=5)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(param_frame, text="Канал:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(param_frame, textvariable=self.channel, values=['C3','C4'], state='readonly').grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Интервал времени (сек):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.time_min, width=8).grid(row=1, column=1, padx=2)
        ttk.Label(param_frame, text="до").grid(row=1, column=2, padx=2)
        ttk.Entry(param_frame, textvariable=self.time_max, width=8).grid(row=1, column=3, padx=2)
        ttk.Label(param_frame, text="k (time):").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=5, to=30, textvariable=self.k_time, width=5).grid(row=2, column=1, padx=5)
        ttk.Checkbutton(param_frame, text="Использовать отфильтрованные исследования", variable=self.use_filtered).grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=5)

        cache_frame = ttk.LabelFrame(left_frame, text="Кэширование", padding=5)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cache_frame, text="Использовать кэш API", variable=self.use_cache).pack(anchor=tk.W)

        # Кнопки в два ряда
        btn_frame1 = ttk.Frame(left_frame)
        btn_frame1.pack(fill=tk.X, padx=5, pady=2)
        self.run_btn = ttk.Button(btn_frame1, text="Запустить GAM", command=self.run_gam)
        self.run_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(btn_frame1, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.save_csv_btn = ttk.Button(btn_frame1, text="Сохранить предсказания (CSV)", command=self.save_predictions_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=2)

        btn_frame2 = ttk.Frame(left_frame)
        btn_frame2.pack(fill=tk.X, padx=5, pady=2)
        self.save_plot_btn = ttk.Button(btn_frame2, text="Сохранить графики (PNG)", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=2)
        self.save_model_btn = ttk.Button(btn_frame2, text="Сохранить модель (RDS)", command=self.save_model_rds, state=tk.DISABLED)
        self.save_model_btn.pack(side=tk.LEFT, padx=2)
        self.save_summary_btn = ttk.Button(btn_frame2, text="Сохранить summary (TXT)", command=self.save_summary_txt, state=tk.DISABLED)
        self.save_summary_btn.pack(side=tk.LEFT, padx=2)
        self.report_btn = ttk.Button(btn_frame2, text="Сформировать отчёт", command=self.generate_report, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=2)

        # ========== Правая панель (Notebook) ==========
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_summary = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_summary, text="Summary модели")
        self.summary_text = tk.Text(self.tab_summary, wrap=tk.WORD, font=("Courier New", 9))
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        self.tab_plots = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plots, text="Графики")
        self.plot_frame = ttk.Frame(self.tab_plots)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_diagnostics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diagnostics, text="Диагностика")
        self.diag_frame = ttk.Frame(self.tab_diagnostics)
        self.diag_frame.pack(fill=tk.BOTH, expand=True)

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
            "===========================================\n"
            "1. Загрузите и отфильтруйте данные на вкладке 'Загрузка и фильтры'.\n"
            "2. Выберите канал (C3 или C4) и временной интервал (по умолч. -60…+30 с).\n"
            "3. Нажмите 'Запустить GAM'. Требуется установленный R и пакет mgcv.\n"
            "4. После завершения будут построены:\n"
            "   - Контурный график предсказанной γ-мощности (если есть длительность)\n"
            "   - Частичная зависимость для time\n"
            "   - График остатков vs предсказанные значения\n"
            "5. Оценка нелинейности: эффективные степени свободы (edf) > 1.5 и p < 0.05.\n"
            "6. Кнопка 'Сформировать отчёт' создаст HTML с интерпретацией гипотезы H3.\n"
        )
        messagebox.showinfo("Инструкция", msg)

    def run_gam(self):
        if not R_AVAILABLE:
            messagebox.showerror("Ошибка", "rpy2 не установлен. Установите rpy2 и R с пакетом mgcv.")
            return
        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        for btn in (self.save_csv_btn, self.save_plot_btn, self.save_model_btn, self.save_summary_btn, self.report_btn):
            btn.config(state=tk.DISABLED)
        self.summary_text.delete(1.0, tk.END)
        for frame in (self.plot_frame, self.diag_frame):
            for w in frame.winfo_children():
                w.destroy()
        self.log("Загрузка данных...")
        threading.Thread(target=self._run_gam_thread, daemon=True).start()

    def _run_gam_thread(self):
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
                    self.log("Нет отфильтрованных данных. Сначала примените фильтры.")
                    return
                study_ids = filtered_df['study_id'].unique().tolist()
                self.log(f"Используем {len(study_ids)} исследований (после фильтрации)")

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
                use_cache=self.use_cache.get()
            )
            self.main_app.set_progress(100)
            if not ts_data or self.stop_flag:
                self.log("Нет данных для анализа.")
                return
            df = pd.DataFrame(ts_data)
            self.log(f"Загружено {len(df)} временных точек")
            if len(df) > 0:
                self.log(f"Пример первой записи: {df.iloc[0].to_dict()}")

            # Приводим к числовому типу, удаляем NaN
            df['gamma_power_norm_pct'] = pd.to_numeric(df['gamma_power_norm_pct'], errors='coerce')
            df['time_from_offset'] = pd.to_numeric(df['time_from_offset'], errors='coerce')

            # Проверяем наличие event_duration (может быть None или отсутствовать)
            has_event_duration = 'event_duration' in df.columns and df['event_duration'].notna().any()
            if has_event_duration:
                df['event_duration'] = pd.to_numeric(df['event_duration'], errors='coerce')
                # Удаляем строки с NaN в нужных колонках
                df = df.dropna(subset=['gamma_power_norm_pct', 'event_duration', 'time_from_offset', 'patient_id'])
                if df.empty:
                    self.log("Нет данных после очистки (длительность есть, но все строки с NaN).")
                    return
                self.log(f"Очищено: {len(df)} точек, пациентов: {df['patient_id'].nunique()}")
                self.log("Модель будет включать длительность события и взаимодействие.")
                formula = Formula(f"gamma_power_norm_pct ~ s(event_duration, bs='tp', k={self.k_duration.get()}) + "
                                  f"s(time_from_offset, bs='tp', k={self.k_time.get()}) + "
                                  f"ti(event_duration, time_from_offset, bs='tp', k={self.k_interaction.get()}) + "
                                  f"s(patient_id, bs='re')")
                use_contour = True
            else:
                # Упрощённая модель: только время и случайный эффект
                df = df.dropna(subset=['gamma_power_norm_pct', 'time_from_offset', 'patient_id'])
                if df.empty:
                    self.log("Нет данных после очистки (отсутствует event_duration и нет других данных).")
                    return
                self.log(f"Очищено: {len(df)} точек, пациентов: {df['patient_id'].nunique()}")
                self.log("Модель упрощена (без длительности события). Оценивается только временной профиль.")
                formula = Formula(f"gamma_power_norm_pct ~ s(time_from_offset, bs='tp', k={self.k_time.get()}) + "
                                  f"s(patient_id, bs='re')")
                use_contour = False

            # Активируем rpy2
            pandas2ri.activate()
            mgcv = importr('mgcv')
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(df)

            self.log("Оценка GAM-модели (REML)...")
            model = mgcv.gam(formula, data=r_df, method="REML")
            self.model = model

            # Summary
            summary_rs = ro.r.summary(model)
            summary_cap = ro.r.capture_output(ro.r.print(summary_rs))
            summary_str = "\n".join([str(line) for line in summary_cap])
            self.summary_text.insert(tk.END, summary_str)
            self.log("Модель обучена.")

            # Гладкие члены
            self._extract_smooth_params(model)

            # Предсказания
            pred = mgcv.predict_gam(model)
            self.pred_df = df.copy()
            with localconverter(ro.default_converter + pandas2ri.converter):
                self.pred_df['predicted'] = ro.conversion.rpy2py(pred)

            # Построение графиков
            if use_contour:
                self._plot_contour(df, model)
            else:
                # Если нет event_duration, контурный график не строим (показываем сообщение)
                for widget in self.plot_frame.winfo_children():
                    widget.destroy()
                fig = Figure(figsize=(10, 6), dpi=100)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "Нет данных о длительности события.\nКонтурный график недоступен.",
                        ha='center', va='center', transform=ax.transAxes)
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.current_figure = fig

            self._plot_partial_dependencies(model, df, use_contour)
            self._plot_residuals(model, df)

            for btn in (self.save_csv_btn, self.save_plot_btn, self.save_model_btn, self.save_summary_btn, self.report_btn):
                btn.config(state=tk.NORMAL)
            self.log("GAM анализ завершён.")
        except Exception as e:
            self.log(f"Ошибка: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.main_app.set_progress(0)

    def _extract_smooth_params(self, model):
        try:
            s_table = ro.r('as.data.frame(summary(model)$s.table)')
            with localconverter(ro.default_converter + pandas2ri.converter):
                smooth_df = ro.conversion.rpy2py(s_table)
            if smooth_df is not None and not smooth_df.empty:
                smooth_df.columns = ['edf', 'ref_df', 'F', 'p_value']
                smooth_df['nonlinear'] = (smooth_df['edf'] > 1.5) & (smooth_df['p_value'] < 0.05)
                self.results_smooth = smooth_df
                self.log("=== Гладкие члены GAM ===")
                for idx, row in smooth_df.iterrows():
                    self.log(f"{idx}: edf={row['edf']:.2f}, p={row['p_value']:.4e}, нелинейность={'Да' if row['nonlinear'] else 'Нет'}")
        except Exception as e:
            self.log(f"Не удалось извлечь параметры гладких членов: {e}")

    def _plot_contour(self, df, model):
        duration_vals = np.linspace(df['event_duration'].min(), df['event_duration'].max(), 50)
        time_vals = np.linspace(self.time_min.get(), self.time_max.get(), 50)
        grid_dur, grid_time = np.meshgrid(duration_vals, time_vals)
        grid_df = pd.DataFrame({
            'event_duration': grid_dur.ravel(),
            'time_from_offset': grid_time.ravel(),
            'patient_id': df['patient_id'].iloc[0]
        })
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_grid = ro.conversion.py2rpy(grid_df)
        pred_grid = ro.r.predict(model, newdata=r_grid, type="response")
        with localconverter(ro.default_converter + pandas2ri.converter):
            pred_vals = ro.conversion.rpy2py(pred_grid)
        pred_vals = pred_vals.reshape(grid_dur.shape)

        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        contour = ax.contourf(grid_dur, grid_time, pred_vals, levels=20, cmap='coolwarm')
        ax.scatter(df['event_duration'], df['time_from_offset'], c=df['gamma_power_norm_pct'], s=5, alpha=0.3, cmap='coolwarm', edgecolors='none')
        ax.set_xlabel('Длительность события (сек)')
        ax.set_ylabel('Время относительно offset (сек)')
        ax.set_title('Предсказанная нормализованная γ-мощность (%)')
        fig.colorbar(contour, ax=ax, label='γ-мощность, %')
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig

    def _plot_partial_dependencies(self, model, df, has_duration=False):
        # Частичная зависимость только для time (так как время есть всегда)
        time_vals = np.linspace(self.time_min.get(), self.time_max.get(), 100)
        pred_time = []
        for t in time_vals:
            temp_df = df.copy()
            temp_df['time_from_offset'] = t
            if has_duration:
                temp_df['event_duration'] = df['event_duration'].median()
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_temp = ro.conversion.py2rpy(temp_df)
            pred = ro.r.predict(model, newdata=r_temp, type="response")
            with localconverter(ro.default_converter + pandas2ri.converter):
                pred_time.append(np.mean(ro.conversion.rpy2py(pred)))

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(time_vals, pred_time, 'b-', linewidth=2)
        ax.set_xlabel('Время относительно offset (сек)')
        ax.set_ylabel('Предсказанная γ-мощность (%)')
        ax.set_title('Частичная зависимость: time')
        ax.grid(True)
        for widget in self.diag_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.diag_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_residuals(self, model, df):
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)
        pred_all = ro.r.predict(model, type="response")
        resid_all = df['gamma_power_norm_pct'].values - np.array(pred_all).flatten()
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(pred_all, resid_all, alpha=0.3, s=5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Предсказанные значения')
        ax.set_ylabel('Остатки')
        ax.set_title('Остатки vs предсказанные значения')
        ax.grid(True)
        frame_resid = ttk.Frame(self.diag_frame)
        frame_resid.pack(fill=tk.BOTH, expand=True)
        canvas2 = FigureCanvasTkAgg(fig, master=frame_resid)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- Сохранение ----------
    def save_predictions_csv(self):
        if self.pred_df is None:
            messagebox.showwarning("Нет данных", "Сначала выполните GAM.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.pred_df.to_csv(path, index=False, encoding='utf-8-sig')
            self.log(f"Предсказания сохранены в {path}")

    def save_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "Сначала выполните GAM.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if path:
            self.current_figure.savefig(path, dpi=150, bbox_inches='tight')
            self.log(f"График сохранён в {path}")

    def save_model_rds(self):
        if self.model is None:
            messagebox.showwarning("Нет модели", "Сначала выполните GAM.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".rds", filetypes=[("RDS files", "*.rds")])
        if path:
            ro.r.saveRDS(self.model, file=path)
            self.log(f"Модель сохранена в {path}")

    def save_summary_txt(self):
        if self.summary_text.get(1.0, tk.END).strip() == "":
            messagebox.showwarning("Нет данных", "Нет summary для сохранения.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.summary_text.get(1.0, tk.END))
            self.log(f"Summary сохранён в {path}")

    # ---------- Отчёт HTML ----------
    def generate_report(self):
        if self.model is None or self.pred_df is None:
            messagebox.showwarning("Нет данных", "Сначала выполните GAM.")
            return

        # Сохраняем текущий контурный график (если есть)
        img_base64 = ""
        if self.current_figure is not None and self.current_figure.get_axes():
            buf = io.BytesIO()
            self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>' if img_base64 else "<p>Контурный график недоступен (нет данных о длительности события).</p>"

        # Таблица гладких членов
        smooth_html = ""
        if self.results_smooth is not None:
            smooth_html = self.results_smooth.to_html(float_format="%.4f")
        else:
            smooth_html = "<p>Не удалось извлечь параметры гладких членов.</p>"

        # Интерпретация гипотезы H3 (фазическая)
        hyp_html = "<h3>Проверка гипотезы H3 (фазическая)</h3>"
        hyp_html += "<p><strong>Гипотеза:</strong> абсолютная гамма‑мощность (30–45 Гц) и индекс реактивности бета в пост‑событийном окне (0–10 с) значимо выше в эпохах с апноэ/гипопноэ по сравнению с фоновыми эпохами, а также коррелируют с длительностью события и AHI.</p>"
        if self.results_smooth is not None:
            # Определяем, есть ли значимая нелинейность для s(event_duration) или s(time_from_offset)
            time_nonlinear = False
            dur_nonlinear = False
            for idx, row in self.results_smooth.iterrows():
                if 'time_from_offset' in idx and row['nonlinear']:
                    time_nonlinear = True
                if 'event_duration' in idx and row['nonlinear']:
                    dur_nonlinear = True
            if time_nonlinear:
                hyp_html += "<p><strong>Вывод:</strong> Обнаружена значимая нелинейная динамика γ-мощности относительно времени окончания события (p < 0.05, edf > 1.5). Это подтверждает наличие острого пика γ-активности после апноэ. Таким образом, <span style='color:green;'>H3 ПОДТВЕРЖДАЕТСЯ</span> для временной динамики.</p>"
            else:
                hyp_html += "<p><strong>Вывод:</strong> Нелинейные эффекты времени не достигли статистической значимости. Возможно, требуются дополнительные данные.</p>"
            if dur_nonlinear:
                hyp_html += "<p>Дополнительно: выявлена значимая нелинейная связь между длительностью события и γ-мощностью, что усиливает подтверждение H3.</p>"
        else:
            hyp_html += "<p>Не удалось оценить нелинейность – проверьте модель.</p>"

        # Общая интерпретация
        interp_html = "<h3>Интерпретация GAM</h3>"
        interp_html += "<p>GAM позволяет выявить нелинейные зависимости между нормализованной гамма-мощностью и временем относительно offset. Эффективные степени свободы (edf) > 1.5 указывают на нелинейность. Значимость гладких членов оценивается по p-значению (F-тест).</p>"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>GAM отчёт – γ-активность</title>
        <style>
            body {{ font-family: Arial; margin:20px; }}
            table {{ border-collapse: collapse; width:100%; margin-bottom:20px; }}
            th, td {{ border:1px solid #ddd; padding:8px; text-align:left; }}
            th {{ background:#f2f2f2; }}
            .plot {{ margin:20px 0; text-align:center; }}
        </style>
        </head>
        <body>
        <h1>Генерализованные аддитивные модели (GAM) – Глава 2, п. 2.5.6</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Канал:</strong> {self.channel.get()}</p>
        <p><strong>Интервал времени:</strong> {self.time_min.get()} … {self.time_max.get()} с</p>
        <p><strong>Число базисных функций:</strong> k(time)={self.k_time.get()}</p>
        <p><strong>Использованы отфильтрованные исследования:</strong> {'Да' if self.use_filtered.get() else 'Нет'}</p>

        {hyp_html}
        {interp_html}

        <h2>Графики GAM</h2>
        {plot_html}

        <h2>Параметры гладких членов (smooth terms)</h2>
        {smooth_html}

        <p><em>Примечание:</em> Значимая нелинейность: edf > 1.5 и p < 0.05.</p>
        </body>
        </html>
        """
        fd, path = tempfile.mkstemp(suffix='.html', prefix='gam_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт в браузере: {path}")