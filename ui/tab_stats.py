# ui/tab_stats.py
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import scipy.stats as sps
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui.base_tab import BaseTab
from core.config import SEVERITY_MAP
from core.data_processor import calc_group_stats


class StatsTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)

        self.group_by_severity = True
        self.include_central_mixed = False

        # Верхняя панель управления
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.calc_btn = ttk.Button(control_frame, text="Рассчитать статистику", command=self.calculate_stats)
        self.calc_btn.pack(side=tk.LEFT, padx=5)

        self.save_data_btn = ttk.Button(control_frame, text="Сохранить данные (CSV)", command=self.save_filtered_data)
        self.save_data_btn.pack(side=tk.LEFT, padx=5)

        self.save_stats_btn = ttk.Button(control_frame, text="Сохранить статистику (CSV)", command=self.save_stats_to_csv)
        self.save_stats_btn.pack(side=tk.LEFT, padx=5)

        self.cb_central = tk.BooleanVar(value=self.include_central_mixed)
        chk = ttk.Checkbutton(control_frame, text="Включить центральное/смешанное апноэ",
                              variable=self.cb_central, command=self._toggle_central_mixed)
        chk.pack(side=tk.LEFT, padx=10)

        self.log_label = ttk.Label(control_frame, text="", foreground="gray")
        self.log_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Вложенный Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Вкладка "Таблица"
        self.tab_table = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_table, text="Таблица")
        self.result_text = tk.Text(self.tab_table, wrap=tk.WORD, font=("Courier New", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Вкладка "Графики" (общие)
        self.tab_plots = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plots, text="Графики")
        self.plot_frame = ttk.Frame(self.tab_plots)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Вкладка "Распределения"
        self.tab_distrib = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_distrib, text="Распределения")
        self._create_distribution_tab()

        self.last_stats_dict = None
        self.last_df = None
        self.calc_btn.config(state=tk.DISABLED)
        self.current_figure = None

    # ---------- Вкладка распределений (исправленная) ----------
    def _create_distribution_tab(self):
        """Создаёт интерфейс с прокруткой для графиков распределений."""
        # Верхняя панель выбора признака
        top_frame = ttk.Frame(self.tab_distrib)
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(top_frame, text="Признак:").pack(side=tk.LEFT, padx=5)
        self.var_feature = tk.StringVar()
        self.feature_combo = ttk.Combobox(top_frame, textvariable=self.var_feature, state="readonly", width=30)
        self.feature_combo.pack(side=tk.LEFT, padx=5)
        self.feature_combo.bind("<<ComboboxSelected>>", self.on_feature_selected)

        self.update_btn = ttk.Button(top_frame, text="Обновить", command=self.update_distribution_plot)
        self.update_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(top_frame, text="Сохранить график в PNG", command=self.save_current_plot)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)

        # Контейнер с прокруткой
        self.distrib_container = ttk.Frame(self.tab_distrib)
        self.distrib_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas + Scrollbar
        self.distrib_canvas = tk.Canvas(self.distrib_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.distrib_container, orient=tk.VERTICAL, command=self.distrib_canvas.yview)
        self.distrib_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.distrib_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Внутренний фрейм, куда будем помещать графики и текст
        self.distrib_inner = ttk.Frame(self.distrib_canvas)
        self.canvas_window = self.distrib_canvas.create_window((0, 0), window=self.distrib_inner, anchor="nw")

        # Привязки для обновления прокрутки и ширины
        self.distrib_inner.bind("<Configure>", self._on_inner_configure)
        self.distrib_canvas.bind("<Configure>", self._on_canvas_configure)

        # Текстовое поле для рекомендаций (будет внутри distrib_inner)
        self.recommendation_text = tk.Text(self.distrib_inner, wrap=tk.WORD, height=10, font=("Courier New", 10))
        self.recommendation_text.pack(fill=tk.X, padx=5, pady=5)

        # Словарь для отображения русских названий
        self.feature_map = {}

    def _on_inner_configure(self, event):
        """Обновляет область прокрутки, когда меняется размер внутреннего фрейма."""
        self.distrib_canvas.configure(scrollregion=self.distrib_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """При изменении размера Canvas подстраивает ширину внутреннего окна."""
        self.distrib_canvas.itemconfig(self.canvas_window, width=event.width)

    def update_feature_list(self):
        """Заполняет выпадающий список русскими названиями признаков."""
        if self.last_df is None or self.last_df.empty:
            self.feature_combo['values'] = []
            self.feature_map = {}
            return
        numeric_cols = ['age_at_study', 'bmi', 'ahi', 'odi', 'total_sleep_time', 'sleep_efficiency',
                        'min_spo2', 'avg_spo2', 'duration_minutes']
        self.feature_map = {}
        available = []
        for col in numeric_cols:
            if col in self.last_df.columns and self.last_df[col].notna().any():
                rus_name = self._get_feature_label(col)
                self.feature_map[rus_name] = col
                available.append(rus_name)
        self.feature_combo['values'] = available
        if available and not self.var_feature.get():
            self.var_feature.set(available[0])

    def on_feature_selected(self, event=None):
        self.update_distribution_plot()

    def update_distribution_plot(self):
        if self.last_df is None or self.last_df.empty:
            self.recommendation_text.delete(1.0, tk.END)
            self.recommendation_text.insert(tk.END, "Нет данных для анализа.")
            return

        rus_name = self.var_feature.get()
        if not rus_name or rus_name not in self.feature_map:
            return
        feature = self.feature_map[rus_name]

        # Очищаем старые графики, оставляя только текстовое поле
        for widget in self.distrib_inner.winfo_children():
            if widget != self.recommendation_text:
                widget.destroy()

        data = self.last_df[feature].dropna()
        if len(data) < 3:
            self.recommendation_text.delete(1.0, tk.END)
            self.recommendation_text.insert(tk.END, f"Недостаточно данных для признака '{rus_name}' (n={len(data)}).")
            return

        # Создаём фигуру с тремя подграфиками
        fig = Figure(figsize=(10, 12), dpi=100)
        fig.subplots_adjust(hspace=0.4)
        self.current_figure = fig

        # 1. Гистограмма + плотность
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.hist(data, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        try:
            kde = sps.gaussian_kde(data)
            x_vals = np.linspace(data.min(), data.max(), 200)
            ax1.plot(x_vals, kde(x_vals), 'r-', linewidth=2, label='Плотность')
            ax1.legend()
        except:
            pass
        ax1.set_title(f'Распределение {rus_name}')
        ax1.set_xlabel(rus_name)
        ax1.set_ylabel('Плотность')

        # 2. Q-Q plot
        ax2 = fig.add_subplot(3, 1, 2)
        sps.probplot(data, dist="norm", plot=ax2)
        ax2.set_title('Q-Q plot (сравнение с нормальным распределением)')

        # 3. Boxplot по группам тяжести
        ax3 = fig.add_subplot(3, 1, 3)
        if self.group_by_severity:
            temp_df = self.last_df.copy()
            temp_df['severity_label'] = temp_df['breathing_impairment_severity'].map(SEVERITY_MAP).fillna(
                temp_df['breathing_impairment_severity'])
            if not self.include_central_mixed:
                temp_df = temp_df[~temp_df['severity_label'].isin(['Центральное', 'Смешанное'])]
            if not temp_df.empty:
                groups = []
                labels = []
                for name, group in temp_df.groupby('severity_label'):
                    vals = group[feature].dropna()
                    if len(vals) > 0:
                        groups.append(vals.values)
                        labels.append(name)
                if groups:
                    ax3.boxplot(groups, labels=labels, patch_artist=True)
                    ax3.set_title(f'Распределение {rus_name} по группам тяжести ОАС')
                    ax3.set_ylabel(rus_name)
                    ax3.tick_params(axis='x', rotation=45)
                else:
                    ax3.text(0.5, 0.5, 'Нет данных для boxplot', transform=ax3.transAxes, ha='center')
            else:
                ax3.text(0.5, 0.5, 'Группы отфильтрованы', transform=ax3.transAxes, ha='center')
        else:
            ax3.text(0.5, 0.5, 'Группировка отключена', transform=ax3.transAxes, ha='center')

        canvas = FigureCanvasTkAgg(fig, master=self.distrib_inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Статистика и рекомендации
        n = len(data)
        if n > 5000:
            shapiro_msg = "Выборка слишком большая для теста Шапиро-Уилка (n>5000). Ориентируйтесь на графики."
            p_value = None
        else:
            statistic, p_value = sps.shapiro(data)
            shapiro_msg = f"Тест Шапиро-Уилка: W = {statistic:.4f}, p = {p_value:.4e}"

        rec_text = f"=== Анализ распределения признака: {rus_name} ===\n"
        rec_text += f"Объем выборки: n = {n}\n"
        rec_text += shapiro_msg + "\n"

        if p_value is not None:
            if p_value > 0.05:
                rec_text += "Распределение не отличается от нормального (p > 0.05).\n"
                rec_text += "✅ Рекомендация: можно использовать параметрические критерии (t-тест, ANOVA) и линейные модели.\n"
            else:
                rec_text += "Распределение значимо отличается от нормального (p ≤ 0.05).\n"
                rec_text += "⚠️ Рекомендация: при сравнении групп используйте непараметрические аналоги (Mann-Whitney U, Kruskal-Wallis).\n"
                rec_text += "   Для линейных смешанных моделей (LMM) проверьте устойчивость (bootstrapping) или рассмотрите преобразования:\n"
                rec_text += "   - логарифмическое (если данные положительные и скошены вправо)\n"
                rec_text += "   - Box-Cox (если данные положительные)\n"
                rec_text += "   - ранговое преобразование (непараметрический LMM)\n"
        else:
            rec_text += "Тест Шапиро-Уилка не может быть выполнен из-за большого объема выборки.\n"
            rec_text += "⚠️ Ориентируйтесь на графики: если гистограмма и Q-Q plot показывают заметное отклонение от нормальности – применяйте непараметрические методы.\n"

        skew_val = sps.skew(data)
        if skew_val > 1:
            rec_text += f"\n🔹 Признак имеет положительную асимметрию (skewness = {skew_val:.2f}). Лог-преобразование может помочь.\n"
        elif skew_val < -1:
            rec_text += f"\n🔹 Признак имеет отрицательную асимметрию (skewness = {skew_val:.2f}).\n"

        self.recommendation_text.delete(1.0, tk.END)
        self.recommendation_text.insert(tk.END, rec_text)

        # Обновляем прокрутку
        self.distrib_inner.update_idletasks()
        self.distrib_canvas.configure(scrollregion=self.distrib_canvas.bbox("all"))

    # ---------- Вспомогательные методы ----------
    def _get_feature_label(self, feature):
        labels = {
            'age_at_study': 'Возраст (лет)',
            'bmi': 'ИМТ (кг/м²)',
            'ahi': 'AHI (событий/ч)',
            'odi': 'ODI (событий/ч)',
            'total_sleep_time': 'Общее время сна (мин)',
            'sleep_efficiency': 'Эффективность сна (%)',
            'min_spo2': 'Минимальная SpO₂ (%)',
            'avg_spo2': 'Средняя SpO₂ (%)',
            'duration_minutes': 'Длительность записи (мин)'
        }
        return labels.get(feature, feature)

    def _toggle_central_mixed(self):
        self.include_central_mixed = self.cb_central.get()
        self.main_app.set_analysis_settings(self.group_by_severity, self.include_central_mixed)
        if self.last_df is not None and not self.last_df.empty:
            self.calculate_stats()

    def update_settings(self, group_by_severity, include_central_mixed):
        self.group_by_severity = group_by_severity
        self.include_central_mixed = include_central_mixed
        self.cb_central.set(include_central_mixed)
        if self.last_df is not None:
            self.calculate_stats()

    def on_tab_selected(self):
        data = self.main_app.get_filtered_data()
        if data is not None and not data.empty:
            self.calc_btn.config(state=tk.NORMAL)
            if self.last_df is None:
                self.calculate_stats()
        else:
            self.calc_btn.config(state=tk.DISABLED)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Нет данных для отображения.\nЗагрузите и отфильтруйте исследования.")
            self.last_df = None
            self.last_stats_dict = None

    def calculate_stats(self):
        df = self.main_app.get_filtered_data()
        if df is None or df.empty:
            messagebox.showwarning("Нет данных", "Нет отфильтрованных данных.")
            return

        self.last_df = df.copy()
        self.log_label.config(text="Расчёт статистики...")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Расчёт статистики...\n")
        self.update_idletasks()

        # Обновляем список признаков для вкладки распределений
        self.update_feature_list()
        if self.var_feature.get():
            self.update_distribution_plot()

        df_work = df.copy()
        df_work['severity_label'] = df_work['breathing_impairment_severity'].map(SEVERITY_MAP).fillna(df_work['breathing_impairment_severity'])

        if not self.include_central_mixed:
            mask = ~df_work['severity_label'].isin(['Центральное', 'Смешанное'])
            df_work = df_work[mask].copy()
            if df_work.empty:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "После исключения центрального/смешанного апноэ нет данных.\n")
                self.log_label.config(text="Нет данных после фильтрации")
                return

        if self.group_by_severity:
            groups = df_work.groupby('severity_label')
            stats = {'Вся выборка': calc_group_stats(df_work)}
            desired_order = ['Норма (<5)', 'Лёгкая (5-14.9)', 'Умеренная (15-29.9)', 'Тяжёлая (≥30)']
            for name in desired_order:
                if name in groups.groups:
                    stats[name] = calc_group_stats(groups.get_group(name))
            for name, group in groups:
                if name not in desired_order:
                    stats[name] = calc_group_stats(group)
        else:
            stats = {'Вся выборка': calc_group_stats(df_work)}

        self.last_stats_dict = stats
        self.display_stats(stats)
        self.plot_stats(df_work, stats)
        self.log_label.config(text=f"Статистика рассчитана. Групп: {len(stats)}")

    def display_stats(self, stats_dict):
        desired_order = ['Норма (<5)', 'Лёгкая (5-14.9)', 'Умеренная (15-29.9)', 'Тяжёлая (≥30)']
        ordered_groups = ['Вся выборка']
        for name in desired_order:
            if name in stats_dict:
                ordered_groups.append(name)
        for name in stats_dict:
            if name not in ordered_groups and name != 'Вся выборка':
                ordered_groups.append(name)
        groups = ordered_groups
        width = 35 + 25 * len(groups)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "=" * width + "\n")
        self.result_text.insert(tk.END, "ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПО ГРУППАМ ТЯЖЕСТИ ОАС\n")
        self.result_text.insert(tk.END, "=" * width + "\n")

        header = f"{'Параметр':<35}"
        for g in groups:
            header += f"{g:<25}"
        self.result_text.insert(tk.END, header + "\n")
        self.result_text.insert(tk.END, "-" * width + "\n")

        rows = [
            ("Пациентов, n", "n"),
            ("Мужчины, %", "male_pct"),
            ("Возраст, лет", "age"),
            ("AHI, событий/ч", "ahi"),
            ("ODI, событий/ч", "odi"),
            ("Минимальная SpO₂, %", "min_spo2"),
            ("Средняя SpO₂, %", "avg_spo2"),
            ("Общее время сна (TST), мин", "tst"),
            ("Эффективность сна, %", "eff")
        ]
        for label, key in rows:
            line = f"{label:<35}"
            for g in groups:
                val = stats_dict[g].get(key, "NA")
                line += f"{val:<25}"
            self.result_text.insert(tk.END, line + "\n")

        self.result_text.insert(tk.END, "\nКоморбидные состояния (% от группы):\n")
        self.result_text.insert(tk.END, "-" * width + "\n")
        comorb = [
            ("Артериальная гипертензия", "hypertension"),
            ("Ишемическая болезнь сердца", "ihd"),
            ("Сахарный диабет 2 типа", "diabetes"),
            ("Инсомния", "insomnia")
        ]
        for label, key in comorb:
            line = f"{label:<35}"
            for g in groups:
                val = stats_dict[g].get(key, "NA")
                line += f"{val:<25}"
            self.result_text.insert(tk.END, line + "\n")

    def plot_stats(self, df, stats_dict):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        if df.empty or len(stats_dict) < 2:
            label = ttk.Label(self.plot_frame, text="Недостаточно данных для построения графиков (нужно минимум 2 группы)")
            label.pack()
            return

        fig = Figure(figsize=(10, 8), dpi=100)
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        groups = [g for g in stats_dict.keys() if g != 'Вся выборка' and g not in ['Центральное', 'Смешанное']]
        # AHI
        ax1 = fig.add_subplot(2, 2, 1)
        ahi_values = []
        for g in groups:
            ahi_str = stats_dict[g].get('ahi', '0±0')
            mean = float(ahi_str.split('±')[0]) if '±' in ahi_str else 0
            ahi_values.append(mean)
        ax1.bar(groups, ahi_values, color='skyblue')
        ax1.set_title('Индекс апноэ-гипопноэ (AHI) по группам')
        ax1.set_ylabel('AHI, событий/ч')
        ax1.tick_params(axis='x', rotation=45)

        # Возраст
        ax2 = fig.add_subplot(2, 2, 2)
        age_vals = []
        for g in groups:
            age_str = stats_dict[g].get('age', '0±0')
            mean = float(age_str.split('±')[0]) if '±' in age_str else 0
            age_vals.append(mean)
        ax2.bar(groups, age_vals, color='lightgreen')
        ax2.set_title('Возраст пациентов')
        ax2.set_ylabel('Возраст, лет')
        ax2.tick_params(axis='x', rotation=45)

        # Гистограмма AHI
        ax3 = fig.add_subplot(2, 2, 3)
        if 'ahi' in df.columns and df['ahi'].notna().any():
            ax3.hist(df['ahi'].dropna(), bins=30, color='salmon', edgecolor='black')
            ax3.set_title('Распределение AHI во всей выборке')
            ax3.set_xlabel('AHI, событий/ч')
            ax3.set_ylabel('Частота')
        else:
            ax3.text(0.5, 0.5, 'Нет данных AHI', transform=ax3.transAxes, ha='center')

        # Коморбидности
        ax4 = fig.add_subplot(2, 2, 4)
        comorbidities = ['cvd_hypertension', 'cvd_ihd', 'endocrine_diabetes', 'som_insomnia']
        labels = ['Гипертензия', 'ИБС', 'СД 2 типа', 'Инсомния']
        values = []
        for col in comorbidities:
            pct = df[col].mean() * 100 if col in df.columns else 0
            values.append(pct)
        ax4.bar(labels, values, color='purple')
        ax4.set_title('Коморбидные состояния (вся выборка)')
        ax4.set_ylabel('Распространённость, %')
        ax4.tick_params(axis='x', rotation=45)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_filtered_data(self):
        if self.last_df is None or self.last_df.empty:
            messagebox.showwarning("Нет данных", "Нет отфильтрованных данных для сохранения.")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df_to_save = self.last_df.copy()
            if 'file_path' in df_to_save.columns:
                df_to_save['file_path'] = df_to_save['file_path'].apply(
                    lambda x: os.path.basename(x) if pd.notna(x) and x else '')
            df_to_save.to_csv(file_path, index=False, encoding='utf-8-sig')
            self.log(f"Отфильтрованные данные сохранены в {file_path}")

    def save_stats_to_csv(self):
        if not self.last_stats_dict:
            messagebox.showwarning("Нет статистики", "Сначала рассчитайте статистику.")
            return
        rows = []
        for group_name, metrics in self.last_stats_dict.items():
            row = {'Группа': group_name}
            row.update(metrics)
            rows.append(row)
        df_stats = pd.DataFrame(rows)
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df_stats.to_csv(file_path, index=False, encoding='utf-8-sig')
            self.log(f"Статистика сохранена в {file_path}")

    def save_current_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "Сначала выберите признак и нажмите 'Обновить'.")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            self.current_figure.savefig(file_path, dpi=150, bbox_inches="tight")
            self.log(f"График сохранён: {file_path}")