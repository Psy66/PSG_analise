# ui/tab_dfa_coherence.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import tempfile
import webbrowser
import io
import base64
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import false_discovery_control, shapiro, chi2
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from ui.base_tab import BaseTab
from core.api_client import get_epochs
from core.config import DEFAULT_API_URL, DEFAULT_TOKEN

# Константы
DFA_CHANNELS = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
COH_PAIRS = [
    ('F3', 'C3'), ('F3', 'F4'), ('F3', 'C4'),
    ('C3', 'C4'), ('C3', 'O1'), ('F4', 'C4'),
    ('O1', 'O2'), ('C3', 'O2')
]
COH_BANDS = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma']


class DfaCoherenceTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        # Настройки
        self.data_type = tk.IntVar(value=2)          # 2 = все эпохи
        self.use_filtered = tk.BooleanVar(value=True)
        self.use_cache = tk.BooleanVar(value=True)
        self.analysis_mode = tk.StringVar(value='lmm_dfa')   # 'lmm_dfa', 'lmm_coh'
        self.include_covariates = tk.BooleanVar(value=True)
        self.include_stage = tk.BooleanVar(value=True)
        self.fdr_threshold = tk.DoubleVar(value=0.05)
        self.exclude_central_mixed = tk.BooleanVar(value=True)

        # Выбор каналов/пар/диапазонов
        self.selected_dfa_channels = []
        self.selected_coh_pairs = []
        self.selected_coh_bands = []

        # Данные и результаты
        self.epochs_df = None
        self.filtered_study_ids = None
        self.results_df = None          # DataFrame с beta, p_value, q_value...
        self.current_figure = None
        self.stop_flag = False

        # Для диагностики и бутстрапа
        self.diag_model = None
        self.diag_feature_name = ""
        self.diag_figure = None
        self.bootstrap_results = None
        self.all_diagnostics = []       # список dict для отчёта
        self.normality_results = {}     # {feature: shapiro_p}

        self._create_widgets()

    # ------------------------------------------------------------
    # Построение интерфейса (аналогично LMM)
    # ------------------------------------------------------------
    def _create_widgets(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Прокрутка всей левой панели (содержимое может быть большим)
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
        # --- Источник данных и описание ---
        info_frame = ttk.LabelFrame(left_frame, text="Источник данных", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, text="Используются отфильтрованные данные из вкладки 'Загрузка'",
                  foreground="blue").pack(anchor=tk.W)
        ttk.Label(info_frame, text="(токен и URL не требуются)").pack(anchor=tk.W)

        desc_frame = ttk.LabelFrame(left_frame, text="Что такое DFA и когерентность?", padding=5)
        desc_frame.pack(fill=tk.X, padx=5, pady=5)
        desc_text = (
            "DFA (детрендовый флуктуационный анализ) оценивает долговременную корреляцию сигнала: "
            "показатель α > 0.5 указывает на персистентность. Когерентность измеряет линейную связь между "
            "двумя отведениями в различных частотных диапазонах. Линейные смешанные модели (LMM) выявляют "
            "значимые различия этих показателей между эпохами с апноэ и без, с учётом возраста, пола, ИМТ и "
            "стадии сна, а также индивидуальных различий пациентов."
        )
        ttk.Label(desc_frame, text=desc_text, justify=tk.LEFT, wraplength=600).pack(anchor=tk.W, fill=tk.X, padx=5, pady=2)
        ttk.Button(desc_frame, text="Показать инструкцию", command=self.show_instructions).pack(anchor=tk.W, padx=5, pady=2)

        # --- Общие настройки данных ---
        data_frame = ttk.LabelFrame(left_frame, text="Данные", padding=5)
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(data_frame, text="Тип набора эпох:").grid(row=0, column=0, sticky=tk.W)
        self.data_type_combo = ttk.Combobox(data_frame, textvariable=self.data_type, values=[1,2,3], state='readonly', width=5)
        self.data_type_combo.grid(row=0, column=1, padx=5)
        ttk.Label(data_frame, text="1=тонический, 2=все эпохи, 3=фильтр по положению").grid(row=0, column=2, sticky=tk.W)
        ttk.Checkbutton(data_frame, text="Использовать отфильтрованные исследования",
                        variable=self.use_filtered).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Checkbutton(data_frame, text="Использовать кэш API", variable=self.use_cache).grid(row=2, column=0, columnspan=3, sticky=tk.W)
        ttk.Checkbutton(data_frame, text="Исключить центральное/смешанное апноэ",
                        variable=self.exclude_central_mixed).grid(row=3, column=0, columnspan=3, sticky=tk.W)

        # --- Режим анализа ---
        mode_frame = ttk.LabelFrame(left_frame, text="Режим анализа", padding=5)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Radiobutton(mode_frame, text="LMM для DFA (показатель α)", variable=self.analysis_mode,
                        value='lmm_dfa', command=self._toggle_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="LMM для когерентности", variable=self.analysis_mode,
                        value='lmm_coh', command=self._toggle_mode).pack(anchor=tk.W)

        # --- Выбор каналов/пар (скрываемые панели) ---
        self.dfa_select_frame = ttk.LabelFrame(left_frame, text="Выбор каналов DFA", padding=5)
        self.dfa_listbox = tk.Listbox(self.dfa_select_frame, selectmode=tk.EXTENDED, height=6, width=20)
        for ch in DFA_CHANNELS:
            self.dfa_listbox.insert(tk.END, ch)
        self.dfa_listbox.selection_set(0, tk.END)
        self.dfa_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.coh_select_frame = ttk.LabelFrame(left_frame, text="Выбор пар и диапазонов", padding=5)
        ttk.Label(self.coh_select_frame, text="Пары отведений:").pack(anchor=tk.W)
        self.pairs_listbox = tk.Listbox(self.coh_select_frame, selectmode=tk.EXTENDED, height=5)
        for a, b in COH_PAIRS:
            self.pairs_listbox.insert(tk.END, f"{a}-{b}")
        self.pairs_listbox.selection_set(0, tk.END)
        self.pairs_listbox.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(self.coh_select_frame, text="Частотные диапазоны:").pack(anchor=tk.W)
        self.bands_listbox = tk.Listbox(self.coh_select_frame, selectmode=tk.EXTENDED, height=4)
        for b in COH_BANDS:
            self.bands_listbox.insert(tk.END, b)
        self.bands_listbox.selection_set(0, tk.END)
        self.bands_listbox.pack(fill=tk.X, padx=5, pady=2)

        # --- Параметры LMM ---
        lmm_frame = ttk.LabelFrame(left_frame, text="Параметры LMM", padding=5)
        lmm_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(lmm_frame, text="Включить ковариаты (возраст, пол, ИМТ)",
                        variable=self.include_covariates).pack(anchor=tk.W)
        ttk.Checkbutton(lmm_frame, text="Включить стадию сна (N2/N3) как фактор",
                        variable=self.include_stage).pack(anchor=tk.W)
        ttk.Label(lmm_frame, text="Порог FDR (q):").pack(anchor=tk.W)
        ttk.Entry(lmm_frame, textvariable=self.fdr_threshold, width=6).pack(anchor=tk.W)

        # --- Диагностика и бутстрап ---
        diag_frame = ttk.LabelFrame(left_frame, text="Диагностика модели", padding=5)
        diag_frame.pack(fill=tk.X, padx=5, pady=5)
        # Выбор признака для диагностики (заполнится после анализа)
        ttk.Label(diag_frame, text="Признак для диагностики:").pack(anchor=tk.W)
        self.diag_feature_combo = ttk.Combobox(diag_frame, state='readonly', width=30)
        self.diag_feature_combo.pack(fill=tk.X, padx=5, pady=2)
        self.diag_btn = ttk.Button(diag_frame, text="Диагностика остатков", command=self.run_diagnostics, state=tk.DISABLED)
        self.diag_btn.pack(anchor=tk.W, pady=2)
        self.bootstrap_btn = ttk.Button(diag_frame, text="Бутстрап (1000 итераций)", command=self.run_bootstrap, state=tk.DISABLED)
        self.bootstrap_btn.pack(anchor=tk.W, pady=2)
        self.save_diag_btn = ttk.Button(diag_frame, text="Сохранить диагностику (CSV)", command=self.save_diagnostics_csv, state=tk.DISABLED)
        self.save_diag_btn.pack(anchor=tk.W, pady=2)

        # --- Кнопки управления ---
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Запустить LMM", command=self.run_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить CSV", command=self.save_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать отчёт", command=self.generate_report, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=5)

        # ========== ПРАВАЯ ПАНЕЛЬ: вкладки ==========
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_table = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_table, text="Таблица результатов")
        self.tree = ttk.Treeview(self.tab_table)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(self.tab_table, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tab_plots = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plots, text="Графики LMM")
        self.plot_frame = ttk.Frame(self.tab_plots)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_diag = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diag, text="Диагностика модели")
        self.diag_text = tk.Text(self.tab_diag, wrap=tk.WORD, font=("Courier New", 9), height=8)
        self.diag_text.pack(fill=tk.X, padx=5, pady=5)
        self.diag_plot_frame = ttk.Frame(self.tab_diag)
        self.diag_plot_frame.pack(fill=tk.BOTH, expand=True)

        self.tab_bootstrap = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bootstrap, text="Бутстрап")
        self.bootstrap_tree_frame = ttk.Frame(self.tab_bootstrap)
        self.bootstrap_tree_frame.pack(fill=tk.BOTH, expand=True)
        self.bootstrap_tree = ttk.Treeview(self.bootstrap_tree_frame,
                                           columns=('feature','beta','ci_low','ci_high','p_bootstrap','significant'),
                                           show='headings')
        for col in ('feature','beta','ci_low','ci_high','p_bootstrap','significant'):
            self.bootstrap_tree.heading(col, text=col)
            self.bootstrap_tree.column(col, width=120)
        self.bootstrap_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_boot = ttk.Scrollbar(self.bootstrap_tree_frame, orient=tk.VERTICAL, command=self.bootstrap_tree.yview)
        sb_boot.pack(side=tk.RIGHT, fill=tk.Y)
        self.bootstrap_tree.configure(yscrollcommand=sb_boot.set)

        self.tab_log = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_log, text="Лог")
        self.log_text = tk.Text(self.tab_log, wrap=tk.WORD, font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._toggle_mode()

    def _toggle_mode(self):
        mode = self.analysis_mode.get()
        self.dfa_select_frame.pack_forget()
        self.coh_select_frame.pack_forget()
        if mode == 'lmm_dfa':
            self.dfa_select_frame.pack(fill=tk.X, padx=5, pady=5)
        elif mode == 'lmm_coh':
            self.coh_select_frame.pack(fill=tk.X, padx=5, pady=5)

    # ------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.main_app.log(msg)

    def stop_analysis(self):
        self.stop_flag = True
        self.log("Остановка анализа запрошена...")

    def show_instructions(self):
        msg = (
            "ИНСТРУКЦИЯ ПО АНАЛИЗУ DFA И КОГЕРЕНТНОСТИ\n"
            "==========================================\n"
            "1. Загрузите и отфильтруйте данные на вкладке 'Загрузка и фильтры'.\n"
            "2. Выберите режим анализа (DFA или когерентность) и отметьте нужные каналы/пары.\n"
            "3. Установите параметры LMM (включение ковариат, стадии сна, порог FDR).\n"
            "4. Нажмите 'Запустить LMM'. Будут рассчитаны модели для всех комбинаций.\n"
            "5. После завершения появятся таблица и графики (forest plot, volcano plot).\n"
            "6. Выберите признак в выпадающем списке в разделе 'Диагностика модели' и нажмите\n"
            "   'Диагностика остатков' – будут построены Q-Q plot и график остатков.\n"
            "7. При нарушении допущений запустите бутстрап (1000 итераций) для робастных CI.\n"
            "8. Кнопка 'Сформировать отчёт' создаст HTML-отчёт со всеми результатами.\n"
            "9. Результаты можно сохранить в CSV (таблица) и PNG (графики)."
        )
        messagebox.showinfo("Инструкция", msg)

    # ------------------------------------------------------------
    # Загрузка данных
    # ------------------------------------------------------------
    def _load_epochs(self):
        """Загружает эпохи через API, используя настройки из вкладки загрузки."""
        # Берём API из вкладки загрузки
        load_tab = self.main_app.tabs['load']
        api_url = load_tab.api_url.get().rstrip('/')
        token = load_tab.token.get().strip()
        if not api_url or not token:
            self.log("Ошибка: не указаны URL или токен API. Перейдите на вкладку 'Загрузка'.")
            return None

        # Определяем список study_ids
        study_ids = None
        if self.use_filtered.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is None or filtered_df.empty:
                self.log("Нет отфильтрованных данных. Сначала загрузите и отфильтруйте исследования.")
                return None
            study_ids = filtered_df['study_id'].unique().tolist()
            if not study_ids:
                self.log("В отфильтрованных данных нет study_id.")
                return None
            self.log(f"Используются исследования: {len(study_ids)}")

        if self.exclude_central_mixed.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is not None:
                allowed = ['no_impairment', 'mild', 'moderate', 'severe']
                filtered_df = filtered_df[filtered_df['breathing_impairment_severity'].isin(allowed)]
                if not filtered_df.empty:
                    study_ids = filtered_df['study_id'].unique().tolist()
                    self.log(f"После исключения центрального/смешанного апноэ осталось исследований: {len(study_ids)}")

        def update_progress(page, total, _):
            if total > 0:
                self.main_app.set_progress(int(page / total * 100))

        self.main_app.set_progress(0)
        self.log("Загрузка эпох из API (может занять несколько минут)...")
        epochs = get_epochs(
            api_url, token,
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
        return df

    # ------------------------------------------------------------
    # Основной анализ (LMM)
    # ------------------------------------------------------------
    def run_analysis(self):
        if self.epochs_df is None:
            df = self._load_epochs()
            if df is None:
                return
        else:
            df = self.epochs_df

        # Фильтрация по типу эпох (data_type уже учтён в API)
        df = df[df['data_type'] == self.data_type.get()].copy()
        if df.empty:
            self.log("Нет эпох для выбранного типа данных.")
            return

        # Подготовка данных
        df['has_apnea'] = df['has_apnea'].astype(bool)
        df['patient_id'] = df['patient_id'].astype(int)
        if 'epoch_stage' in df.columns:
            df['epoch_stage'] = df['epoch_stage'].astype('category')
        # Добавляем ковариаты
        if self.include_covariates.get():
            cov_df = self._get_covariates()
            if cov_df is not None and not cov_df.empty:
                for c in ['age_at_study', 'gender_code', 'bmi']:
                    if c in df.columns:
                        df = df.drop(columns=[c])
                df = df.merge(cov_df[['study_id', 'age_at_study', 'gender_code', 'bmi']], on='study_id', how='left')
                self.log("Ковариаты добавлены.")
            else:
                self.log("Ковариаты не загружены, продолжаем без них.")
                self.include_covariates.set(False)

        # Формируем список задач в зависимости от режима
        tasks = []
        if self.analysis_mode.get() == 'lmm_dfa':
            selected_indices = self.dfa_listbox.curselection()
            if not selected_indices:
                selected_channels = DFA_CHANNELS
            else:
                selected_channels = [self.dfa_listbox.get(i) for i in selected_indices]
            for ch in selected_channels:
                col = f"{ch}_dfa"
                if col in df.columns:
                    tasks.append((col, f"DFA {ch}", ch))
                else:
                    self.log(f"Столбец {col} не найден, пропускаем.")
        else:  # когерентность
            pair_indices = self.pairs_listbox.curselection()
            if not pair_indices:
                selected_pairs = [f"{a}-{b}" for a, b in COH_PAIRS]
            else:
                selected_pairs = [self.pairs_listbox.get(i) for i in pair_indices]
            band_indices = self.bands_listbox.curselection()
            if not band_indices:
                selected_bands = COH_BANDS
            else:
                selected_bands = [self.bands_listbox.get(i) for i in band_indices]
            for pair_str in selected_pairs:
                a, b = pair_str.split('-')
                for band in selected_bands:
                    col = f"{a}{b}_coh_{band}"
                    if col in df.columns:
                        tasks.append((col, f"Coh {pair_str} {band}", f"{pair_str}_{band}"))
                    else:
                        self.log(f"Столбец {col} не найден, пропускаем.")

        if not tasks:
            self.log("Нет доступных признаков для анализа.")
            return

        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_csv_btn.config(state=tk.DISABLED)
        self.save_plot_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        self.diag_btn.config(state=tk.DISABLED)
        self.bootstrap_btn.config(state=tk.DISABLED)
        self.save_diag_btn.config(state=tk.DISABLED)

        threading.Thread(target=self._run_lmm_thread, args=(df, tasks), daemon=True).start()

    def _run_lmm_thread(self, df, tasks):
        try:
            self.log(f"Всего моделей: {len(tasks)}")
            results = []
            total = len(tasks)
            for i, (col, feature_name, channel) in enumerate(tasks):
                if self.stop_flag:
                    break
                self.main_app.set_progress(int(i / total * 100))
                res = self._fit_lmm_model(df, col, feature_name)
                if res:
                    results.append(res)
                if (i+1) % 20 == 0:
                    self.log(f"Обработано {i+1} из {total}")
            if not results:
                self.log("Нет результатов (модели не сошлись).")
                return
            res_df = pd.DataFrame(results)
            pvals = res_df['p_value'].values
            res_df['q_value'] = false_discovery_control(pvals, method='bh')
            res_df['significant'] = res_df['q_value'] < self.fdr_threshold.get()
            self.results_df = res_df
            self.main_app.root.after(0, self._display_table, res_df)
            self.main_app.root.after(0, self._plot_lmm_results, res_df)
            self.save_csv_btn.config(state=tk.NORMAL)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.diag_btn.config(state=tk.NORMAL)
            self.bootstrap_btn.config(state=tk.NORMAL)
            # Заполняем комбобокс для диагностики
            features = sorted(res_df['feature'].unique())
            self.diag_feature_combo['values'] = features
            if features:
                self.diag_feature_combo.set(features[0])
            n_sign = res_df['significant'].sum()
            self.log(f"LMM завершён. Значимых признаков: {n_sign} (q<{self.fdr_threshold.get()})")
        except Exception as e:
            self.log(f"Ошибка: {e}")
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.main_app.set_progress(0)

    def _fit_lmm_model(self, df, col, feature_name):
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
            beta = result.params.get('has_apnea[T.True]', result.params.get('has_apnea', None))
            pval = result.pvalues.get('has_apnea[T.True]', result.pvalues.get('has_apnea', None))
            if beta is None or pval is None:
                return None
            ci = result.conf_int().loc['has_apnea[T.True]' if 'has_apnea[T.True]' in result.params else 'has_apnea']
            return {
                'feature': feature_name,
                'beta': beta,
                'p_value': pval,
                'ci_low': ci[0],
                'ci_high': ci[1],
                'n_obs': len(sub)
            }
        except Exception as e:
            self.log(f"Модель не сошлась для {feature_name}: {e}")
            return None

    def _get_covariates(self):
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            return None
        needed = ['study_id', 'age_at_study', 'gender', 'bmi']
        if not all(c in filtered_df.columns for c in needed):
            return None
        cov = filtered_df[needed].drop_duplicates(subset=['study_id'])
        cov['gender_code'] = (cov['gender'] == 'M').astype(int)
        return cov

    # ------------------------------------------------------------
    # Отображение таблицы и графиков
    # ------------------------------------------------------------
    def _display_table(self, df):
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

    def _plot_lmm_results(self, res_df):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        if res_df.empty:
            return

        fig = Figure(figsize=(12, 8), dpi=100)
        # Forest plot (топ-20 значимых)
        ax1 = fig.add_subplot(2, 1, 1)
        sign = res_df[res_df['significant']].copy()
        if sign.empty:
            ax1.text(0.5, 0.5, f"Нет значимых результатов (q < {self.fdr_threshold.get()})",
                     transform=ax1.transAxes, ha='center')
        else:
            top20 = sign.nsmallest(20, 'q_value')
            y_pos = np.arange(len(top20))
            labels = top20['feature']
            ax1.errorbar(top20['beta'], y_pos,
                         xerr=[top20['beta'] - top20['ci_low'], top20['ci_high'] - top20['beta']],
                         fmt='o', capsize=5, color='blue', ecolor='gray')
            ax1.axvline(x=0, linestyle='--', color='gray')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.set_xlabel('Beta коэффициент (эффект апноэ)')
            ax1.set_title(f'Топ-20 значимых признаков (FDR < {self.fdr_threshold.get()})')
            ax1.grid(True, alpha=0.3)

        # Volcano plot
        ax2 = fig.add_subplot(2, 1, 2)
        colors = np.where(res_df['significant'], 'red', 'gray')
        ax2.scatter(res_df['beta'], -np.log10(res_df['p_value']), c=colors, alpha=0.6, s=20)
        if not sign.empty:
            top10 = sign.nsmallest(10, 'q_value')
            for _, row in top10.iterrows():
                ax2.annotate(row['feature'], (row['beta'], -np.log10(row['p_value'])),
                             textcoords="offset points", xytext=(5,5), ha='left', fontsize=7,
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

    # ------------------------------------------------------------
    # Диагностика остатков
    # ------------------------------------------------------------
    def run_diagnostics(self):
        if self.results_df is None:
            messagebox.showwarning("Нет модели", "Сначала выполните LMM анализ.")
            return
        feature = self.diag_feature_combo.get()
        if not feature:
            messagebox.showwarning("Выбор признака", "Выберите признак для диагностики.")
            return
        # Находим соответствующую строку в results_df
        row = self.results_df[self.results_df['feature'] == feature]
        if row.empty:
            self.log(f"Признак {feature} не найден в результатах.")
            return
        # Загружаем эпохи заново или используем сохранённые
        if self.epochs_df is None:
            self.log("Нет загруженных эпох.")
            return
        # Определяем, какой столбец соответствует этому признаку
        # Для DFA: признак имеет вид "DFA C3" -> столбец "C3_dfa"
        # Для когерентности: "Coh F3-C3 delta" -> столбец "F3C3_coh_delta"
        if self.analysis_mode.get() == 'lmm_dfa':
            channel = feature.split()[1]
            col = f"{channel}_dfa"
        else:
            # feature: "Coh F3-C3 delta"
            parts = feature.split()
            pair = parts[1]  # "F3-C3"
            band = parts[2]  # "delta"
            a, b = pair.split('-')
            col = f"{a}{b}_coh_{band}"
        if col not in self.epochs_df.columns:
            self.log(f"Столбец {col} не найден в данных.")
            return
        df = self.epochs_df.copy()
        df['has_apnea'] = df['has_apnea'].astype(bool)
        df['patient_id'] = df['patient_id'].astype(int)
        if 'epoch_stage' in df.columns:
            df['epoch_stage'] = df['epoch_stage'].astype('category')
        if self.include_covariates.get():
            cov_df = self._get_covariates()
            if cov_df is not None:
                df = df.merge(cov_df[['study_id', 'age_at_study', 'gender_code', 'bmi']], on='study_id', how='left')
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
            fitted = result.fittedvalues
            resid = result.resid
            shapiro_p = None
            if len(resid) <= 5000:
                _, shapiro_p = shapiro(resid)
            # Breusch-Pagan
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
            self.diag_feature_name = feature
            self.save_diag_btn.config(state=tk.NORMAL)
            # Запоминаем для отчёта
            self.all_diagnostics.append({
                'feature': feature,
                'shapiro_p': shapiro_p,
                'bp_p': bp_p,
                'n': len(resid)
            })
            self.log(f"Диагностика завершена. Shapiro p={shapiro_p}, BP p={bp_p:.4f}")
            self.notebook.select(self.tab_diag)
        except Exception as e:
            self.log(f"Ошибка диагностики: {e}")

    def _show_diagnostic_plot(self, fig):
        for widget in self.diag_plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.diag_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_diagnostics_csv(self):
        if not hasattr(self, 'last_diagnostics'):
            messagebox.showwarning("Нет данных", "Сначала выполните диагностику для выбранного признака.")
            return
        diag = self.last_diagnostics
        df = pd.DataFrame({'fitted': diag['fitted'], 'residuals': diag['residuals']})
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            df.to_csv(path, index=False)
            self.log(f"Диагностика сохранена в {path}")

    # ------------------------------------------------------------
    # Блок-бутстрап
    # ------------------------------------------------------------
    def run_bootstrap(self):
        if self.diag_model is None:
            messagebox.showwarning("Нет модели", "Сначала выполните диагностику для выбранного признака.")
            return
        messagebox.showinfo("Бутстрап", "Будет выполнено 1000 итераций (блок-бутстрап по пациентам).\nЭто может занять 1-2 минуты.")
        threading.Thread(target=self._bootstrap_thread, daemon=True).start()

    def _bootstrap_thread(self):
        feature = self.diag_feature_name
        col = self.last_diagnostics['col']
        df = self.epochs_df.copy()
        df['has_apnea'] = df['has_apnea'].astype(bool)
        df['patient_id'] = df['patient_id'].astype(int)
        if 'epoch_stage' in df.columns:
            df['epoch_stage'] = df['epoch_stage'].astype('category')
        if self.include_covariates.get():
            cov_df = self._get_covariates()
            if cov_df is not None:
                df = df.merge(cov_df[['study_id', 'age_at_study', 'gender_code', 'bmi']], on='study_id', how='left')
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
            if (i+1) % 100 == 0:
                self.log(f"Бутстрап: {i+1}/{n_iter} итераций")
        if betas:
            ci_low = np.percentile(betas, 2.5)
            ci_high = np.percentile(betas, 97.5)
            p_bootstrap = (np.sum(np.abs(betas) < 1e-8) * 2) / len(betas)
            self.bootstrap_results = {
                'feature': feature,
                'beta': np.mean(betas),
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_bootstrap': p_bootstrap,
                'n_iter': len(betas)
            }
            # Обновляем таблицу
            for row in self.bootstrap_tree.get_children():
                self.bootstrap_tree.delete(row)
            self.bootstrap_tree.insert('', 'end', values=(
                feature,
                f"{self.bootstrap_results['beta']:.4f}",
                f"{ci_low:.4f}",
                f"{ci_high:.4f}",
                f"{p_bootstrap:.4f}",
                "Да" if (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0) else "Нет"
            ))
            # Гистограмма
            fig = Figure(figsize=(8,5))
            ax = fig.add_subplot(111)
            ax.hist(betas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=ci_low, color='red', linestyle='--', label=f'2.5%: {ci_low:.2f}')
            ax.axvline(x=ci_high, color='red', linestyle='--', label=f'97.5%: {ci_high:.2f}')
            ax.axvline(x=np.mean(betas), color='green', linestyle='-', label=f'Mean: {np.mean(betas):.2f}')
            ax.set_xlabel('Beta coefficient')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Bootstrap distribution of beta for {feature}')
            ax.legend()
            self.main_app.root.after(0, self._show_bootstrap_plot, fig)
            self.log(f"Бутстрап CI для beta: [{ci_low:.4f}, {ci_high:.4f}], p≈{p_bootstrap:.4f}")
        else:
            self.log("Бутстрап не дал результатов.")

    def _show_bootstrap_plot(self, fig):
        # Очищаем область бутстрапа, кроме дерева
        for widget in self.bootstrap_tree_frame.winfo_children():
            if widget not in (self.bootstrap_tree, self.bootstrap_tree.master.winfo_children()):
                widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.bootstrap_tree_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.notebook.select(self.tab_bootstrap)

    # ------------------------------------------------------------
    # Генерация HTML-отчёта
    # ------------------------------------------------------------
    def generate_report(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Сначала выполните LMM анализ.")
            return
        # Подготовка данных
        sign = self.results_df[self.results_df['significant']]
        n_sign = len(sign)
        mode = "DFA" if self.analysis_mode.get() == 'lmm_dfa' else "когерентности"
        # График LMM в base64
        buf = io.BytesIO()
        self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'

        # Таблица значимых признаков
        if not sign.empty:
            table_html = sign[['feature', 'beta', 'p_value', 'q_value', 'ci_low', 'ci_high']].to_html(index=False, float_format="%.4f")
        else:
            table_html = "<p>Нет значимых признаков.</p>"

        # Диагностика
        diag_html = "<h2>Диагностика остатков</h2>"
        if self.all_diagnostics:
            diag_html += "<table border='1'><tr><th>Признак</th><th>n</th><th>Shapiro-Wilk p</th><th>Breusch-Pagan p</th><th>Заключение</th></tr>"
            for d in self.all_diagnostics:
                shapiro_ok = d['shapiro_p'] > 0.05 if d['shapiro_p'] is not None else "n>5000"
                bp_ok = d['bp_p'] > 0.05
                conclusion = []
                if d['shapiro_p'] is not None and d['shapiro_p'] <= 0.05:
                    conclusion.append("❗ остатки не нормальны")
                elif d['shapiro_p'] is None:
                    conclusion.append("⚠️ выборка >5000")
                else:
                    conclusion.append("✅ нормальность")
                if d['bp_p'] <= 0.05:
                    conclusion.append("❗ гетероскедастичность")
                else:
                    conclusion.append("✅ гомоскедастичность")
                diag_html += f"<tr><td>{d['feature']}</td><td>{d['n']}</td><td>{d['shapiro_p']:.4f if d['shapiro_p'] else '>5000'}</td><td>{d['bp_p']:.4f}</td><td>{', '.join(conclusion)}</td></tr>"
            diag_html += "</table>"
        else:
            diag_html += "<p>Диагностика не выполнялась.</p>"

        # Бутстрап
        boot_html = "<h2>Бутстрап-проверка</h2>"
        if self.bootstrap_results:
            boot_html += f"""
            <p><strong>Признак:</strong> {self.bootstrap_results['feature']}</p>
            <p><strong>95% доверительный интервал для beta:</strong> [{self.bootstrap_results['ci_low']:.4f}, {self.bootstrap_results['ci_high']:.4f}]</p>
            <p><strong>Бутстрап p-value:</strong> {self.bootstrap_results['p_bootstrap']:.4f}</p>
            <p><strong>Количество успешных итераций:</strong> {self.bootstrap_results['n_iter']}</p>
            """
        else:
            boot_html += "<p>Бутстрап не выполнялся.</p>"

        params = f"""
        <p><strong>Тип анализа:</strong> {mode}</p>
        <p><strong>Тип набора эпох:</strong> {self.data_type.get()} (1=тонический,2=все,3=фильтр по положению)</p>
        <p><strong>Ковариаты:</strong> {'включены' if self.include_covariates.get() else 'не включены'}</p>
        <p><strong>Стадия сна:</strong> {'включена' if self.include_stage.get() else 'не включена'}</p>
        <p><strong>FDR порог:</strong> q = {self.fdr_threshold.get()}</p>
        """

        html = f"""<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>LMM анализ DFA и когерентности</title>
        <style>
            body {{ font-family: Arial; margin:20px; }}
            table {{ border-collapse: collapse; width:100%; margin-bottom:20px; }}
            th, td {{ border:1px solid #ddd; padding:8px; text-align:left; }}
            th {{ background:#f2f2f2; }}
            .plot {{ margin:20px 0; text-align:center; }}
        </style>
        </head>
        <body>
        <h1>Отчёт о линейных смешанных моделях (LMM) для {mode}</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {params}
        <h2>Графики LMM</h2>
        {plot_html}
        <h2>Значимые признаки (всего {n_sign})</h2>
        {table_html}
        {diag_html}
        {boot_html}
        <p><em>Примечание:</em> При нарушении нормальности остатков или гетероскедастичности рекомендуется использовать бутстрап-доверительные интервалы.</p>
        </body></html>
        """
        fd, path = tempfile.mkstemp(suffix='.html', prefix='dfa_coh_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт в браузере: {path}")

    # ------------------------------------------------------------
    # Сохранение результатов
    # ------------------------------------------------------------
    def save_csv(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Нет данных", "Нет результатов для сохранения.")
            return
        from tkinter import filedialog
        fname = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if fname:
            self.results_df.to_csv(fname, index=False, encoding='utf-8-sig')
            self.log(f"Сохранено в {fname}")

    def save_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "График не построен.")
            return
        from tkinter import filedialog
        fname = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if fname:
            self.current_figure.savefig(fname, dpi=150, bbox_inches='tight')
            self.log(f"График сохранён в {fname}")
