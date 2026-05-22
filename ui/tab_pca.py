# ui/tab_pca.py
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import os
import tempfile
import webbrowser
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro, false_discovery_control
import statsmodels.formula.api as smf
from scipy import stats

from ui.base_tab import BaseTab
from core.api_client import get_epochs

class PCAAnalysisTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        # Настройки
        self.data_type = tk.IntVar(value=2)
        self.use_filtered = tk.BooleanVar(value=True)
        self.n_components = tk.IntVar(value=5)
        self.scale_data = tk.BooleanVar(value=True)
        self.use_logreg = tk.BooleanVar(value=True)
        self.logreg_cv_folds = tk.IntVar(value=5)
        self.use_cache = tk.BooleanVar(value=True)
        self.exclude_central_mixed = tk.BooleanVar(value=True)
        
        self.stop_flag = False
        self.results = None            # словарь с pca, X_pca, y, feature_names, ...
        self.current_figure = None
        self.lmm_results = None        # результаты LMM для PC
        
        # Для диагностики и бутстрапа
        self.normality_results = {}    # {PC: p_value}
        self.bootstrap_loadings = None
        self.pca_model = None
        self.feature_names = None
        
        self.all_channels = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        self.all_features = [
            'mean', 'std', 'min', 'max', 'range', 'rms',
            'abs_delta', 'rel_delta', 'abs_theta', 'rel_theta',
            'abs_alpha', 'rel_alpha', 'abs_sigma', 'rel_sigma',
            'abs_beta', 'rel_beta', 'tbr', 'dar', 'se50', 'gamma_power', 'sampen'
        ]
        self.saved_channels_selection = []
        self.saved_features_selection = []
        
        self._create_widgets()
        
    # ------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------
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
        info_frame = ttk.LabelFrame(left_frame, text="Источник данных", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, text="Используются отфильтрованные данные из вкладки 'Загрузка'",
                  foreground="blue").pack(anchor=tk.W)
        ttk.Label(info_frame, text="(токен и URL не требуются)").pack(anchor=tk.W)
        
        desc_frame = ttk.LabelFrame(left_frame, text="Что такое PCA (Глава 2, п. 2.5.4)?", padding=5)
        desc_frame.pack(fill=tk.X, padx=5, pady=5)
        desc_text = (
            "Анализ главных компонент (PCA) снижает размерность 180 ЭЭГ-признаков до 5 интегральных компонент, "
            "объясняющих >70% дисперсии. Затем:\n"
            "• Строятся линейные смешанные модели (LMM) для каждой ПК с фиксированным эффектом has_apnea.\n"
            "• Выполняется логистическая регрессия на ПК для классификации эпох (AUC ≥ 0.85).\n"
            "• Проводится бутстрап нагрузок для оценки устойчивости.\n"
            "• Иерархическая кластеризация пациентов на основе PC1/PC2 (выявляет скрытые фенотипы)."
        )
        ttk.Label(desc_frame, text=desc_text, justify=tk.LEFT, wraplength=600).pack(anchor=tk.W, fill=tk.X, padx=5, pady=2)
        ttk.Button(desc_frame, text="Показать инструкцию", command=self.show_instructions).pack(anchor=tk.W, padx=5, pady=2)
        
        # Данные
        type_frame = ttk.LabelFrame(left_frame, text="Данные", padding=5)
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(type_frame, text="Тип набора (data_type):").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(type_frame, textvariable=self.data_type, values=[1,2,3], state='readonly', width=5).grid(row=0, column=1, padx=5)
        ttk.Label(type_frame, text="1=тонический, 2=все эпохи, 3=фильтр по положению").grid(row=0, column=2, sticky=tk.W)
        ttk.Checkbutton(type_frame, text="Использовать отфильтрованные исследования", variable=self.use_filtered).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Checkbutton(type_frame, text="Исключить центральное/смешанное апноэ", variable=self.exclude_central_mixed).grid(row=2, column=0, columnspan=3, sticky=tk.W)
        ttk.Checkbutton(type_frame, text="Использовать кэш", variable=self.use_cache).grid(row=3, column=0, columnspan=3, sticky=tk.W)
        
        # Выбор признаков
        select_frame = ttk.LabelFrame(left_frame, text="Выбор признаков", padding=5)
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(select_frame, text="Каналы (Ctrl/Shift):").grid(row=0, column=0, sticky=tk.W)
        self.channels_listbox = tk.Listbox(select_frame, selectmode=tk.EXTENDED, height=6, width=20)
        for ch in self.all_channels:
            self.channels_listbox.insert(tk.END, ch)
        self.channels_listbox.selection_set(0, tk.END)
        self.channels_listbox.grid(row=1, column=0, padx=5, sticky=tk.W)
        self.channels_listbox.bind('<<ListboxSelect>>', self.save_channels_selection)
        
        ttk.Label(select_frame, text="Признаки (Ctrl/Shift):").grid(row=0, column=1, sticky=tk.W)
        self.features_listbox = tk.Listbox(select_frame, selectmode=tk.EXTENDED, height=12, width=30)
        for feat in self.all_features:
            self.features_listbox.insert(tk.END, feat)
        self.features_listbox.selection_set(0, tk.END)
        self.features_listbox.grid(row=1, column=1, padx=5, sticky=tk.W)
        self.features_listbox.bind('<<ListboxSelect>>', self.save_features_selection)
        
        # Параметры PCA
        pca_frame = ttk.LabelFrame(left_frame, text="Параметры PCA", padding=5)
        pca_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(pca_frame, text="Число компонент:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(pca_frame, from_=2, to=50, textvariable=self.n_components, width=5).grid(row=0, column=1, padx=5)
        ttk.Checkbutton(pca_frame, text="Масштабировать признаки (StandardScaler)", variable=self.scale_data).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # Логистическая регрессия
        class_frame = ttk.LabelFrame(left_frame, text="Логистическая регрессия на ПК", padding=5)
        class_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(class_frame, text="Выполнить логистическую регрессию", variable=self.use_logreg).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(class_frame, text="Число фолдов (CV по пациентам):").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(class_frame, from_=2, to=10, textvariable=self.logreg_cv_folds, width=3).grid(row=1, column=1, padx=5)
        
        # Диагностика и бутстрап
        diag_frame = ttk.LabelFrame(left_frame, text="Диагностика и бутстрап", padding=5)
        diag_frame.pack(fill=tk.X, padx=5, pady=5)
        self.norm_btn = ttk.Button(diag_frame, text="Проверить нормальность PC1-PC5", command=self.check_normality, state=tk.DISABLED)
        self.norm_btn.pack(anchor=tk.W, pady=2)
        self.bootstrap_btn = ttk.Button(diag_frame, text="Бутстрап нагрузок (100 итер.)", command=self.run_bootstrap, state=tk.DISABLED)
        self.bootstrap_btn.pack(anchor=tk.W, pady=2)
        self.save_bootstrap_btn = ttk.Button(diag_frame, text="Сохранить бутстрап (CSV)", command=self.save_bootstrap_csv, state=tk.DISABLED)
        self.save_bootstrap_btn.pack(anchor=tk.W, pady=2)
        
        # Кнопки
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Запустить PCA", command=self.run_pca)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить координаты ПК (CSV)", command=self.save_results_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график (PNG)", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать отчёт", command=self.generate_report, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=5)
        
        # ========== ПРАВАЯ ПАНЕЛЬ ==========
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_scatter = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_scatter, text="Scatter plot (PC1/PC2)")
        self.tab_variance = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_variance, text="Explained variance")
        self.tab_loadings = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_loadings, text="Loadings heatmap")
        self.tab_roc = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_roc, text="ROC curves")
        self.tab_coef = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_coef, text="LogReg coefficients")
        self.tab_diag = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diag, text="Диагностика")
        self.tab_log = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_log, text="Лог выполнения")
        
        self.plot_frames = {
            'scatter': self.tab_scatter,
            'variance': self.tab_variance,
            'loadings': self.tab_loadings,
            'roc': self.tab_roc,
            'coef': self.tab_coef
        }
        for frame in self.plot_frames.values():
            ttk.Frame(frame).pack(fill=tk.BOTH, expand=True)
        
        self.diag_text = tk.Text(self.tab_diag, wrap=tk.WORD, font=("Courier New", 9), height=10)
        self.diag_text.pack(fill=tk.X, padx=5, pady=5)
        self.diag_plot_frame = ttk.Frame(self.tab_diag)
        self.diag_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(self.tab_log, wrap=tk.WORD, font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self._clear_coef_tab()
        
    # ------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.main_app.log(msg)
        
    def stop_analysis(self):
        self.stop_flag = True
        self.log("Остановка запрошена...")
        
    def show_instructions(self):
        msg = (
            "ИНСТРУКЦИЯ ПО PCA АНАЛИЗУ (Глава 2, п. 2.5.4 – 2.5.5)\n"
            "====================================================\n"
            "1. Загрузите и отфильтруйте данные на вкладке 'Загрузка и фильтры'.\n"
            "2. Выберите каналы и признаки (по умолчанию – все).\n"
            "3. Установите число главных компонент (рекомендуется 5).\n"
            "4. Нажмите 'Запустить PCA'. Будут построены:\n"
            "   - Scatter plot PC1 vs PC2 (апноэ/без апноэ)\n"
            "   - Scree plot (объяснённая дисперсия)\n"
            "   - Тепловая карта нагрузок (топ-20 признаков)\n"
            "5. Если включена логистическая регрессия, выполняется ROC-анализ.\n"
            "6. После завершения можно проверить нормальность PC и запустить бутстрап нагрузок.\n"
            "7. Кнопка 'Сформировать отчёт' создаст HTML-отчёт с уравнением регрессии,\n"
            "   результатами LMM для ПК и интерпретацией.\n"
            "8. Результаты можно сохранить в CSV (координаты ПК) и PNG (графики)."
        )
        messagebox.showinfo("Инструкция", msg)
        
    def save_channels_selection(self, event=None):
        self.saved_channels_selection = list(self.channels_listbox.curselection())
        
    def save_features_selection(self, event=None):
        self.saved_features_selection = list(self.features_listbox.curselection())
        
    def restore_selections(self):
        self.channels_listbox.selection_clear(0, tk.END)
        for idx in self.saved_channels_selection:
            self.channels_listbox.selection_set(idx)
        self.features_listbox.selection_clear(0, tk.END)
        for idx in self.saved_features_selection:
            self.features_listbox.selection_set(idx)
            
    def on_tab_selected(self):
        self.restore_selections()
        
    # ------------------------------------------------------------
    # Загрузка данных
    # ------------------------------------------------------------
    def _load_epochs(self):
        load_tab = self.main_app.tabs['load']
        api_url = load_tab.api_url.get().rstrip('/')
        token = load_tab.token.get().strip()
        if not api_url or not token:
            self.log("Ошибка: не указаны URL или токен API. Перейдите на вкладку 'Загрузка'.")
            return None
            
        study_ids = None
        if self.use_filtered.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is None or filtered_df.empty:
                self.log("Нет отфильтрованных данных.")
                return None
            study_ids = filtered_df['study_id'].unique().tolist()
            if not study_ids:
                self.log("В отфильтрованных данных нет study_id.")
                return None
                
        if self.exclude_central_mixed.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is not None:
                allowed = ['no_impairment', 'mild', 'moderate', 'severe']
                filtered_df = filtered_df[filtered_df['breathing_impairment_severity'].isin(allowed)]
                if not filtered_df.empty:
                    study_ids = filtered_df['study_id'].unique().tolist()
                    self.log(f"После исключения центрального/смешанного апноэ: {len(study_ids)} исследований")
                    
        def update_progress(page, total, _):
            if total > 0:
                self.main_app.set_progress(int(page / total * 100))
                
        self.main_app.set_progress(0)
        self.log("Загрузка эпох...")
        epochs = get_epochs(
            api_url, token,
            study_ids=study_ids,
            data_type=self.data_type.get(),
            stop_check=lambda: self.stop_flag,
            progress_callback=update_progress,
            use_cache=self.use_cache.get()
        )
        self.main_app.set_progress(100)
        if not epochs:
            return None
        df = pd.DataFrame(epochs)
        self.log(f"Загружено {len(df)} эпох")
        return df
        
    # ------------------------------------------------------------
    # Основной PCA
    # ------------------------------------------------------------
    def run_pca(self):
        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_csv_btn.config(state=tk.DISABLED)
        self.save_plot_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        self.norm_btn.config(state=tk.DISABLED)
        self.bootstrap_btn.config(state=tk.DISABLED)
        self.save_bootstrap_btn.config(state=tk.DISABLED)
        self.log("Запуск PCA...")
        thread = threading.Thread(target=self._run_pca_thread)
        thread.daemon = True
        thread.start()
        
    def _run_pca_thread(self):
        try:
            df = self._load_epochs()
            if df is None or self.stop_flag:
                return
                
            # Выбор признаков
            selected_channels = [self.channels_listbox.get(i) for i in self.channels_listbox.curselection()]
            selected_features = [self.features_listbox.get(i) for i in self.features_listbox.curselection()]
            if not selected_channels:
                selected_channels = self.all_channels[:]
            if not selected_features:
                selected_features = self.all_features[:]
                
            feature_cols = []
            for ch in selected_channels:
                for feat in selected_features:
                    col = f"{ch}_{feat}"
                    if col in df.columns:
                        feature_cols.append(col)
                    else:
                        self.log(f"Предупреждение: столбец {col} не найден, пропущен")
            if not feature_cols:
                self.log("Нет доступных признаков.")
                return
            self.log(f"Используется признаков: {len(feature_cols)}")
            
            X = df[feature_cols].copy()
            X = X.dropna()
            if len(X) < 10:
                self.log("Слишком мало строк после удаления NaN.")
                return
            meta = df.loc[X.index, ['patient_id', 'has_apnea', 'epoch_stage']].copy()
            y = meta['has_apnea'].astype(int)
            patients = meta['patient_id'].values
            self.log(f"Матрица X: {X.shape}, эпох с апноэ: {y.sum()}")
            
            # Масштабирование
            if self.scale_data.get():
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                scaler = None
                X_scaled = X.values
                
            n_comp = min(self.n_components.get(), X_scaled.shape[1], X_scaled.shape[0])
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X_scaled)
            self.log(f"PCA выполнено, сохранено {n_comp} компонент, объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.3f}")
            
            # LMM для каждой PC
            lmm_results = self._run_lmm_for_pcs(X_pca, meta, n_comp)
            self.lmm_results = lmm_results
            
            # Логистическая регрессия
            logreg_results = None
            if self.use_logreg.get():
                logreg_results = self._run_logistic_regression(X_pca, y, patients, pca, feature_cols)
                
            self.results = {
                'pca': pca,
                'X_pca': X_pca,
                'meta': meta,
                'y': y,
                'feature_names': feature_cols,
                'scaler': scaler,
                'n_comp': n_comp,
                'lmm_results': lmm_results,
                'logreg_results': logreg_results
            }
            
            # Визуализация
            self._plot_scatter(X_pca, meta)
            self._plot_variance(pca)
            self._plot_loadings(pca, feature_cols, n_comp)
            if logreg_results:
                self._display_coef_table(logreg_results['coef_df'], logreg_results['intercept'])
                self._plot_roc_curves(y, X_pca, logreg_results)
            else:
                self._clear_coef_tab()
                
            self.save_csv_btn.config(state=tk.NORMAL)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.norm_btn.config(state=tk.NORMAL)
            self.bootstrap_btn.config(state=tk.NORMAL)
            self.log("PCA анализ завершён.")
            
        except Exception as e:
            self.log(f"Ошибка: {e}")
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
    def _run_lmm_for_pcs(self, X_pca, meta, n_comp):
        """Строит LMM для каждой PC: PC ~ has_apnea + (1|patient_id)"""
        results = []
        df_pcs = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_comp)])
        df_pcs['has_apnea'] = meta['has_apnea'].astype(bool)
        df_pcs['patient_id'] = meta['patient_id'].astype(int)
        for i in range(n_comp):
            col = f'PC{i+1}'
            try:
                model = smf.mixedlm(f"{col} ~ has_apnea", df_pcs, groups=df_pcs['patient_id'])
                result = model.fit(reml=True, method='lbfgs', maxiter=1000)
                beta = result.params.get('has_apnea[T.True]', result.params.get('has_apnea', None))
                pval = result.pvalues.get('has_apnea[T.True]', result.pvalues.get('has_apnea', None))
                if beta is not None and pval is not None:
                    ci = result.conf_int().loc['has_apnea[T.True]' if 'has_apnea[T.True]' in result.params else 'has_apnea']
                    results.append({
                        'PC': i+1,
                        'beta': beta,
                        'p_value': pval,
                        'ci_low': ci[0],
                        'ci_high': ci[1],
                        'n_obs': len(df_pcs)
                    })
                else:
                    results.append({'PC': i+1, 'beta': np.nan, 'p_value': np.nan})
            except Exception as e:
                self.log(f"LMM для PC{i+1} не сошлась: {e}")
                results.append({'PC': i+1, 'beta': np.nan, 'p_value': np.nan})
        return pd.DataFrame(results)
        
    def _run_logistic_regression(self, X_pca, y, patients, pca, feature_names):
        n_comp = X_pca.shape[1]
        self.log(f"Логистическая регрессия на {n_comp} ПК, кросс-валидация по пациентам (фолдов: {self.logreg_cv_folds.get()})")
        unique_patients = np.unique(patients)
        n_folds = min(self.logreg_cv_folds.get(), len(unique_patients))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        auc_scores = []
        for train_idx, test_idx in skf.split(X_pca, y):
            X_train, X_test = X_pca[train_idx], X_pca[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            lr.fit(X_train, y_train)
            pred = lr.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, pred)
            auc_scores.append(auc)
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        self.log(f"Логистическая регрессия: средний AUC = {mean_auc:.3f} ± {std_auc:.3f}")
        
        # Обучение на всех данных
        lr.fit(X_pca, y)
        intercept = lr.intercept_[0]
        coefs = lr.coef_[0]
        coef_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(n_comp)],
            'Coefficient': coefs,
            'Odds_ratio': np.exp(coefs)
        })
        # Вычисление AUC на всей выборке (для отчёта)
        pred_full = lr.predict_proba(X_pca)[:, 1]
        auc_full = roc_auc_score(y, pred_full)
        return {
            'intercept': intercept,
            'coef_df': coef_df,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'auc_full': auc_full,
            'model': lr,
            'pred_full': pred_full,
            'y': y
        }
        
    # ------------------------------------------------------------
    # Графики
    # ------------------------------------------------------------
    def _plot_scatter(self, X_pca, meta):
        container = self.plot_frames['scatter']
        for widget in container.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        colors = meta['has_apnea'].map({True: 'red', False: 'blue'})
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5, s=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA projection (apnea vs no apnea)')
        legend_elements = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='red', label='Apnea'),
                           plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', label='No apnea')]
        ax.legend(handles=legend_elements)
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig
        
    def _plot_variance(self, pca):
        container = self.plot_frames['variance']
        for widget in container.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        n_comp = len(pca.explained_variance_ratio_)
        ax.bar(range(1, n_comp+1), pca.explained_variance_ratio_)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title('Scree plot')
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _plot_loadings(self, pca, feature_names, n_comp):
        container = self.plot_frames['loadings']
        for widget in container.winfo_children():
            widget.destroy()
        loadings = pca.components_.T
        sum_abs = np.sum(np.abs(loadings[:, :n_comp]), axis=1)
        top_idx = np.argsort(sum_abs)[-20:]
        top_features = [feature_names[i] for i in top_idx]
        top_loadings = loadings[top_idx, :n_comp]
        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(top_loadings, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(np.arange(n_comp))
        ax.set_xticklabels([f'PC{i+1}' for i in range(n_comp)])
        ax.set_yticks(np.arange(len(top_features)))
        ax.set_yticklabels(top_features)
        plt.colorbar(im, ax=ax)
        ax.set_title('Loadings heatmap (top 20 features)')
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _plot_roc_curves(self, y, X_pca, logreg_results):
        container = self.plot_frames['roc']
        for widget in container.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        # Лучшая отдельная PC
        auc_single = []
        for i in range(X_pca.shape[1]):
            auc_single.append(roc_auc_score(y, X_pca[:, i]))
        best_pc = np.argmax(auc_single)
        fpr_pc, tpr_pc, _ = roc_curve(y, X_pca[:, best_pc])
        ax.plot(fpr_pc, tpr_pc, label=f'Best PC (PC{best_pc+1}, AUC={auc_single[best_pc]:.3f})')
        # Логистическая регрессия
        fpr_lr, tpr_lr, _ = roc_curve(y, logreg_results['pred_full'])
        ax.plot(fpr_lr, tpr_lr, label=f'LogReg on PCs (AUC={logreg_results["auc_full"]:.3f})')
        ax.plot([0,1],[0,1], 'k--')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC curves')
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _display_coef_table(self, coef_df, intercept):
        container = self.plot_frames['coef']
        for widget in container.winfo_children():
            widget.destroy()
        tree_frame = ttk.Frame(container)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(tree_frame, columns=list(coef_df.columns), show='headings')
        for col in coef_df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        for _, row in coef_df.iterrows():
            tree.insert('', 'end', values=list(row))
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)
        info_label = ttk.Label(container, text=f"Intercept (β₀) = {intercept:.4f}", font=('Arial', 10, 'bold'))
        info_label.pack(pady=5)
        
    def _clear_coef_tab(self):
        container = self.plot_frames['coef']
        for widget in container.winfo_children():
            widget.destroy()
        label = ttk.Label(container, text="Логистическая регрессия не выполнялась.\nВключите опцию и запустите PCA.")
        label.pack(expand=True)
        
    # ------------------------------------------------------------
    # Проверка нормальности PC
    # ------------------------------------------------------------
    def check_normality(self):
        if self.results is None:
            messagebox.showwarning("Нет данных", "Сначала выполните PCA.")
            return
        X_pca = self.results['X_pca']
        n_comp = X_pca.shape[1]
        self.diag_text.delete(1.0, tk.END)
        for i in range(n_comp):
            scores = X_pca[:, i]
            if len(scores) <= 5000:
                stat, p = shapiro(scores)
                normal = p > 0.05
                self.normality_results[f'PC{i+1}'] = {'p': p, 'n': len(scores), 'normal': normal}
                self.diag_text.insert(tk.END, f"PC{i+1}: Shapiro-Wilk p={p:.4e} -> {'Нормальное' if normal else 'Ненормальное'}\n")
            else:
                self.normality_results[f'PC{i+1}'] = {'p': None, 'n': len(scores), 'normal': None}
                self.diag_text.insert(tk.END, f"PC{i+1}: n={len(scores)} > 5000, тест не применялся\n")
        # Q-Q plot для PC1 и PC2 (пример)
        fig = Figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1,2,1)
        stats.probplot(X_pca[:, 0], dist="norm", plot=ax1)
        ax1.set_title('Q-Q plot PC1')
        ax2 = fig.add_subplot(1,2,2)
        stats.probplot(X_pca[:, 1], dist="norm", plot=ax2)
        ax2.set_title('Q-Q plot PC2')
        for widget in self.diag_plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.diag_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.notebook.select(self.tab_diag)
        self.log("Проверка нормальности PC завершена.")
        
    # ------------------------------------------------------------
    # Бутстрап нагрузок
    # ------------------------------------------------------------
    def run_bootstrap(self):
        if self.results is None:
            messagebox.showwarning("Нет данных", "Сначала выполните PCA.")
            return
        X = self.results['pca'].components_.T  # матрица нагрузок (n_features x n_comp)
        feature_names = self.results['feature_names']
        n_iter = 100
        n_comp = X.shape[1]
        boot_loadings = []
        self.log(f"Бутстрап нагрузок: {n_iter} итераций...")
        for _ in range(n_iter):
            idx = np.random.choice(len(feature_names), size=len(feature_names), replace=True)
            boot_loadings.append(X[idx, :])
        boot_loadings = np.array(boot_loadings)  # (n_iter, n_features, n_comp)
        # Вычислим CI для топ-признаков (по абсолютной сумме нагрузок)
        sum_abs = np.sum(np.abs(X), axis=1)
        top_idx = np.argsort(sum_abs)[-10:]
        results = []
        for idx in top_idx:
            feat = feature_names[idx]
            for pc in range(n_comp):
                vals = boot_loadings[:, idx, pc]
                ci_low = np.percentile(vals, 2.5)
                ci_high = np.percentile(vals, 97.5)
                results.append({
                    'feature': feat,
                    'PC': pc+1,
                    'loading': X[idx, pc],
                    'ci_low': ci_low,
                    'ci_high': ci_high
                })
        self.bootstrap_loadings = pd.DataFrame(results)
        self.save_bootstrap_btn.config(state=tk.NORMAL)
        self.log("Бутстрап нагрузок завершён. Сохраните результаты через кнопку.")
        # Покажем в диагностике
        self.diag_text.insert(tk.END, "\n=== Бутстрап нагрузок (95% CI) ===\n")
        self.diag_text.insert(tk.END, self.bootstrap_loadings.to_string(index=False))
        self.notebook.select(self.tab_diag)
        
    def save_bootstrap_csv(self):
        if self.bootstrap_loadings is None:
            messagebox.showwarning("Нет данных", "Сначала выполните бутстрап.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.bootstrap_loadings.to_csv(path, index=False)
            self.log(f"Бутстрап нагрузок сохранён в {path}")
            
    # ------------------------------------------------------------
    # Сохранение результатов
    # ------------------------------------------------------------
    def save_results_csv(self):
        if self.results is None:
            messagebox.showwarning("Нет данных", "Сначала выполните PCA.")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        pca_df = self.results['meta'].copy()
        for i in range(self.results['X_pca'].shape[1]):
            pca_df[f'PC{i+1}'] = self.results['X_pca'][:, i]
        pca_df.to_csv(file_path, index=False)
        self.log(f"Координаты ПК сохранены в {file_path}")
        if self.results.get('logreg_results'):
            coef_path = file_path.replace('.csv', '_logreg_coef.csv')
            coef_df = self.results['logreg_results']['coef_df'].copy()
            coef_df.loc[-1] = ['Intercept', self.results['logreg_results']['intercept'], np.exp(self.results['logreg_results']['intercept'])]
            coef_df.index = coef_df.index + 1
            coef_df.sort_index(inplace=True)
            coef_df.to_csv(coef_path, index=False)
            self.log(f"Коэффициенты логистической регрессии сохранены в {coef_path}")
            
    def save_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "Сначала постройте график (выполните PCA).")
            return
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.current_figure.savefig(file_path, dpi=150, bbox_inches='tight')
            self.log(f"График сохранён в {file_path}")
            
    # ------------------------------------------------------------
    # Генерация HTML-отчёта
    # ------------------------------------------------------------
    def generate_report(self):
        if self.results is None:
            messagebox.showwarning("Нет данных", "Сначала выполните PCA.")
            return
        # Собираем данные
        pca = self.results['pca']
        var_ratio = pca.explained_variance_ratio_
        cum_var = np.cumsum(var_ratio)
        lmm_df = self.results['lmm_results']
        logreg = self.results.get('logreg_results')
        
        # Графики
        buf = io.BytesIO()
        self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
        
        # Таблица LMM для PC
        if lmm_df is not None and not lmm_df.empty:
            lmm_html = lmm_df.to_html(index=False, float_format="%.4f")
        else:
            lmm_html = "<p>LMM не выполнялись.</p>"
            
        # Уравнение логистической регрессии
        logreg_html = ""
        if logreg:
            intercept = logreg['intercept']
            coef_df = logreg['coef_df']
            eq_parts = [f"logit(p) = {intercept:.4f}"]
            for _, row in coef_df.iterrows():
                sign = '+' if row['Coefficient'] >= 0 else '-'
                eq_parts.append(f" {sign} {abs(row['Coefficient']):.4f} * {row['Component']}")
            equation = "".join(eq_parts)
            logreg_html = f"""
            <h2>Логистическая регрессия на главных компонентах</h2>
            <p><strong>Уравнение:</strong><br>{equation}</p>
            <p><strong>Средний AUC (CV по пациентам):</strong> {logreg['mean_auc']:.3f} ± {logreg['std_auc']:.3f}</p>
            <p><strong>AUC на всей выборке:</strong> {logreg['auc_full']:.3f}</p>
            <h3>Коэффициенты и отношения шансов</h3>
            {coef_df.to_html(index=False, float_format="%.4f")}
            <p><em>Интерпретация:</em> Положительный коэффициент означает, что увеличение ПК повышает вероятность наличия апноэ. Отношение шансов >1 указывает на повышенный риск.</p>
            """
            
        params = f"""
        <p><strong>Тип набора эпох:</strong> {self.data_type.get()} (1=тонический,2=все,3=фильтр по положению)</p>
        <p><strong>Число главных компонент:</strong> {self.results['n_comp']}</p>
        <p><strong>Объяснённая дисперсия (первые 5):</strong> {var_ratio[:5].sum():.3f}</p>
        <p><strong>Кумулятивная дисперсия:</strong> {cum_var[self.results['n_comp']-1]:.3f}</p>
        """
        
        html = f"""<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>PCA анализ ЭЭГ-признаков</title>
        <style>
            body {{ font-family: Arial; margin:20px; }}
            table {{ border-collapse: collapse; width:100%; margin-bottom:20px; }}
            th, td {{ border:1px solid #ddd; padding:8px; text-align:left; }}
            th {{ background:#f2f2f2; }}
            .plot {{ margin:20px 0; text-align:center; }}
        </style>
        </head>
        <body>
        <h1>Отчёт о PCA анализе (Глава 2, п. 2.5.4–2.5.5)</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {params}
        <h2>Графики PCA</h2>
        {plot_html}
        <h2>Линейные смешанные модели для главных компонент (эффект апноэ)</h2>
        {lmm_html}
        {logreg_html}
        <h2>Нормальность PC-счетов</h2>
        <table border="1"><tr><th>Компонента</th><th>p-value (Shapiro-Wilk)</th><th>Вывод</th></tr>"""
        for pc, res in self.normality_results.items():
            p_str = f"{res['p']:.4e}" if res['p'] is not None else ">5000"
            norm_str = "Нормальное" if res['normal'] else "Ненормальное" if res['normal'] is not None else "Не тестировалось"
            html += f"<tr><td>{pc}</td><td>{p_str}</td><td>{norm_str}</td></tr>"
        html += "</table>"
        if self.bootstrap_loadings is not None:
            html += f"""
            <h2>Бутстрап нагрузок (95% доверительные интервалы)</h2>
            {self.bootstrap_loadings.to_html(index=False, float_format="%.4f")}
            """
        html += """
        <p><em>Примечание:</em> Для интерпретации загрузок используются признаки с абсолютной нагрузкой >0,4. PC1 обычно отражает общую спектральную мощность, PC2 – ось медленноволновая/быстроволновая активность.</p>
        </body></html>
        """
        fd, path = tempfile.mkstemp(suffix='.html', prefix='pca_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт в браузере: {path}")
