# ui/tab_pca.py

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
from scipy.stats import false_discovery_control, chi2_contingency, f_oneway, zscore
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_curve, roc_auc_score
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ui.base_tab import BaseTab
from core.api_client import get_epochs

warnings.filterwarnings("ignore", module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning)

class PCAAnalysisTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        self.pca_data_type = tk.IntVar(value=1)
        self.lmm_data_type = tk.IntVar(value=2)
        self.n_components = tk.IntVar(value=5)
        self.fdr_threshold = tk.DoubleVar(value=0.05)
        self.include_stage = tk.BooleanVar(value=True)
        self.include_covariates = tk.BooleanVar(value=True)
        self.use_filtered = tk.BooleanVar(value=True)
        self.use_cache = tk.BooleanVar(value=True)
        self.stop_flag = False
        self.results = None
        self.current_figure = None

        self.clust_normalize = tk.BooleanVar(value=False)
        self.clust_method = tk.StringVar(value='ward')
        self.clust_metric = tk.StringVar(value='euclidean')
        self.exclude_outliers = tk.BooleanVar(value=True)
        self.outlier_threshold = tk.DoubleVar(value=3.0)
        self.clust_components_listbox = None
        self.recluster_btn = None

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

    def _create_widgets(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)
        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(paned)
        right_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        paned.add(right_frame, weight=3)

        # ---- Левая панель ----
        info_frame = ttk.LabelFrame(left_frame, text="Описание анализа", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, text="PCA + LMM + кластеризация пациентов\n(Глава 2, п. 2.5.4, 2.5.5)", justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        ttk.Button(info_frame, text="Инструкция", command=self.show_instructions).pack(anchor=tk.W, pady=2)

        settings_frame = ttk.LabelFrame(left_frame, text="Настройки", padding=5)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(settings_frame, text="Тип эпох для PCA:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(settings_frame, textvariable=self.pca_data_type, values=[1,2,3], state='readonly', width=5).grid(row=0, column=1, padx=5)
        ttk.Label(settings_frame, text="1=тонич., 2=все, 3=полож.").grid(row=0, column=2, sticky=tk.W)
        ttk.Label(settings_frame, text="Тип эпох для LMM:").grid(row=1, column=0, sticky=tk.W)
        ttk.Combobox(settings_frame, textvariable=self.lmm_data_type, values=[1,2,3], state='readonly', width=5).grid(row=1, column=1, padx=5)
        ttk.Label(settings_frame, text="должен содержать апноэ (рекоменд. 2)").grid(row=1, column=2, sticky=tk.W)
        ttk.Label(settings_frame, text="Число компонент:").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(settings_frame, from_=2, to=20, textvariable=self.n_components, width=5).grid(row=2, column=1, padx=5)
        ttk.Label(settings_frame, text="Порог FDR (q):").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(settings_frame, textvariable=self.fdr_threshold, width=6).grid(row=3, column=1, padx=5)
        ttk.Checkbutton(settings_frame, text="Включить стадию сна в LMM", variable=self.include_stage).grid(row=4, column=0, columnspan=3, sticky=tk.W)
        ttk.Checkbutton(settings_frame, text="Включить ковариаты (возраст, пол, ИМТ)", variable=self.include_covariates).grid(row=5, column=0, columnspan=3, sticky=tk.W)
        ttk.Checkbutton(settings_frame, text="Использовать отфильтрованные исследования", variable=self.use_filtered).grid(row=6, column=0, columnspan=3, sticky=tk.W)
        ttk.Checkbutton(settings_frame, text="Использовать кэш API", variable=self.use_cache).grid(row=7, column=0, columnspan=3, sticky=tk.W)

        # ---- Параметры кластеризации ----
        clust_frame = ttk.LabelFrame(left_frame, text="Параметры кластеризации (пациенты)", padding=5)
        clust_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(clust_frame, text="Компоненты для кластеризации:").grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.clust_components_listbox = tk.Listbox(clust_frame, selectmode=tk.EXTENDED, height=4, exportselection=False)
        self.clust_components_listbox.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(clust_frame, text="Нормализовать (StandardScaler)", variable=self.clust_normalize).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(clust_frame, text="Метод:").grid(row=3, column=0, sticky=tk.W)
        ttk.Combobox(clust_frame, textvariable=self.clust_method, values=['ward','complete','average','single'], state='readonly', width=10).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(clust_frame, text="Метрика:").grid(row=4, column=0, sticky=tk.W)
        ttk.Combobox(clust_frame, textvariable=self.clust_metric, values=['euclidean','correlation','cityblock'], state='readonly', width=10).grid(row=4, column=1, sticky=tk.W)
        ttk.Checkbutton(clust_frame, text="Исключать выбросы (Z-score > порога)", variable=self.exclude_outliers).grid(row=5, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(clust_frame, text="Порог (Z-score):").grid(row=6, column=0, sticky=tk.W)
        ttk.Entry(clust_frame, textvariable=self.outlier_threshold, width=6).grid(row=6, column=1, sticky=tk.W)
        self.recluster_btn = ttk.Button(clust_frame, text="Перекластеризовать", command=self.recluster_patients, state=tk.DISABLED)
        self.recluster_btn.grid(row=7, column=0, columnspan=2, pady=5)

        # ---- Кнопки управления ----
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Запустить PCA и кластеризацию", command=self.run_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить CSV", command=self.save_results_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=2)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=2)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать отчёт", command=self.generate_report, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=2)

        # ---- Правая панель (Notebook) ----
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tab_pca = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pca, text="Графики PCA")
        self.pca_frame = ttk.Frame(self.tab_pca)
        self.pca_frame.pack(fill=tk.BOTH, expand=True)
        self.tab_clusters = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_clusters, text="Кластеризация")
        self.cluster_frame = ttk.Frame(self.tab_clusters)
        self.cluster_frame.pack(fill=tk.BOTH, expand=True)
        self.tab_stats = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_stats, text="Статистика")
        self.stats_frame = ttk.Frame(self.tab_stats)
        self.stats_frame.pack(fill=tk.BOTH, expand=True)
        self.stats_tree = ttk.Treeview(self.stats_frame, show='headings')
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(self.stats_frame, orient=tk.VERTICAL, command=self.stats_tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_tree.configure(yscrollcommand=sb.set)
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
        msg = ("ИНСТРУКЦИЯ ПО PCA И КЛАСТЕРИЗАЦИИ (Глава 2, п. 2.5.4, 2.5.5)\n"
               "1. Загрузите и отфильтруйте данные на вкладке 'Загрузка'.\n"
               "2. Настройте параметры (типы эпох, число компонент, FDR).\n"
               "3. Нажмите 'Запустить PCA и кластеризацию'.\n"
               "4. После завершения можно менять компоненты, метод, метрику, фильтр выбросов и нажать 'Перекластеризовать'.\n"
               "5. Сохраняйте CSV и HTML-отчёт.\n"
               "Примечание: LMM для PC выполняется на смешанных эпохах, поэтому тип эпох для LMM должен содержать апноэ.")
        messagebox.showinfo("Инструкция", msg)

    def run_analysis(self):
        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_csv_btn.config(state=tk.DISABLED)
        self.save_plot_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        self.recluster_btn.config(state=tk.DISABLED)
        self.log("Загрузка данных...")
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _run_analysis_thread(self):
        try:
            self.results = {}
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
                self.log(f"Используем {len(study_ids)} исследований (после фильтрации)")

            def update_progress(page, total, _):
                if total > 0:
                    self.main_app.set_progress(int(page / total * 100))

            self.main_app.set_progress(0)
            self.log(f"Загрузка тонических эпох (data_type={self.pca_data_type.get()}) для PCA...")
            epochs_pca = get_epochs(api_url, token, study_ids=study_ids, data_type=self.pca_data_type.get(),
                                    stop_check=lambda: self.stop_flag, progress_callback=update_progress,
                                    use_cache=self.use_cache.get())
            if not epochs_pca or self.stop_flag:
                self.log("Нет данных для PCA.")
                return
            df_pca = pd.DataFrame(epochs_pca)
            self.log(f"Загружено {len(df_pca)} эпох для PCA")

            self.log(f"Загрузка эпох для LMM (data_type={self.lmm_data_type.get()})...")
            epochs_lmm = get_epochs(api_url, token, study_ids=study_ids, data_type=self.lmm_data_type.get(),
                                    stop_check=lambda: self.stop_flag, progress_callback=update_progress,
                                    use_cache=self.use_cache.get())
            if not epochs_lmm or self.stop_flag:
                self.log("Нет данных для LMM.")
                return
            df_lmm = pd.DataFrame(epochs_lmm)
            self.log(f"Загружено {len(df_lmm)} эпох для LMM")

            # Сбор признаков
            feature_cols = []
            for ch in self.all_channels:
                for feat in self.all_features:
                    col = f"{ch}_{feat}"
                    if col in df_pca.columns:
                        feature_cols.append(col)
            for a,b in self.coh_pairs:
                pair = f"{a}{b}"
                for band in self.coh_bands:
                    col = f"{pair}_coh_{band}"
                    if col in df_pca.columns:
                        feature_cols.append(col)
            for ch in self.all_channels:
                col = f"{ch}_dfa"
                if col in df_pca.columns:
                    feature_cols.append(col)
            feature_cols = list(set(feature_cols))
            self.log(f"Всего признаков: {len(feature_cols)}")
            if not feature_cols:
                self.log("Нет доступных признаков.")
                return

            # PCA на тонических
            X_pca = df_pca[feature_cols].copy().dropna()
            if len(X_pca) < 10:
                self.log("Слишком мало строк после удаления NaN в PCA.")
                return
            self.log(f"Матрица X для PCA: {X_pca.shape}")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_pca)
            n_comp = min(self.n_components.get(), X_scaled.shape[1], X_scaled.shape[0])
            pca = PCA(n_components=n_comp)
            X_pca_transformed = pca.fit_transform(X_scaled)
            self.log(f"PCA обучено, объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.3f}")
            self.log(f"Доли дисперсии: {pca.explained_variance_ratio_}")

            # Проецирование LMM-эпох
            X_lmm = df_lmm[feature_cols].copy().dropna()
            if len(X_lmm) < 30:
                self.log("Слишком мало строк для LMM.")
                return
            X_lmm_scaled = scaler.transform(X_lmm)
            X_lmm_pca = pca.transform(X_lmm_scaled)
            meta_lmm = df_lmm.loc[X_lmm.index, ['patient_id','study_id','has_apnea','epoch_stage']].copy()
            y_lmm = meta_lmm['has_apnea'].astype(int)
            self.log(f"Матрица для LMM: {X_lmm.shape}, эпох с апноэ: {y_lmm.sum()}")

            # Данные для кластеризации (все PC для каждого пациента)
            pca_idx = X_pca.index
            patient_ids = df_pca.loc[pca_idx, 'patient_id'].values
            patient_pc_all = pd.DataFrame({'patient_id': patient_ids})
            for pc in range(n_comp):
                patient_pc_all[f'PC{pc+1}'] = X_pca_transformed[:, pc]
            patient_avg = patient_pc_all.groupby('patient_id').mean().reset_index()
            self.results['patient_avg_all_pc'] = patient_avg
            self.results['feature_names'] = feature_cols
            self.results['pca'] = pca
            self.results['scaler'] = scaler
            self.results['n_components'] = n_comp
            self.results['pca_epochs'] = len(X_pca)
            self.results['lmm_epochs'] = len(X_lmm)

            # LMM для PC
            self._lmm_for_pcs(X_lmm_pca, meta_lmm)
            if 'lmm' in self.results and not self.results['lmm'].empty:
                self._roc_for_significant_pcs(X_lmm_pca, y_lmm)

            # Инициализация кластеризации
            self.clust_components_listbox.delete(0, tk.END)
            for i in range(n_comp):
                self.clust_components_listbox.insert(tk.END, f'PC{i+1}')
            self.clust_components_listbox.selection_set(0)
            if n_comp > 1:
                self.clust_components_listbox.selection_set(1)
            self.recluster_patients()
            self.recluster_btn.config(state=tk.NORMAL)

            self._plot_pca_results(pca, X_pca_transformed, X_lmm_pca, y_lmm)
            self._display_stats_table()

            self.save_csv_btn.config(state=tk.NORMAL)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.log("Анализ завершён.")
        except Exception as e:
            self.log(f"Ошибка: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.main_app.set_progress(0)

    def _lmm_for_pcs(self, X_pca, meta):
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            cov_df = None
        else:
            cov_df = filtered_df[['study_id','age_at_study','gender','bmi']].drop_duplicates('study_id')
            cov_df['gender_code'] = (cov_df['gender'] == 'M').astype(int)
            cov_df = cov_df[['study_id','age_at_study','gender_code','bmi']]
        meta = meta.copy()
        if cov_df is not None:
            meta = meta.merge(cov_df, on='study_id', how='left')
        else:
            for c in ['age_at_study','gender_code','bmi']:
                meta[c] = np.nan
        results = []
        for pc in range(self.results['n_components']):
            col = f'PC{pc+1}'
            data = meta.copy()
            data[col] = X_pca[:, pc]
            data = data.dropna(subset=[col, 'has_apnea', 'patient_id'])
            if len(data) < 30 or data['has_apnea'].nunique() < 2:
                continue
            formula = f"{col} ~ has_apnea"
            if self.include_stage.get() and 'epoch_stage' in data.columns:
                formula += " + epoch_stage"
            if self.include_covariates.get() and cov_df is not None:
                formula += " + age_at_study + gender_code + bmi"
            try:
                model = smf.mixedlm(formula, data, groups=data['patient_id'])
                result = model.fit(reml=True, method='lbfgs', maxiter=1000)
                beta = result.params.get('has_apnea[T.True]', result.params.get('has_apnea', None))
                pval = result.pvalues.get('has_apnea[T.True]', result.pvalues.get('has_apnea', None))
                if beta is None or pval is None:
                    continue
                ci = result.conf_int().loc['has_apnea[T.True]' if 'has_apnea[T.True]' in result.params else 'has_apnea']
                results.append({'PC': pc+1, 'beta': beta, 'p_value': pval,
                                'ci_low': ci[0], 'ci_high': ci[1], 'n_obs': len(data)})
            except Exception as e:
                self.log(f"LMM PC{pc+1}: {e}")
        if results:
            df = pd.DataFrame(results)
            df['q_value'] = false_discovery_control(df['p_value'].values, method='bh')
            df['significant'] = df['q_value'] < self.fdr_threshold.get()
            self.results['lmm'] = df
            self.log("LMM для PC завершён.")
        else:
            self.results['lmm'] = pd.DataFrame()

    def _roc_for_significant_pcs(self, X_pca, y):
        if self.results['lmm'].empty:
            return
        sign_pcs = self.results['lmm'][self.results['lmm']['significant']]['PC'].values
        if len(sign_pcs) == 0:
            return
        roc_data = []
        for pc in sign_pcs:
            scores = X_pca[:, pc-1]
            auc = roc_auc_score(y, scores)
            fpr, tpr, _ = roc_curve(y, scores)
            roc_data.append({'PC': pc, 'auc': auc, 'fpr': fpr, 'tpr': tpr})
            self.log(f"PC{pc}: AUC = {auc:.3f}")
        self.results['roc'] = roc_data

    def recluster_patients(self):
        if 'patient_avg_all_pc' not in self.results:
            self.log("Нет данных для кластеризации. Сначала выполните PCA.")
            return
        sel = self.clust_components_listbox.curselection()
        if not sel:
            self.log("Выберите хотя бы одну компоненту.")
            return
        comp_cols = [f'PC{i+1}' for i in sel]
        data = self.results['patient_avg_all_pc'].copy()
        X_clust = data[comp_cols].values

        # Исключение выбросов
        if self.exclude_outliers.get():
            z_scores = np.abs(zscore(X_clust, axis=0))
            keep = (z_scores < self.outlier_threshold.get()).all(axis=1)
            n_outliers = (~keep).sum()
            if n_outliers > 0:
                self.log(f"Исключено {n_outliers} пациентов как выбросы (Z-score > {self.outlier_threshold.get()})")
                data = data[keep].copy()
                X_clust = data[comp_cols].values
            else:
                self.log("Выбросов не обнаружено")

        if self.clust_normalize.get():
            scaler_clust = StandardScaler()
            X_clust = scaler_clust.fit_transform(X_clust)

        linkage_matrix = linkage(X_clust, method=self.clust_method.get(), metric=self.clust_metric.get())
        max_k = min(10, len(data)-1)
        sil_scores = []
        for k in range(2, max_k+1):
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            if len(set(labels)) > 1:
                score = silhouette_score(X_clust, labels)
                sil_scores.append((k, score))
        best_k = max(sil_scores, key=lambda x: x[1])[0] if sil_scores else 2
        self.log(f"Оптимальное число кластеров: {best_k}")
        labels = fcluster(linkage_matrix, best_k, criterion='maxclust')
        data['cluster'] = labels
        self.results['patient_clusters'] = data
        self.results['linkage_matrix'] = linkage_matrix
        self.results['best_k'] = best_k
        self._bootstrap_clusters(X_clust, labels)
        self._compare_clusters_with_clinical(data)
        self._plot_cluster_results()
        self.log("Кластеризация обновлена.")

    def _bootstrap_clusters(self, X, original_labels, n_iter=500):
        from sklearn.utils import resample
        n = X.shape[0]
        ari = []
        for _ in range(min(n_iter, 500)):
            if self.stop_flag:
                break
            idx = resample(range(n), replace=True, n_samples=n)
            Xb = X[idx]
            try:
                link = linkage(Xb, method=self.clust_method.get(), metric=self.clust_metric.get())
                labels = fcluster(link, self.results['best_k'], criterion='maxclust')
                ari.append(adjusted_rand_score(original_labels, labels[:n]))
            except:
                pass
        if ari:
            mean_ari = np.mean(ari)
            self.log(f"Бутстрап устойчивость: средний ARI = {mean_ari:.3f}")
            self.results['bootstrap_ari'] = mean_ari

    def _compare_clusters_with_clinical(self, cluster_df):
        filtered_df = self.main_app.get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            return
        clinical = filtered_df[['patient_id','age_at_study','gender','bmi',
                                'breathing_impairment_severity','cvd_hypertension','endocrine_diabetes']].drop_duplicates('patient_id')
        merged = cluster_df.merge(clinical, on='patient_id', how='left')
        stats = {}
        for cl in sorted(merged['cluster'].unique()):
            sub = merged[merged['cluster'] == cl]
            stats[cl] = {
                'n': len(sub),
                'age_mean': sub['age_at_study'].mean(),
                'age_std': sub['age_at_study'].std(),
                'bmi_mean': sub['bmi'].mean(),
                'bmi_std': sub['bmi'].std(),
                'gender_M': (sub['gender'] == 'M').mean(),
                'hypertension': sub['cvd_hypertension'].mean(),
                'diabetes': sub['endocrine_diabetes'].mean()
            }
        self.results['cluster_stats'] = stats
        age_grp = [merged[merged['cluster']==cl]['age_at_study'].dropna().values for cl in sorted(merged['cluster'].unique())]
        if len(age_grp) > 1 and all(len(g)>1 for g in age_grp):
            f_age, p_age = f_oneway(*age_grp)
            self.results['anova_age'] = {'f': f_age, 'p': p_age}
            self.log(f"ANOVA возраст: F={f_age:.3f}, p={p_age:.4f}")
        bmi_grp = [merged[merged['cluster']==cl]['bmi'].dropna().values for cl in sorted(merged['cluster'].unique())]
        if len(bmi_grp) > 1 and all(len(g)>1 for g in bmi_grp):
            f_bmi, p_bmi = f_oneway(*bmi_grp)
            self.results['anova_bmi'] = {'f': f_bmi, 'p': p_bmi}
            self.log(f"ANOVA ИМТ: F={f_bmi:.3f}, p={p_bmi:.4f}")
        tab = pd.crosstab(merged['cluster'], merged['breathing_impairment_severity'])
        if tab.shape[0] > 1 and tab.shape[1] > 1:
            chi2, p_chi2, _, _ = chi2_contingency(tab)
            self.results['chi2_severity'] = {'chi2': chi2, 'p': p_chi2}
            self.log(f"χ² тяжесть ОАС: χ²={chi2:.3f}, p={p_chi2:.4f}")

    def _display_stats_table(self):
        for row in self.stats_tree.get_children():
            self.stats_tree.delete(row)
        if 'lmm' in self.results and not self.results['lmm'].empty:
            cols = list(self.results['lmm'].columns)
            self.stats_tree['columns'] = cols
            for c in cols:
                self.stats_tree.heading(c, text=c)
                self.stats_tree.column(c, width=80, anchor='center')
            for _, row in self.results['lmm'].iterrows():
                self.stats_tree.insert('', 'end', values=list(row))
        else:
            self.stats_tree['columns'] = ['message']
            self.stats_tree.heading('message', text='Информация')
            self.stats_tree.insert('', 'end', values=['Нет результатов LMM (нет эпох с апноэ в выбранном типе)'])

    def _plot_pca_results(self, pca, X_tonic, X_mixed, y_mixed):
        for w in self.pca_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(12,10), dpi=100)
        ax1 = fig.add_subplot(2,2,1)
        colors = np.where(y_mixed==1, 'red', 'blue')
        ax1.scatter(X_mixed[:,0], X_mixed[:,1], c=colors, alpha=0.3, s=5)
        ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
        ax1.set_title('Проекция смешанных эпох (красный – апноэ)')
        ax2 = fig.add_subplot(2,2,2)
        var = pca.explained_variance_ratio_[:self.results['n_components']]
        ax2.bar(range(1,len(var)+1), var)
        ax2.set_xlabel('Компонента'); ax2.set_ylabel('Доля дисперсии')
        ax2.set_title('Scree plot')
        ax3 = fig.add_subplot(2,2,3)
        loadings = pca.components_.T
        sum_abs = np.sum(np.abs(loadings[:,:self.results['n_components']]), axis=1)
        top_idx = np.argsort(sum_abs)[-20:]
        top_feat = [self.results['feature_names'][i] for i in top_idx]
        top_load = loadings[top_idx, :self.results['n_components']]
        im = ax3.imshow(top_load, cmap='RdBu_r', aspect='auto')
        ax3.set_xticks(np.arange(self.results['n_components']))
        ax3.set_xticklabels([f'PC{i+1}' for i in range(self.results['n_components'])])
        ax3.set_yticks(np.arange(len(top_feat)))
        ax3.set_yticklabels(top_feat, fontsize=8)
        fig.colorbar(im, ax=ax3)
        ax3.set_title('Loadings (топ-20)')
        ax4 = fig.add_subplot(2,2,4)
        if 'roc' in self.results and self.results['roc']:
            for roc in self.results['roc']:
                ax4.plot(roc['fpr'], roc['tpr'], label=f"PC{roc['PC']} (AUC={roc['auc']:.3f})")
            ax4.plot([0,1],[0,1],'k--')
            ax4.set_xlabel('FPR'); ax4.set_ylabel('TPR'); ax4.set_title('ROC')
            ax4.legend()
        else:
            ax4.text(0.5,0.5,'Нет значимых PC', ha='center')
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.pca_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig

    def _plot_cluster_results(self):
        if 'patient_clusters' not in self.results:
            return
        for w in self.cluster_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(12,8), dpi=100)
        ax1 = fig.add_subplot(2,2,1)
        labels = self.results['patient_clusters']['patient_id'].astype(str).tolist()
        dendrogram(self.results['linkage_matrix'], ax=ax1, labels=labels, leaf_rotation=90, leaf_font_size=8)
        ax1.set_title('Дендрограмма')
        ax2 = fig.add_subplot(2,2,2)
        clust = self.results['patient_clusters']
        if 'PC1' in clust.columns and 'PC2' in clust.columns:
            for cl in sorted(clust['cluster'].unique()):
                sub = clust[clust['cluster']==cl]
                ax2.scatter(sub['PC1'], sub['PC2'], label=f'Кластер {cl}', alpha=0.7)
            ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
            ax2.set_title('Кластеры пациентов по PC1/PC2')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'PC1/PC2 не найдены\n(возможно, выбраны другие компоненты)', ha='center')
        ax3 = fig.add_subplot(2,2,3)
        ax3.axis('tight'); ax3.axis('off')
        if 'cluster_stats' in self.results:
            rows = []
            for cl, st in self.results['cluster_stats'].items():
                rows.append([cl, st['n'],
                             f"{st['age_mean']:.1f}±{st['age_std']:.1f}",
                             f"{st['bmi_mean']:.1f}±{st['bmi_std']:.1f}",
                             f"{st['gender_M']*100:.1f}",
                             f"{st['hypertension']*100:.1f}",
                             f"{st['diabetes']*100:.1f}"])
            table = ax3.table(cellText=rows,
                              colLabels=['Кластер','n','Возраст','ИМТ','Мужчины%','Гиперт.%','Диабет%'],
                              loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            ax3.set_title('Сравнение кластеров')
        ax4 = fig.add_subplot(2,2,4)
        ax4.axis('off')
        text = "Статистические тесты:\n"
        if 'anova_age' in self.results:
            text += f"ANOVA возраст: F={self.results['anova_age']['f']:.3f}, p={self.results['anova_age']['p']:.4f}\n"
        if 'anova_bmi' in self.results:
            text += f"ANOVA ИМТ: F={self.results['anova_bmi']['f']:.3f}, p={self.results['anova_bmi']['p']:.4f}\n"
        if 'chi2_severity' in self.results:
            text += f"χ² тяжесть ОАС: χ²={self.results['chi2_severity']['chi2']:.3f}, p={self.results['chi2_severity']['p']:.4f}\n"
        if 'bootstrap_ari' in self.results:
            text += f"Устойчивость (ARI бутстрап): {self.results['bootstrap_ari']:.3f}\n"
        ax4.text(0.1,0.9, text, transform=ax4.transAxes, fontsize=10, verticalalignment='top')
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.cluster_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_results_csv(self):
        if not self.results:
            messagebox.showwarning("Нет данных", "Сначала выполните анализ.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        if 'lmm' in self.results and not self.results['lmm'].empty:
            self.results['lmm'].to_csv(path.replace('.csv','_lmm.csv'), index=False)
            self.log(f"LMM сохранён в {path.replace('.csv','_lmm.csv')}")
        if 'patient_clusters' in self.results:
            self.results['patient_clusters'].to_csv(path.replace('.csv','_clusters.csv'), index=False)
            self.log(f"Кластеры сохранены в {path.replace('.csv','_clusters.csv')}")
        if 'patient_avg_all_pc' in self.results:
            self.results['patient_avg_all_pc'].to_csv(path.replace('.csv','_patient_pc.csv'), index=False)
            self.log(f"Средние PC пациентов сохранены в {path.replace('.csv','_patient_pc.csv')}")

    def save_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "График не построен.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if path:
            self.current_figure.savefig(path, dpi=150, bbox_inches='tight')
            self.log(f"График сохранён в {path}")

    def generate_report(self):
        if not self.results:
            messagebox.showwarning("Нет данных", "Сначала выполните анализ.")
            return
        buf_pca = io.BytesIO()
        self.current_figure.savefig(buf_pca, format='png', dpi=100, bbox_inches='tight')
        img_pca = base64.b64encode(buf_pca.getvalue()).decode('utf-8')
        buf_clust = io.BytesIO()
        for w in self.cluster_frame.winfo_children():
            if isinstance(w, FigureCanvasTkAgg):
                w.figure.savefig(buf_clust, format='png', dpi=100, bbox_inches='tight')
                break
        img_clust = base64.b64encode(buf_clust.getvalue()).decode('utf-8') if buf_clust.getvalue() else ""

        lmm_html = ""
        eq_html = ""
        if 'lmm' in self.results and not self.results['lmm'].empty:
            lmm_html = self.results['lmm'].to_html(index=False, float_format="%.4f")
            sign = self.results['lmm'][self.results['lmm']['significant']]
            if not sign.empty:
                eq_html = "<h3>Уравнения линейной регрессии для значимых PC</h3>"
                for _, row in sign.iterrows():
                    eq_html += f"<p><strong>PC{int(row['PC'])}</strong> = β₀ + {row['beta']:.4f}·has_apnea + ...<br>"
                    eq_html += f"95% CI: [{row['ci_low']:.4f}, {row['ci_high']:.4f}]</p>"
        else:
            lmm_html = "<p>LMM не выполнялся (нет эпох с апноэ в выбранном типе).</p>"

        cluster_table = ""
        if 'cluster_stats' in self.results:
            rows = []
            for cl, st in self.results['cluster_stats'].items():
                rows.append({
                    'Кластер': cl, 'n': st['n'],
                    'Возраст (M±SD)': f"{st['age_mean']:.1f}±{st['age_std']:.1f}",
                    'ИМТ (M±SD)': f"{st['bmi_mean']:.1f}±{st['bmi_std']:.1f}",
                    'Мужчины (%)': f"{st['gender_M']*100:.1f}",
                    'Гипертензия (%)': f"{st['hypertension']*100:.1f}",
                    'Диабет (%)': f"{st['diabetes']*100:.1f}"
                })
            cluster_table = pd.DataFrame(rows).to_html(index=False)

        html = f"""<!DOCTYPE html>
        <html><head><meta charset="utf-8"><title>PCA и кластеризация ЭЭГ</title>
        <style> body{{font-family:Arial;margin:20px;}} table{{border-collapse:collapse;width:100%;margin-bottom:20px;}} th,td{{border:1px solid #ddd;padding:8px;text-align:left;}} th{{background:#f2f2f2;}} .plot{{margin:20px 0;text-align:center;}} </style>
        </head><body>
        <h1>Анализ главных компонент (PCA) и кластеризация пациентов</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Тип эпох для PCA:</strong> {self.pca_data_type.get()} | <strong>Тип эпох для LMM:</strong> {self.lmm_data_type.get()}</p>
        <p><strong>Число компонент:</strong> {self.results['n_components']} | <strong>Объяснённая дисперсия:</strong> {self.results['pca'].explained_variance_ratio_.sum():.3f}</p>
        <p><strong>Эпох PCA:</strong> {self.results['pca_epochs']} | <strong>Эпох LMM:</strong> {self.results['lmm_epochs']}</p>
        <p><strong>FDR порог:</strong> q = {self.fdr_threshold.get()} | <strong>Ковариаты:</strong> {'включены' if self.include_covariates.get() else 'нет'}</p>
        <div class="plot"><img src="data:image/png;base64,{img_pca}" style="max-width:100%;"/></div>
        <h2>Результаты LMM для главных компонент</h2>{lmm_html}{eq_html}
        <h2>Кластеризация пациентов</h2>
        <div class="plot"><img src="data:image/png;base64,{img_clust}" style="max-width:100%;"/></div>
        <h3>Сравнение кластеров</h3>{cluster_table}
        <p><em>Интерпретация:</em> Значимые различия между кластерами по клиническим данным указывают на нейрофизиологические фенотипы.</p>
        </body></html>"""
        fd, path = tempfile.mkstemp(suffix='.html', prefix='pca_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт: {path}")