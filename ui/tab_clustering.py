# ui/tab_clustering.py
import threading
import tkinter as tk
from tkinter import ttk, messagebox
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import f_oneway, chi2_contingency, false_discovery_control
import statsmodels.formula.api as smf

from ui.base_tab import BaseTab
from core.api_client import get_epochs
from core.config import SEVERITY_MAP

# Константы
CLUSTERING_PAGE_SIZE = 5000


class ClusteringTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        # Настройки
        self.data_type = tk.IntVar(value=1)          # тонические эпохи (1)
        self.use_filtered = tk.BooleanVar(value=True)
        self.use_cache = tk.BooleanVar(value=True)
        self.exclude_central_mixed = tk.BooleanVar(value=True)
        self.n_clusters = tk.IntVar(value=0)         # 0 = автоопределение
        self.n_bootstrap = tk.IntVar(value=500)
        
        self.stop_flag = False
        self.epochs_df = None
        self.patient_data = None          # DataFrame с средними PC1, PC2 для каждого пациента
        self.cluster_labels = None
        self.linkage_matrix = None
        self.silhouette_score = None
        self.bootstrap_ari = None
        self.current_figure = None
        self.cluster_comparison = None    # результаты сравнения кластеров
        
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
        right_frame.config(width=600)
        
        # ========== ЛЕВАЯ ПАНЕЛЬ ==========
        info_frame = ttk.LabelFrame(left_frame, text="Источник данных", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, text="Используются отфильтрованные данные из вкладки 'Загрузка'",
                  foreground="blue").pack(anchor=tk.W)
        ttk.Label(info_frame, text="(токен и URL не требуются)").pack(anchor=tk.W)
        
        desc_frame = ttk.LabelFrame(left_frame, text="Иерархическая кластеризация пациентов (Глава 2, п. 2.5.5)", padding=5)
        desc_frame.pack(fill=tk.X, padx=5, pady=5)
        desc_text = (
            "На основе средних значений первых двух главных компонент (PC1 и PC2) от всех тонических эпох "
            "выполняется иерархическая кластеризация пациентов методом Уорда. Оптимальное число кластеров "
            "определяется по максимуму силуэта. Кластеры сравниваются по клиническим характеристикам (возраст, "
            "ИМТ, AHI, коморбидности) с FDR-коррекцией. Устойчивость оценивается бутстрапом (индекс скорректированного ранда)."
        )
        ttk.Label(desc_frame, text=desc_text, justify=tk.LEFT, wraplength=600).pack(anchor=tk.W, fill=tk.X, padx=5, pady=2)
        ttk.Button(desc_frame, text="Показать инструкцию", command=self.show_instructions).pack(anchor=tk.W, padx=5, pady=2)
        
        # Параметры
        param_frame = ttk.LabelFrame(left_frame, text="Параметры", padding=5)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(param_frame, text="Тип набора эпох:").grid(row=0, column=0, sticky=tk.W)
        self.data_type_combo = ttk.Combobox(param_frame, textvariable=self.data_type, values=[1,2,3], state='readonly', width=5)
        self.data_type_combo.grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="1=тонический, 2=все, 3=фильтр по положению").grid(row=0, column=2, sticky=tk.W)
        ttk.Checkbutton(param_frame, text="Использовать отфильтрованные исследования", variable=self.use_filtered).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Checkbutton(param_frame, text="Исключить центральное/смешанное апноэ", variable=self.exclude_central_mixed).grid(row=2, column=0, columnspan=3, sticky=tk.W)
        ttk.Checkbutton(param_frame, text="Использовать кэш", variable=self.use_cache).grid(row=3, column=0, columnspan=3, sticky=tk.W)
        
        cluster_frame = ttk.LabelFrame(left_frame, text="Кластеризация", padding=5)
        cluster_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(cluster_frame, text="Число кластеров (0 = авто):").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(cluster_frame, from_=0, to=10, textvariable=self.n_clusters, width=5).grid(row=0, column=1, padx=5)
        ttk.Label(cluster_frame, text="Итераций бутстрапа:").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(cluster_frame, from_=100, to=1000, textvariable=self.n_bootstrap, width=5).grid(row=1, column=1, padx=5)
        
        # Кнопки
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Запустить кластеризацию", command=self.run_clustering)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить кластеры (CSV)", command=self.save_clusters_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график (PNG)", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)
        self.report_btn = ttk.Button(btn_frame, text="Сформировать отчёт", command=self.generate_report, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=5)
        
        # ========== ПРАВАЯ ПАНЕЛЬ ==========
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="Графики")
        self.plot_frame = ttk.Frame(self.tab_plot)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tab_table = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_table, text="Сравнение кластеров")
        self.tree = ttk.Treeview(self.tab_table)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(self.tab_table, orient=tk.VERTICAL, command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)
        
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
            "ИНСТРУКЦИЯ ПО КЛАСТЕРИЗАЦИИ ПАЦИЕНТОВ (Глава 2, п. 2.5.5)\n"
            "=======================================================\n"
            "1. Загрузите и отфильтруйте данные на вкладке 'Загрузка'.\n"
            "2. Выберите тип набора эпох (рекомендуется тонический, data_type=1).\n"
            "3. Нажмите 'Запустить кластеризацию'.\n"
            "4. Программа вычислит для каждого пациента средние PC1 и PC2,\n"
            "   выполнит иерархическую кластеризацию (метод Уорда) и определит\n"
            "   оптимальное число кластеров по силуэту.\n"
            "5. Будут построены дендрограмма и scatter plot кластеров.\n"
            "6. Выполнится сравнение кластеров по клиническим переменным (ANOVA/χ²)\n"
            "   с FDR-коррекцией.\n"
            "7. Бутстрап (500 итераций) оценит устойчивость кластерного решения.\n"
            "8. Кнопка 'Сформировать отчёт' создаст HTML-отчёт со всеми результатами."
        )
        messagebox.showinfo("Инструкция", msg)
        
    def _load_epochs(self):
        load_tab = self.main_app.tabs['load']
        api_url = load_tab.api_url.get().rstrip('/')
        token = load_tab.token.get().strip()
        if not api_url or not token:
            self.log("Ошибка: не указаны URL или токен API.")
            return None
            
        study_ids = None
        if self.use_filtered.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is None or filtered_df.empty:
                self.log("Нет отфильтрованных данных.")
                return None
            study_ids = filtered_df['study_id'].unique().tolist()
            
        if self.exclude_central_mixed.get():
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is not None:
                allowed = ['no_impairment', 'mild', 'moderate', 'severe']
                filtered_df = filtered_df[filtered_df['breathing_impairment_severity'].isin(allowed)]
                if not filtered_df.empty:
                    study_ids = filtered_df['study_id'].unique().tolist()
                    
        def update_progress(page, total, _):
            if total > 0:
                self.main_app.set_progress(int(page / total * 100))
                
        self.main_app.set_progress(0)
        self.log("Загрузка тонических эпох...")
        epochs = get_epochs(
            api_url, token,
            study_ids=study_ids,
            data_type=self.data_type.get(),
            stop_check=lambda: self.stop_flag,
            progress_callback=update_progress,
            use_cache=self.use_cache.get(),
            page_size=CLUSTERING_PAGE_SIZE
        )
        self.main_app.set_progress(100)
        if not epochs:
            return None
        df = pd.DataFrame(epochs)
        self.log(f"Загружено {len(df)} эпох")
        return df
        
    def run_clustering(self):
        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_csv_btn.config(state=tk.DISABLED)
        self.save_plot_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        self.log("Загрузка данных...")
        thread = threading.Thread(target=self._run_clustering_thread)
        thread.daemon = True
        thread.start()
        
    def _run_clustering_thread(self):
        try:
            df = self._load_epochs()
            if df is None or self.stop_flag:
                return
                
            # Выбираем все количественные признаки для PCA (все каналы, все признаки)
            # Для простоты используем предопределённый набор признаков для PCA (как в PCA вкладке)
            # Но можно также использовать уже готовые PC1/PC2 из сохранённой модели? Пересчитаем на лету.
            feature_cols = [col for col in df.columns if any(ch in col for ch in ['F3','C3','O1','F4','C4','O2']) 
                            and any(feat in col for feat in ['mean','std','abs_delta','rel_delta','abs_theta','rel_theta',
                                                              'abs_alpha','rel_alpha','abs_sigma','rel_sigma','abs_beta',
                                                              'rel_beta','tbr','dar','se50','gamma_power','sampen','dfa'])]
            if not feature_cols:
                self.log("Не найдены признаки для PCA.")
                return
            X = df[feature_cols].copy()
            X = X.dropna()
            if len(X) < 10:
                self.log("Недостаточно данных после удаления NaN.")
                return
            # Масштабирование
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # PCA до 2 компонент
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            # Добавляем в df
            df_pca = df.loc[X.index].copy()
            df_pca['PC1'] = X_pca[:, 0]
            df_pca['PC2'] = X_pca[:, 1]
            
            # Агрегируем по пациенту: средние PC1 и PC2
            patient_agg = df_pca.groupby('patient_id').agg({
                'PC1': 'mean',
                'PC2': 'mean',
                'study_id': 'first'  # для связи с клиническими данными
            }).reset_index()
            # Добавляем клинические данные из отфильтрованного датафрейма
            filtered_df = self.main_app.get_filtered_data()
            if filtered_df is not None:
                clin = filtered_df[['patient_id', 'age_at_study', 'gender', 'bmi', 'ahi', 
                                    'cvd_hypertension', 'cvd_ihd', 'endocrine_diabetes', 'som_insomnia',
                                    'breathing_impairment_severity']].drop_duplicates(subset=['patient_id'])
                patient_agg = patient_agg.merge(clin, on='patient_id', how='left')
            else:
                self.log("Нет клинических данных для сравнения кластеров.")
                
            if len(patient_agg) < 5:
                self.log(f"Слишком мало пациентов: {len(patient_agg)}")
                return
                
            # Кластеризация
            X_clust = patient_agg[['PC1', 'PC2']].values
            # Определение оптимального числа кластеров (если не задано вручную)
            n_clust = self.n_clusters.get()
            if n_clust == 0:
                # Оцениваем силуэт для 2..min(10, n_patients-1)
                max_clust = min(10, len(patient_agg)-1)
                sil_scores = []
                for k in range(2, max_clust+1):
                    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
                    labels = clustering.fit_predict(X_clust)
                    if len(set(labels)) > 1:
                        sil = silhouette_score(X_clust, labels)
                        sil_scores.append((k, sil))
                if sil_scores:
                    n_clust = max(sil_scores, key=lambda x: x[1])[0]
                    self.log(f"Оптимальное число кластеров по силуэту: {n_clust}")
                else:
                    n_clust = 2
            # Финальная кластеризация
            clustering = AgglomerativeClustering(n_clusters=n_clust, linkage='ward')
            labels = clustering.fit_predict(X_clust)
            patient_agg['cluster'] = labels
            self.cluster_labels = labels
            self.patient_data = patient_agg
            
            # Вычисление силуэта
            self.silhouette_score = silhouette_score(X_clust, labels)
            self.log(f"Силуэт для {n_clust} кластеров: {self.silhouette_score:.3f}")
            
            # Построение дендрограммы
            linkage_matrix = linkage(X_clust, method='ward')
            self.linkage_matrix = linkage_matrix
            
            # Визуализация
            self._plot_clusters(X_clust, labels, patient_agg)
            
            # Сравнение кластеров по клиническим данным
            self._compare_clusters(patient_agg, n_clust)
            
            # Бутстрап для оценки устойчивости
            self._run_bootstrap_stability(X_clust, n_clust)
            
            self.save_csv_btn.config(state=tk.NORMAL)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            self.log("Кластеризация завершена.")
            
        except Exception as e:
            self.log(f"Ошибка: {e}")
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
    def _plot_clusters(self, X, labels, patient_agg):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(12, 10), dpi=100)
        
        # Дендрограмма
        ax1 = fig.add_subplot(2, 2, 1)
        from scipy.cluster.hierarchy import dendrogram
        dendrogram(self.linkage_matrix, ax=ax1, labels=patient_agg['patient_id'].astype(str), leaf_rotation=90, leaf_font_size=8)
        ax1.set_title('Дендрограмма (метод Уорда)')
        ax1.set_xlabel('Пациент')
        ax1.set_ylabel('Расстояние')
        
        # Scatter plot PC1 vs PC2 с цветом кластеров
        ax2 = fig.add_subplot(2, 2, 2)
        scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        ax2.set_xlabel('PC1 (средняя по пациенту)')
        ax2.set_ylabel('PC2 (средняя по пациенту)')
        ax2.set_title(f'Кластеры пациентов в пространстве PC1-PC2 (силуэт={self.silhouette_score:.3f})')
        plt.colorbar(scatter, ax=ax2, label='Кластер')
        
        # Boxplot AHI по кластерам
        ax3 = fig.add_subplot(2, 2, 3)
        ahi_by_cluster = [patient_agg[patient_agg['cluster']==c]['ahi'].dropna().values for c in sorted(patient_agg['cluster'].unique())]
        if ahi_by_cluster:
            ax3.boxplot(ahi_by_cluster, labels=[f'Кластер {c}' for c in sorted(patient_agg['cluster'].unique())])
            ax3.set_ylabel('AHI (событий/ч)')
            ax3.set_title('Распределение AHI по кластерам')
            ax3.tick_params(axis='x', rotation=45)
            
        # Распределение тяжести ОАС
        ax4 = fig.add_subplot(2, 2, 4)
        severity_labels = {
            'no_impairment': 'Норма', 'mild': 'Лёгкая', 'moderate': 'Умеренная', 'severe': 'Тяжёлая'
        }
        severity_counts = {}
        for c in sorted(patient_agg['cluster'].unique()):
            sub = patient_agg[patient_agg['cluster']==c]
            counts = sub['breathing_impairment_severity'].value_counts()
            severity_counts[c] = [counts.get(sev, 0) for sev in ['no_impairment','mild','moderate','severe']]
        # Построим stacked bar
        ind = np.arange(len(severity_counts))
        bottom = np.zeros(len(severity_counts))
        for i, sev in enumerate(['no_impairment','mild','moderate','severe']):
            vals = [severity_counts[c][i] for c in sorted(severity_counts.keys())]
            ax4.bar(ind, vals, bottom=bottom, label=severity_labels[sev])
            bottom += vals
        ax4.set_xticks(ind)
        ax4.set_xticklabels([f'Кластер {c}' for c in sorted(severity_counts.keys())])
        ax4.set_ylabel('Число пациентов')
        ax4.set_title('Распределение тяжести ОАС по кластерам')
        ax4.legend()
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_figure = fig
        
    def _compare_clusters(self, df, n_clust):
        """Сравнивает кластеры по клиническим переменным (ANOVA/χ²) с FDR."""
        results = []
        # Количественные переменные
        quant_vars = ['age_at_study', 'bmi', 'ahi']
        for var in quant_vars:
            if var in df.columns:
                groups = [df[df['cluster']==c][var].dropna().values for c in range(n_clust)]
                if all(len(g) > 1 for g in groups):
                    f_stat, p_val = f_oneway(*groups)
                    results.append({
                        'variable': var,
                        'test': 'ANOVA',
                        'statistic': f_stat,
                        'p_value': p_val
                    })
        # Категориальные переменные (коморбидности)
        cat_vars = ['gender', 'cvd_hypertension', 'cvd_ihd', 'endocrine_diabetes', 'som_insomnia']
        for var in cat_vars:
            if var in df.columns:
                # Строим таблицу сопряжённости
                contingency = pd.crosstab(df['cluster'], df[var])
                if contingency.shape[1] >= 2:
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    results.append({
                        'variable': var,
                        'test': 'Chi-square',
                        'statistic': chi2,
                        'p_value': p_val
                    })
        # Коррекция FDR
        if results:
            pvals = [r['p_value'] for r in results]
            qvals = false_discovery_control(pvals, method='bh')
            for i, r in enumerate(results):
                r['q_value'] = qvals[i]
                r['significant'] = qvals[i] < 0.05
        self.cluster_comparison = pd.DataFrame(results)
        # Отобразить в таблице
        self._display_comparison_table(self.cluster_comparison)
        
    def _display_comparison_table(self, df):
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
            
    def _run_bootstrap_stability(self, X, n_clust):
        """Бутстрап для оценки устойчивости кластеров (скорректированный индекс ранда)."""
        n_iter = self.n_bootstrap.get()
        self.log(f"Бутстрап устойчивости: {n_iter} итераций...")
        ari_scores = []
        n_patients = X.shape[0]
        for i in range(n_iter):
            if self.stop_flag:
                break
            # Ресэмплируем пациентов с возвращением (блок-бутстрап)
            boot_idx = np.random.choice(n_patients, size=n_patients, replace=True)
            X_boot = X[boot_idx, :]
            # Кластеризация на бутстрап-выборке
            clustering_boot = AgglomerativeClustering(n_clusters=n_clust, linkage='ward')
            labels_boot = clustering_boot.fit_predict(X_boot)
            # Индекс ранда между исходными метками (для тех же пациентов) и бутстрап-метками
            # Но метки переставлены – используем adjusted_rand_score
            labels_orig = self.cluster_labels[boot_idx]
            ari = adjusted_rand_score(labels_orig, labels_boot)
            ari_scores.append(ari)
            if (i+1) % 100 == 0:
                self.log(f"Бутстрап: {i+1}/{n_iter}")
        if ari_scores:
            self.bootstrap_ari = np.mean(ari_scores)
            self.log(f"Средний скорректированный индекс ранда (ARI): {self.bootstrap_ari:.3f}")
        else:
            self.bootstrap_ari = np.nan
            
    def save_clusters_csv(self):
        if self.patient_data is None:
            messagebox.showwarning("Нет данных", "Сначала выполните кластеризацию.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.patient_data.to_csv(path, index=False, encoding='utf-8-sig')
            self.log(f"Результаты сохранены в {path}")
            
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
        if self.patient_data is None:
            messagebox.showwarning("Нет данных", "Сначала выполните кластеризацию.")
            return
        # Преобразуем данные в HTML
        # Клиническое сравнение
        comp_html = ""
        if self.cluster_comparison is not None and not self.cluster_comparison.empty:
            comp_html = self.cluster_comparison.to_html(index=False, float_format="%.4f")
        else:
            comp_html = "<p>Нет данных для сравнения.</p>"
            
        # График в base64
        buf = io.BytesIO()
        self.current_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plot_html = f'<div class="plot"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
        
        # Таблица пациентов с кластерами
        patients_table = self.patient_data[['patient_id', 'cluster', 'PC1', 'PC2', 'age_at_study', 'bmi', 'ahi', 'breathing_impairment_severity']].to_html(index=False, float_format="%.2f")
        
        # Распределение тяжести по кластерам
        severity_counts = self.patient_data.groupby(['cluster', 'breathing_impairment_severity']).size().unstack(fill_value=0)
        severity_html = severity_counts.to_html()
        
        html = f"""<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>Иерархическая кластеризация пациентов – Глава 2, п. 2.5.5</title>
        <style>
            body {{ font-family: Arial; margin:20px; }}
            table {{ border-collapse: collapse; width:100%; margin-bottom:20px; }}
            th, td {{ border:1px solid #ddd; padding:8px; text-align:left; }}
            th {{ background:#f2f2f2; }}
            .plot {{ margin:20px 0; text-align:center; }}
        </style>
        </head>
        <body>
        <h1>Отчёт об иерархической кластеризации пациентов на основе ЭЭГ-признаков</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Число кластеров:</strong> {len(set(self.cluster_labels))}</p>
        <p><strong>Силуэт:</strong> {self.silhouette_score:.3f}</p>
        <p><strong>Средний скорректированный индекс ранда (бутстрап):</strong> {self.bootstrap_ari:.3f if self.bootstrap_ari else 'не определён'}</p>
        <h2>Графики</h2>
        {plot_html}
        <h2>Пациенты и их кластеры</h2>
        {patients_table}
        <h2>Распределение тяжести ОАС по кластерам</h2>
        {severity_html}
        <h2>Сравнение кластеров по клиническим переменным (FDR-коррекция)</h2>
        {comp_html}
        <p><em>Интерпретация:</em> Кластеры, различающиеся по AHI, возрасту, ИМТ или коморбидностям, свидетельствуют о наличии скрытых нейрофизиологических фенотипов, не полностью определяемых AHI.</p>
        </body></html>
        """
        fd, path = tempfile.mkstemp(suffix='.html', prefix='clustering_report_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{path}')
        self.log(f"Отчёт открыт в браузере: {path}")