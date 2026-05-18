# ui/tab_pca.py

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from ui.base_tab import BaseTab
from core.api_client import get_epochs
from core.config import DEFAULT_API_URL, DEFAULT_TOKEN

class PCAAnalysisTab(BaseTab):
    def __init__(self, parent, main_app):
        super().__init__(parent, main_app)
        self.api_url = tk.StringVar(value=DEFAULT_API_URL)
        self.token = tk.StringVar(value=DEFAULT_TOKEN)
        self.data_type = tk.IntVar(value=2)          # 2 = все эпохи
        self.n_components = tk.IntVar(value=5)
        self.scale_data = tk.BooleanVar(value=True)
        self.use_filtered = tk.BooleanVar(value=False)
        self.use_logreg = tk.BooleanVar(value=True)
        self.logreg_cv_folds = tk.IntVar(value=5)
        self.use_cache = tk.BooleanVar(value=True)   # ДОБАВЛЕНО
        self.stop_flag = False
        self.results = None
        self.current_figure = None

        self.all_channels = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        self.all_features = [
            'mean', 'std', 'min', 'max', 'range', 'rms',
            'abs_delta', 'rel_delta', 'abs_theta', 'rel_theta',
            'abs_alpha', 'rel_alpha', 'abs_sigma', 'rel_sigma',
            'abs_beta', 'rel_beta', 'tbr', 'dar', 'se50', 'gamma_power', 'sampen'
        ]

        # Сохраняем выделения в листбоксах
        self.saved_channels_selection = []
        self.saved_features_selection = []

        self._create_widgets()

    # ------------------------------------------------------------
    # Интерфейс
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

        # Горизонтальное разделение
        paned = ttk.PanedWindow(inner, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)

        # ========== ЛЕВАЯ ПАНЕЛЬ ==========
        # API
        api_frame = ttk.LabelFrame(left_frame, text="Подключение к API", padding=5)
        api_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(api_frame, text="URL:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(api_frame, textvariable=self.api_url, width=40).grid(row=0, column=1, padx=5)
        ttk.Label(api_frame, text="Токен:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(api_frame, textvariable=self.token, width=40, show="*").grid(row=1, column=1, padx=5)

        # Тип данных
        type_frame = ttk.LabelFrame(left_frame, text="Данные", padding=5)
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(type_frame, text="Тип набора (data_type):").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(type_frame, textvariable=self.data_type, values=[1,2,3], state='readonly', width=5).grid(row=0, column=1, padx=5)
        ttk.Label(type_frame, text="1=тонический, 2=все эпохи, 3=фильтр по положению").grid(row=0, column=2, sticky=tk.W)
        ttk.Checkbutton(type_frame, text="Использовать отфильтрованные исследования (из вкладки 'Загрузка')",
                        variable=self.use_filtered).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)

        # Выбор каналов и признаков
        select_frame = ttk.LabelFrame(left_frame, text="Выбор признаков", padding=5)
        select_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(select_frame, text="Каналы (Ctrl/Shift):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.channels_listbox = tk.Listbox(select_frame, selectmode=tk.EXTENDED, height=6, width=20)
        for ch in self.all_channels:
            self.channels_listbox.insert(tk.END, ch)
        self.channels_listbox.selection_set(0, tk.END)
        self.channels_listbox.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.channels_listbox.bind('<<ListboxSelect>>', self.save_channels_selection)

        ttk.Label(select_frame, text="Признаки (Ctrl/Shift):").grid(row=0, column=1, sticky=tk.W, pady=2)
        self.features_listbox = tk.Listbox(select_frame, selectmode=tk.EXTENDED, height=10, width=30)
        for feat in self.all_features:
            self.features_listbox.insert(tk.END, feat)
        self.features_listbox.selection_set(0, tk.END)
        self.features_listbox.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
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
        ttk.Spinbox(class_frame, from_=2, to=10, textvariable=self.logreg_cv_folds, width=3).grid(row=1, column=1, padx=5, sticky=tk.W)

        # Кэширование
        cache_frame = ttk.LabelFrame(left_frame, text="Кэширование", padding=5)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cache_frame, text="Использовать кэш (ускоряет повторные запуски)",
                        variable=self.use_cache).pack(anchor=tk.W)

        # Кнопки
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Запустить PCA", command=self.run_pca)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.save_csv_btn = ttk.Button(btn_frame, text="Сохранить результаты (CSV)", command=self.save_results_csv, state=tk.DISABLED)
        self.save_csv_btn.pack(side=tk.LEFT, padx=5)
        self.save_plot_btn = ttk.Button(btn_frame, text="Сохранить график (PNG)", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)

        # ========== ПРАВАЯ ПАНЕЛЬ: Вкладки ==========
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

        self.log_text = tk.Text(self.tab_log, wrap=tk.WORD, font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._clear_coef_tab()

    # ------------------------------------------------------------
    # Сохранение выделения
    # ------------------------------------------------------------
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
    # Логирование
    # ------------------------------------------------------------
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.main_app.log(msg)

    # ------------------------------------------------------------
    # Остановка
    # ------------------------------------------------------------
    def stop_analysis(self):
        self.stop_flag = True
        self.log("Остановка запрошена...")

    def run_pca(self):
        self.stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_csv_btn.config(state=tk.DISABLED)
        self.save_plot_btn.config(state=tk.DISABLED)
        self.log("Загрузка данных...")
        thread = threading.Thread(target=self._run_pca_thread)
        thread.daemon = True
        thread.start()

    # ------------------------------------------------------------
    # Основной поток с прогрессом и кэшем
    # ------------------------------------------------------------
    def _run_pca_thread(self):
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

            data_type = self.data_type.get()

            def update_progress(page, total, _):
                if total > 0:
                    self.main_app.set_progress(int(page / total * 100))

            self.main_app.set_progress(0)
            epochs = get_epochs(
                self.api_url.get(), self.token.get(),
                study_ids=study_ids,
                data_type=data_type,
                stop_check=lambda: self.stop_flag,
                progress_callback=update_progress,
                use_cache=self.use_cache.get()
            )
            self.main_app.set_progress(100)

            if not epochs:
                self.log("Нет данных.")
                return
            df = pd.DataFrame(epochs)
            self.log(f"Загружено {len(df)} эпох")

            # Выбор каналов и признаков
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

            self.results = {
                'pca': pca,
                'X_pca': X_pca,
                'meta': meta,
                'y': y,
                'feature_names': feature_cols,
                'scaler': scaler
            }

            self._plot_scatter(X_pca, meta)
            self._plot_variance(pca)
            self._plot_loadings(pca, feature_cols, n_comp)
            if self.use_logreg.get():
                self._run_logistic_regression(X_pca, y, meta['patient_id'].values, pca)
            else:
                self._clear_coef_tab()

            self.save_csv_btn.config(state=tk.NORMAL)
            self.save_plot_btn.config(state=tk.NORMAL)
            self.log("PCA завершён.")

        except Exception as e:
            self.log(f"Ошибка: {e}")
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------
    # Визуализация
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

    def _run_logistic_regression(self, X_pca, y, patients, pca):
        n_comp = X_pca.shape[1]
        self.log(f"Логистическая регрессия на {n_comp} ПК, кросс-валидация по {self.logreg_cv_folds.get()} фолдам")
        auc_single = []
        for i in range(n_comp):
            auc = roc_auc_score(y, X_pca[:, i])
            auc_single.append(auc)
        best_pc = np.argmax(auc_single)
        self.log(f"Лучшая отдельная ПК: PC{best_pc+1} (AUC={auc_single[best_pc]:.3f})")

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

        lr.fit(X_pca, y)
        intercept = lr.intercept_[0]
        coefs = lr.coef_[0]
        coef_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(n_comp)],
            'Coefficient': coefs,
            'Odds_ratio': np.exp(coefs)
        })
        self.results['logreg_intercept'] = intercept
        self.results['logreg_coef'] = coef_df
        self.results['logreg_auc'] = mean_auc
        self.results['logreg_model'] = lr

        eq_parts = [f"logit(p) = {intercept:.4f}"]
        for i, (coef, comp) in enumerate(zip(coefs, coef_df['Component'])):
            sign = '+' if coef >= 0 else '-'
            eq_parts.append(f" {sign} {abs(coef):.4f} * {comp}")
        equation = "".join(eq_parts)
        self.log("=== Уравнение логистической регрессии ===")
        self.log(equation)
        self.log("Отношения шансов (odds ratios):")
        for _, row in coef_df.iterrows():
            self.log(f"  {row['Component']}: {row['Odds_ratio']:.4f}")

        self._display_coef_table(coef_df, intercept)
        self._plot_roc_curves(y, X_pca[:, best_pc], lr.predict_proba(X_pca)[:, 1], auc_single[best_pc], roc_auc_score(y, lr.predict_proba(X_pca)[:, 1]))

    def _plot_roc_curves(self, y, pc_best, lr_proba, auc_pc, auc_lr):
        container = self.plot_frames['roc']
        for widget in container.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        fpr_pc, tpr_pc, _ = roc_curve(y, pc_best)
        ax.plot(fpr_pc, tpr_pc, label=f'Best PC (AUC={auc_pc:.3f})')
        fpr_lr, tpr_lr, _ = roc_curve(y, lr_proba)
        ax.plot(fpr_lr, tpr_lr, label=f'LogReg on PCs (AUC={auc_lr:.3f})')
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
        label = ttk.Label(container, text="Логистическая регрессия не выполнялась.\nВключите опцию 'Выполнить логистическую регрессию' и запустите PCA.")
        label.pack(expand=True)

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
        if 'logreg_coef' in self.results:
            coef_path = file_path.replace('.csv', '_logreg_coef.csv')
            coef_df = self.results['logreg_coef'].copy()
            coef_df.loc[-1] = ['Intercept', self.results['logreg_intercept'], np.exp(self.results['logreg_intercept'])]
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