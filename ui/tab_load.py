# ui/tab_load.py
"""
Вкладка загрузки и фильтрации ПСГ-исследований из API.
Содержит панель подключения к серверу (фиксированную) и прокручиваемую область
с критериями включения/исключения. После загрузки и фильтрации передаёт
отфильтрованный DataFrame в главное окно.
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import pandas as pd

from ui.base_tab import BaseTab
from core.api_client import get_studies, get_patients, get_clinical
from core.data_processor import apply_filters
from core.config import DEFAULT_API_URL, DEFAULT_TOKEN


class LoadDataTab(BaseTab):
    """
    Вкладка загрузки и фильтрации ПСГ-исследований.
    Содержит все критерии включения/исключения, кнопку загрузки.
    После загрузки и фильтрации передаёт отфильтрованный DataFrame в главное окно.
    """

    def __init__(self, parent, main_app):
        """
        Инициализация вкладки.

        Параметры:
            parent: родительский виджет (вкладка Notebook)
            main_app: ссылка на главное окно (MainWindow) для доступа к общим методам
        """
        super().__init__(parent, main_app)
        self.parent_window = parent.winfo_toplevel()  # корневое окно для глобальной привязки событий

        # ---- API настройки (значения по умолчанию из конфига) ----
        self.api_url = tk.StringVar(value=DEFAULT_API_URL)
        self.token = tk.StringVar(value=DEFAULT_TOKEN)

        # ---- Критерии включения ----
        self.include_age = tk.BooleanVar(value=True)
        self.age_min = tk.IntVar(value=18)
        self.age_max = tk.IntVar(value=70)

        self.include_duration = tk.BooleanVar(value=True)
        self.duration_min = tk.IntVar(value=360)

        self.include_quality = tk.BooleanVar(value=True)
        self.quality_min = tk.StringVar(value="good")

        self.include_ahi = tk.BooleanVar(value=True)

        self.include_ahi_range = tk.BooleanVar(value=False)
        self.ahi_min = tk.DoubleVar(value=0.0)
        self.ahi_max = tk.DoubleVar(value=100.0)

        self.include_odi_range = tk.BooleanVar(value=False)
        self.odi_min = tk.DoubleVar(value=0.0)
        self.odi_max = tk.DoubleVar(value=100.0)

        self.include_sleep_eff = tk.BooleanVar(value=False)
        self.sleep_eff_min = tk.DoubleVar(value=70.0)

        self.include_tst = tk.BooleanVar(value=False)
        self.tst_min = tk.IntVar(value=240)

        self.include_spo2 = tk.BooleanVar(value=False)
        self.spo2_type = tk.StringVar(value="avg")
        self.spo2_min = tk.DoubleVar(value=90.0)

        self.require_hypnogram = tk.BooleanVar(value=False)

        # ---- Критерии исключения ----
        self.exclude_hf = tk.BooleanVar(value=True)
        self.exclude_neuro = tk.BooleanVar(value=True)
        self.exclude_stroke_tbi = tk.BooleanVar(value=True)
        self.exclude_rls_plmd = tk.BooleanVar(value=True)
        self.exclude_bmi = tk.BooleanVar(value=True)
        self.bmi_max = tk.IntVar(value=40)
        self.exclude_age_gt70 = tk.BooleanVar(value=True)
        self.exclude_psycho = tk.BooleanVar(value=True)
        self.exclude_manual = tk.BooleanVar(value=True)
        self.exclude_hypertension = tk.BooleanVar(value=False)
        self.exclude_ihd = tk.BooleanVar(value=False)
        self.exclude_diabetes = tk.BooleanVar(value=False)
        self.exclude_insomnia = tk.BooleanVar(value=False)
        self.exclude_copd = tk.BooleanVar(value=False)
        self.exclude_asthma = tk.BooleanVar(value=False)
        self.exclude_ckd = tk.BooleanVar(value=False)
        self.exclude_anxiety_depression = tk.BooleanVar(value=False)

        # ---- Группировка (для статистики, настройки остаются здесь) ----
        self.group_by_severity = tk.BooleanVar(value=True)
        self.include_central_mixed = tk.BooleanVar(value=False)

        # ---- Данные ----
        self.studies_df = None
        self.filtered_df = None
        self.stop_flag = False

        # ---- Создаём интерфейс ----
        self._create_widgets()

    # ------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------
    def _create_widgets(self):
        """
        Создаёт все виджеты вкладки:
        - Верхняя фиксированная панель с настройками API
        - Прокручиваемая область с критериями включения/исключения, кнопками и меткой
        """
        # Основной контейнер, заполняющий всю вкладку
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ========== 1. Верхняя фиксированная панель (настройки API) ==========
        frame_api = ttk.LabelFrame(main_frame, text="Подключение к серверу", padding=10)
        frame_api.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame_api, text="Адрес API:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(frame_api, textvariable=self.api_url, width=100).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(frame_api, text="Токен (Bearer):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(frame_api, textvariable=self.token, width=100, show="*").grid(row=1, column=1, padx=5, pady=2)

        self.load_btn = ttk.Button(frame_api, text="Загрузить данные исследований", command=self.load_studies)
        self.load_btn.grid(row=2, column=0, columnspan=2, pady=5)

        # ========== 2. Прокручиваемая область ==========
        canvas = tk.Canvas(main_frame, bg='#f0f0f0', highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollable = ttk.Frame(canvas)  # фрейм, который будет содержать всё прокручиваемое содержимое

        # Функция прокрутки колесом мыши
        def _on_mousewheel(event):
            # Проверяем, что активна именно эта вкладка
            current_tab = self.main_app.notebook.select()
            current_widget = self.main_app.notebook.nametowidget(current_tab)
            if current_widget == self:
                if event.delta:
                    # Windows: event.delta = ±120
                    canvas.yview_scroll(int(-event.delta / 120), "units")
                elif event.num == 4:
                    canvas.yview_scroll(-1, "units")   # Linux
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")    # Linux

        # Привязываем события к корневому окну (глобально)
        self.parent_window.bind("<MouseWheel>", _on_mousewheel)
        self.parent_window.bind("<Button-4>", _on_mousewheel)
        self.parent_window.bind("<Button-5>", _on_mousewheel)

        # Обновляем область прокрутки при изменении размера содержимого
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ========== 3. Содержимое прокручиваемой области ==========
        # ----- Критерии включения -----
        frame_inc = ttk.LabelFrame(scrollable, text="Критерии включения", padding=10)
        frame_inc.pack(fill=tk.X, padx=10, pady=5)

        row_inc = 0
        ttk.Checkbutton(frame_inc, text="Возраст от", variable=self.include_age).grid(row=row_inc, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(frame_inc, from_=0, to=120, textvariable=self.age_min, width=5).grid(row=row_inc, column=1, padx=2)
        ttk.Label(frame_inc, text="до").grid(row=row_inc, column=2, padx=2)
        ttk.Spinbox(frame_inc, from_=0, to=120, textvariable=self.age_max, width=5).grid(row=row_inc, column=3, padx=2)

        row_inc += 1
        ttk.Checkbutton(frame_inc, text="Длительность записи ≥", variable=self.include_duration).grid(row=row_inc, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(frame_inc, from_=0, to=1000, textvariable=self.duration_min, width=6).grid(row=row_inc, column=1, padx=2)
        ttk.Label(frame_inc, text="минут").grid(row=row_inc, column=2, sticky=tk.W)

        row_inc += 1
        ttk.Checkbutton(frame_inc, text="Качество записи не ниже", variable=self.include_quality).grid(row=row_inc, column=0, sticky=tk.W, padx=5)
        quality_combo = ttk.Combobox(frame_inc, textvariable=self.quality_min, values=['excellent', 'good', 'fair', 'poor'], width=10)
        quality_combo.grid(row=row_inc, column=1, sticky=tk.W, padx=2)
        ttk.Label(frame_inc, text="(excellent > good > fair > poor)").grid(row=row_inc, column=2, sticky=tk.W, padx=5)

        row_inc += 1
        ttk.Checkbutton(frame_inc, text="Только исследования с рассчитанным AHI", variable=self.include_ahi).grid(row=row_inc, column=0, sticky=tk.W, padx=5, columnspan=3)

        row_inc += 1
        ttk.Checkbutton(frame_inc, text="AHI в диапазоне от", variable=self.include_ahi_range).grid(row=row_inc, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame_inc, textvariable=self.ahi_min, width=6).grid(row=row_inc, column=1, padx=2)
        ttk.Label(frame_inc, text="до").grid(row=row_inc, column=2)
        ttk.Entry(frame_inc, textvariable=self.ahi_max, width=6).grid(row=row_inc, column=3, padx=2)

        row_inc += 1
        ttk.Checkbutton(frame_inc, text="ODI в диапазоне от", variable=self.include_odi_range).grid(row=row_inc, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame_inc, textvariable=self.odi_min, width=6).grid(row=row_inc, column=1, padx=2)
        ttk.Label(frame_inc, text="до").grid(row=row_inc, column=2)
        ttk.Entry(frame_inc, textvariable=self.odi_max, width=6).grid(row=row_inc, column=3, padx=2)

        row_inc += 1
        ttk.Checkbutton(frame_inc, text="Эффективность сна ≥", variable=self.include_sleep_eff).grid(row=row_inc, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame_inc, textvariable=self.sleep_eff_min, width=6).grid(row=row_inc, column=1, padx=2)
        ttk.Label(frame_inc, text="%").grid(row=row_inc, column=2, sticky=tk.W)

        row_inc += 1
        ttk.Checkbutton(frame_inc, text="Общее время сна (TST) ≥", variable=self.include_tst).grid(row=row_inc, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame_inc, textvariable=self.tst_min, width=6).grid(row=row_inc, column=1, padx=2)
        ttk.Label(frame_inc, text="минут").grid(row=row_inc, column=2, sticky=tk.W)

        row_inc += 1
        ttk.Checkbutton(frame_inc, text="SpO₂ (", variable=self.include_spo2).grid(row=row_inc, column=0, sticky=tk.W, padx=5)
        spo2_combo = ttk.Combobox(frame_inc, textvariable=self.spo2_type, values=['avg', 'min'], width=5)
        spo2_combo.grid(row=row_inc, column=1, sticky=tk.W, padx=2)
        ttk.Label(frame_inc, text=") ≥").grid(row=row_inc, column=2, sticky=tk.W, padx=2)
        ttk.Entry(frame_inc, textvariable=self.spo2_min, width=6).grid(row=row_inc, column=3, padx=2)
        ttk.Label(frame_inc, text="%").grid(row=row_inc, column=4, sticky=tk.W)

        row_inc += 1
        ttk.Checkbutton(frame_inc, text="Только исследования с рассчитанной гипнограммой", variable=self.require_hypnogram).grid(row=row_inc, column=0, sticky=tk.W, padx=5, columnspan=4)

        # ----- Критерии исключения -----
        frame_exc = ttk.LabelFrame(scrollable, text="Критерии исключения (пациент исключается, если отмеченное условие выполняется)", padding=10)
        frame_exc.pack(fill=tk.X, padx=10, pady=5)

        ttk.Checkbutton(frame_exc, text="Хроническая сердечная недостаточность (ХСН)", variable=self.exclude_hf).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_exc, text="Эпилепсия, деменция или болезнь Паркинсона", variable=self.exclude_neuro).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_exc, text="Инсульт или черепно-мозговая травма в анамнезе", variable=self.exclude_stroke_tbi).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_exc, text="Синдром беспокойных ног / периодические движения конечностей", variable=self.exclude_rls_plmd).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame_exc, text="Индекс массы тела (ИМТ) >", variable=self.exclude_bmi).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(frame_exc, from_=0, to=100, textvariable=self.bmi_max, width=5).grid(row=1, column=1, sticky=tk.W, padx=2)
        ttk.Checkbutton(frame_exc, text="Возраст старше 70 лет", variable=self.exclude_age_gt70).grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_exc, text="Приём психотропных, противоэпилептических или снотворных", variable=self.exclude_psycho).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame_exc, text="Ручная метка исключения (проставлена врачом)", variable=self.exclude_manual).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2, columnspan=2)

        ttk.Checkbutton(frame_exc, text="Артериальная гипертензия", variable=self.exclude_hypertension).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_exc, text="Ишемическая болезнь сердца (ИБС)", variable=self.exclude_ihd).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_exc, text="Сахарный диабет 2 типа", variable=self.exclude_diabetes).grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame_exc, text="Инсомния", variable=self.exclude_insomnia).grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_exc, text="ХОБЛ", variable=self.exclude_copd).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_exc, text="Бронхиальная астма", variable=self.exclude_asthma).grid(row=4, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_exc, text="Хроническая болезнь почек", variable=self.exclude_ckd).grid(row=4, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(frame_exc, text="Тревожное расстройство или депрессия", variable=self.exclude_anxiety_depression).grid(row=5, column=0, sticky=tk.W, padx=5, pady=2, columnspan=2)

        # ----- Кнопки управления фильтрами -----
        btn_frame = ttk.Frame(scrollable)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Применить фильтры и обновить данные",
                                  command=self.apply_filters_and_update)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        reset_btn = ttk.Button(btn_frame, text="Сбросить все фильтры", command=self.reset_filters)
        reset_btn.pack(side=tk.LEFT, padx=5)

        # ----- Информационная метка -----
        self.info_label = ttk.Label(scrollable, text="")
        self.info_label.pack(pady=5)

    # ------------------------------------------------------------
    # Загрузка данных из API
    # ------------------------------------------------------------
    def load_studies(self):
        """
        Загружает данные исследований, пациентов и клиническую информацию из API.
        Выполняется в отдельном потоке, чтобы не блокировать GUI.
        Результат сохраняется в self.studies_df и отображается в общей таблице.
        """
        api_url = self.api_url.get().rstrip('/')
        token = self.token.get().strip()
        if not api_url or not token:
            messagebox.showerror("Ошибка", "Укажите адрес API и токен")
            return

        def task():
            self.stop_flag = False
            self.load_btn.config(state=tk.DISABLED)
            self.run_btn.config(state=tk.DISABLED)
            self.set_progress(0)
            self.log("Загрузка списка исследований...")

            try:
                # 1. Загружаем исследования
                studies = get_studies(
                    api_url, token,
                    stop_check=lambda: self.stop_flag,
                    progress_callback=lambda page, total, cnt: self.log(f"Страница {page}/{total}, получено {cnt} записей")
                )
                if not studies:
                    self.log("Не удалось загрузить исследования.")
                    return
                self.log(f"Загружено {len(studies)} исследований.")
                self.set_progress(20)

                # 2. Пациенты
                self.log("Загрузка данных пациентов...")
                patients = get_patients(api_url, token, stop_check=lambda: self.stop_flag)
                patient_info = {}
                for p in patients:
                    pid = p.get('patient_id')
                    if pid:
                        patient_info[pid] = {
                            'birth_date': p.get('birth_date'),
                            'gender': p.get('gender')
                        }
                self.log(f"Загружено {len(patient_info)} пациентов.")
                self.set_progress(40)

                # 3. Клинические данные (берём только последнюю запись на пациента)
                self.log("Загрузка клинических данных...")
                all_clinical = get_clinical(api_url, token, stop_check=lambda: self.stop_flag)
                all_clinical_sorted = sorted(all_clinical, key=lambda x: x.get('clinical_id', 0), reverse=True)
                clinical_by_patient = {}
                for rec in all_clinical_sorted:
                    pid = rec.get('patient_id')
                    if pid and pid not in clinical_by_patient:
                        clinical_by_patient[pid] = rec
                self.log(f"Загружено клинических записей: {len(all_clinical)}, уникальных: {len(clinical_by_patient)}")
                self.set_progress(60)

                # 4. Сборка DataFrame
                records = []
                total = len(studies)
                for i, study in enumerate(studies):
                    if self.stop_flag:
                        self.log("Загрузка прервана пользователем")
                        return

                    pid = study.get('patient_id')
                    clin = clinical_by_patient.get(pid, {})
                    pat = patient_info.get(pid, {})

                    birth_date = pat.get('birth_date')
                    study_date = study.get('study_date')
                    age = None
                    if birth_date and study_date:
                        try:
                            birth = datetime.strptime(birth_date, '%Y-%m-%d')
                            study_dt = datetime.strptime(study_date, '%Y-%m-%d')
                            age = study_dt.year - birth.year - ((study_dt.month, study_dt.day) < (birth.month, birth.day))
                        except:
                            pass

                    record = {
                        'study_id': study.get('study_id'),
                        'patient_id': pid,
                        'age_at_study': age,
                        'gender': pat.get('gender'),
                        'study_date': study.get('study_date'),
                        'duration_minutes': study.get('edf_duration'),
                        'rec_quality': study.get('rec_quality'),
                        'ahi': study.get('ahi'),
                        'odi': study.get('odi'),
                        'total_sleep_time': study.get('total_sleep_time'),
                        'sleep_efficiency': study.get('sleep_efficiency'),
                        'min_spo2': study.get('min_spo2'),
                        'avg_spo2': study.get('avg_spo2'),
                        'breathing_impairment_severity': study.get('breathing_impairment_severity'),
                        'data_quality': study.get('sleep_data_quality'),
                        'hypnogram_data': study.get('hypnogram_data'),
                        'bmi': clin.get('bmi'),
                        'cvd_hypertension': clin.get('cvd_hypertension'),
                        'cvd_ihd': clin.get('cvd_ihd'),
                        'cvd_heart_failure': clin.get('cvd_heart_failure'),
                        'endocrine_diabetes': clin.get('endocrine_diabetes'),
                        'som_insomnia': clin.get('som_insomnia'),
                        'neuro_epilepsy': clin.get('neuro_epilepsy'),
                        'neuro_dementia': clin.get('neuro_dementia'),
                        'neuro_parkinson': clin.get('neuro_parkinson'),
                        'exclude_stroke': clin.get('exclude_stroke'),
                        'exclude_tbi': clin.get('exclude_tbi'),
                        'som_rls': clin.get('som_rls'),
                        'som_plmd': clin.get('som_plmd'),
                        'exclude_psychotropics': clin.get('exclude_psychotropics'),
                        'is_excluded': clin.get('is_excluded'),
                        'resp_copd': clin.get('resp_copd'),
                        'resp_asthma': clin.get('resp_asthma'),
                        'nephro_ckd': clin.get('nephro_ckd'),
                        'neuro_anxiety': clin.get('neuro_anxiety'),
                        'neuro_depression': clin.get('neuro_depression'),
                        'missing_hypnogram': study.get('missing_hypnogram', 1),
                        'file_path': study.get('file_path'),
                    }
                    records.append(record)
                    if i % 50 == 0:
                        self.set_progress(60 + int(30 * i / total))

                self.studies_df = pd.DataFrame(records)
                numeric_cols = ['ahi', 'odi', 'total_sleep_time', 'sleep_efficiency', 'min_spo2', 'avg_spo2',
                                'bmi', 'age_at_study', 'duration_minutes']
                for col in numeric_cols:
                    self.studies_df[col] = pd.to_numeric(self.studies_df[col], errors='coerce')

                self.log(f"Сформирован DataFrame, строк: {len(self.studies_df)}")
                self.run_btn.config(state=tk.NORMAL)
                self.set_progress(100)
                self.info_label.config(text=f"Загружено {len(self.studies_df)} исследований. Нажмите 'Применить фильтры'.")
                self.log("Загрузка завершена. Теперь можно применить фильтры.")
                # Сохраняем загруженные данные в главном окне
                self.main_app.set_filtered_data(self.studies_df)

            except Exception as e:
                self.log(f"Ошибка загрузки: {e}")
                messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")
            finally:
                self.load_btn.config(state=tk.NORMAL)

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------
    # Применение фильтров
    # ------------------------------------------------------------
    def apply_filters_and_update(self):
        """
        Применяет все установленные критерии включения/исключения к загруженным данным.
        Результат (отфильтрованный DataFrame) передаётся в главное окно и сохраняется.
        Также передаются настройки группировки для вкладки статистики.
        """
        if self.studies_df is None or self.studies_df.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные исследований.")
            return

        self.log("Применение фильтров...")
        self.filtered_df = apply_filters(self.studies_df.copy(), self)
        self.log(f"После фильтрации осталось {len(self.filtered_df)} записей.")

        # Сохраняем отфильтрованные данные
        self.main_app.set_filtered_data(self.filtered_df)

        # Настройки группировки
        group_by = self.group_by_severity.get()
        include_central = self.include_central_mixed.get()
        self.main_app.set_analysis_settings(group_by_severity=group_by, include_central_mixed=include_central)

        self.info_label.config(text=f"Отфильтровано: {len(self.filtered_df)} записей.")
        self.log("Фильтрация завершена. Переключитесь на вкладку 'Статистика' для расчёта.")

    # ------------------------------------------------------------
    # Сброс фильтров
    # ------------------------------------------------------------
    def reset_filters(self):
        """
        Сбрасывает все критерии включения/исключения и настройки группировки
        к значениям по умолчанию.
        """
        # Критерии включения
        self.include_age.set(True)
        self.age_min.set(18)
        self.age_max.set(70)
        self.include_duration.set(True)
        self.duration_min.set(360)
        self.include_quality.set(True)
        self.quality_min.set("good")
        self.include_ahi.set(True)
        self.include_ahi_range.set(False)
        self.ahi_min.set(0.0)
        self.ahi_max.set(100.0)
        self.include_odi_range.set(False)
        self.odi_min.set(0.0)
        self.odi_max.set(100.0)
        self.include_sleep_eff.set(False)
        self.sleep_eff_min.set(70.0)
        self.include_tst.set(False)
        self.tst_min.set(240)
        self.include_spo2.set(False)
        self.spo2_type.set("avg")
        self.spo2_min.set(90.0)
        self.require_hypnogram.set(False)

        # Критерии исключения
        self.exclude_hf.set(True)
        self.exclude_neuro.set(True)
        self.exclude_stroke_tbi.set(True)
        self.exclude_rls_plmd.set(True)
        self.exclude_bmi.set(True)
        self.bmi_max.set(40)
        self.exclude_age_gt70.set(True)
        self.exclude_psycho.set(True)
        self.exclude_manual.set(True)
        self.exclude_hypertension.set(False)
        self.exclude_ihd.set(False)
        self.exclude_diabetes.set(False)
        self.exclude_insomnia.set(False)
        self.exclude_copd.set(False)
        self.exclude_asthma.set(False)
        self.exclude_ckd.set(False)
        self.exclude_anxiety_depression.set(False)

        # Группировка
        self.group_by_severity.set(True)
        self.include_central_mixed.set(False)

        self.log("Все фильтры сброшены к значениям по умолчанию")