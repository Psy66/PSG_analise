# data_processor.py

from core.config import QUALITY_ORDER, SEVERITY_MAP

def apply_filters(df, settings):
    """
    Применяет критерии включения/исключения.
    settings – объект (например, GUI) с булевыми переменными и значениями фильтров.
    Возвращает отфильтрованный DataFrame.
    """
    # Критерии включения
    if settings.include_age.get():
        df = df[df['age_at_study'].between(settings.age_min.get(), settings.age_max.get(), inclusive='both')]
    if settings.include_duration.get():
        df = df[df['duration_minutes'] >= settings.duration_min.get()]
    if settings.include_quality.get():
        min_q = settings.quality_min.get()
        df = df[df['rec_quality'].map(lambda x: QUALITY_ORDER.get(x, 0) >= QUALITY_ORDER.get(min_q, 0))]
    if settings.include_ahi.get():
        df = df[df['ahi'].notna()]
    if settings.include_ahi_range.get():
        df = df[df['ahi'].notna() & df['ahi'].between(settings.ahi_min.get(), settings.ahi_max.get())]
    if settings.include_odi_range.get():
        df = df[df['odi'].notna() & df['odi'].between(settings.odi_min.get(), settings.odi_max.get())]
    if settings.include_sleep_eff.get():
        df = df[df['sleep_efficiency'].notna() & (df['sleep_efficiency'] >= settings.sleep_eff_min.get())]
    if settings.include_tst.get():
        df = df[df['total_sleep_time'].notna() & (df['total_sleep_time'] >= settings.tst_min.get())]
    if settings.include_spo2.get():
        spo2_col = 'avg_spo2' if settings.spo2_type.get() == 'avg' else 'min_spo2'
        df = df[df[spo2_col].notna() & (df[spo2_col] >= settings.spo2_min.get())]
    if settings.require_hypnogram.get():
        df = df[df['missing_hypnogram'] == 0]

    # Критерии исключения (пациент исключается, если условие выполняется)
    if settings.exclude_hf.get():
        df = df[df['cvd_heart_failure'] != 1]
    if settings.exclude_neuro.get():
        neuro_mask = (df['neuro_epilepsy'] == 1) | (df['neuro_dementia'] == 1) | (df['neuro_parkinson'] == 1)
        df = df[~neuro_mask]
    if settings.exclude_stroke_tbi.get():
        stroke_tbi_mask = (df['exclude_stroke'] == 1) | (df['exclude_tbi'] == 1)
        df = df[~stroke_tbi_mask]
    if settings.exclude_rls_plmd.get():
        rls_plmd_mask = (df['som_rls'] == 1) | (df['som_plmd'] == 1)
        df = df[~rls_plmd_mask]
    if settings.exclude_bmi.get():
        df = df[df['bmi'].notna() & (df['bmi'] <= settings.bmi_max.get())]
    if settings.exclude_age_gt70.get():
        df = df[df['age_at_study'] <= 70]
    if settings.exclude_psycho.get():
        df = df[df['exclude_psychotropics'] != 1]
    if settings.exclude_manual.get():
        df = df[df['is_excluded'] != 1]
    if settings.exclude_hypertension.get():
        df = df[df['cvd_hypertension'] != 1]
    if settings.exclude_ihd.get():
        df = df[df['cvd_ihd'] != 1]
    if settings.exclude_diabetes.get():
        df = df[df['endocrine_diabetes'] != 1]
    if settings.exclude_insomnia.get():
        df = df[df['som_insomnia'] != 1]
    if settings.exclude_copd.get():
        df = df[df['resp_copd'] != 1]
    if settings.exclude_asthma.get():
        df = df[df['resp_asthma'] != 1]
    if settings.exclude_ckd.get():
        df = df[df['nephro_ckd'] != 1]
    if settings.exclude_anxiety_depression.get():
        df = df[(df['neuro_anxiety'] != 1) & (df['neuro_depression'] != 1)]

    return df

def calc_group_stats(df):
    """Вычисляет описательные статистики для группы DataFrame."""
    n = len(df)
    age = df['age_at_study'].dropna()
    age_str = f"{age.mean():.1f}±{age.std():.1f}" if not age.empty else "NA"
    male_pct = (df['gender'] == 'M').mean() * 100 if 'gender' in df else 0

    ahi = df['ahi'].dropna()
    ahi_str = f"{ahi.mean():.2f}±{ahi.std():.2f}" if not ahi.empty else "NA"
    odi = df['odi'].dropna()
    odi_str = f"{odi.mean():.2f}±{odi.std():.2f}" if not odi.empty else "NA"

    min_spo2 = df['min_spo2'].dropna()
    min_spo2_str = f"{min_spo2.mean():.1f}±{min_spo2.std():.1f}" if not min_spo2.empty else "NA"
    avg_spo2 = df['avg_spo2'].dropna()
    avg_spo2_str = f"{avg_spo2.mean():.1f}±{avg_spo2.std():.1f}" if not avg_spo2.empty else "NA"

    tst = df['total_sleep_time'].dropna()
    tst_str = f"{tst.mean():.0f}±{tst.std():.0f}" if not tst.empty else "NA"
    eff = df['sleep_efficiency'].dropna()
    eff_str = f"{eff.mean():.1f}±{eff.std():.1f}" if not eff.empty else "NA"

    hypertension = (df['cvd_hypertension'] == 1).mean() * 100
    ihd = (df['cvd_ihd'] == 1).mean() * 100
    diabetes = (df['endocrine_diabetes'] == 1).mean() * 100
    insomnia = (df['som_insomnia'] == 1).mean() * 100

    return {
        'n': n,
        'age': age_str,
        'male_pct': f"{male_pct:.1f}",
        'ahi': ahi_str,
        'odi': odi_str,
        'min_spo2': min_spo2_str,
        'avg_spo2': avg_spo2_str,
        'tst': tst_str,
        'eff': eff_str,
        'hypertension': f"{hypertension:.1f}",
        'ihd': f"{ihd:.1f}",
        'diabetes': f"{diabetes:.1f}",
        'insomnia': f"{insomnia:.1f}"
    }

def prepare_severity_groups(df, include_central_mixed):
    """
    Добавляет колонку severity_label и фильтрует группы в соответствии с опцией.
    Возвращает DataFrame с колонкой severity_label.
    """
    df = df.copy()
    df['severity_label'] = df['breathing_impairment_severity'].map(SEVERITY_MAP).fillna(df['breathing_impairment_severity'])
    if not include_central_mixed:
        allowed_groups = ['Норма (<5)', 'Лёгкая (5-14.9)', 'Умеренная (15-29.9)', 'Тяжёлая (≥30)']
        df = df[df['severity_label'].isin(allowed_groups)]
    return df