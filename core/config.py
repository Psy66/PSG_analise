"""
config.py
---------
Централизованное хранение всех параметров для модульной обработки ПСГ.
"""
import os
# ========== ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ==========
MAX_PARALLEL_WORKERS = 8   # количество процессов (для i9-12900 можно 8-12)

# ========== ПУТИ И ФАЙЛЫ ==========
CSV_PATH = r"D:\PythonProject\PSG_analise\results\full2.csv" # файл с метаданными пациентов и исследований
EDF_DIR = r"F:\psg\edf"                  # директория с EDF-файлами
TONIC_OUTPUT_CSV = "tonic_epochs_features.csv"   # выходной CSV для тонических признаков
PHASIC_OUTPUT_CSV = "phasic_events_features.csv" # выходной CSV для фазических признаков
ALL_EPOCHS_OUTPUT_CSV = "all_epochs_features.csv" # выходной CSV для всех эпох (LMM)
LOG_FILE = "processing_log.txt"          # общий файл логов
OUTPUT_DIR = "results"          # можно изменить на любой путь, например "/path/to/my_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== РЕЖИМ ТЕСТИРОВАНИЯ ==========
TEST_MODE = False          # True – обрабатывать только MAX_PATIENTS пациентов
MAX_PATIENTS = 4          # количество пациентов для тестирования (если TEST_MODE = True)

# ========== КЭШИРОВАНИЕ ==========
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")   # теперь кэш будет внутри OUTPUT_DIR/cache
CACHE_EDF_DIR = os.path.join(OUTPUT_DIR, "edf_cache")
USE_CACHE = True          # если True, пропускать уже обработанных пациентов
TONIC_PROGRESS_FILE = os.path.join(CACHE_DIR, "processed_tonic.txt")      # для чистых тонических эпох
PHASIC_PROGRESS_FILE = os.path.join(CACHE_DIR, "processed_phasic.txt")    # для фазических событий
ALL_EPOCHS_PROGRESS_FILE = os.path.join(CACHE_DIR, "processed_all_epochs.txt") # для всех эпох

# ========== КЭШИРОВАНИЕ API ==========
CACHE_API_DIR = os.path.join(OUTPUT_DIR, "cache", "api")   # папка для кэша API
CACHE_TTL_SECONDS = 86400   # 24 часа, время жизни кэша

# ========== ЗАГРУЗКА EDF ==========
ENCODINGS = ['utf-8', 'latin1', 'cp1251']   # порядок перебора кодировок

# ========== ПРЕДОБРАБОТКА ЭЭГ ==========
EEG_FILTER_LOW = 0.5       # Гц, нижняя частота полосового фильтра
EEG_FILTER_HIGH = 45.0     # Гц, верхняя частота полосового фильтра

# ========== ЭПОХИ / ОКНА ==========
EPOCH_DURATION = 30.0      # секунд, длительность анализируемой эпохи
DURATION_TOLERANCE = 0.1   # секунд, допустимое отклонение длительности от EPOCH_DURATION

# === Стадии сна ===
STAGE_MAP = {
    'Sleep stage W(eventUnknown)': 'Wake',
    'Sleep stage 1(eventUnknown)': 'N1',
    'Sleep stage 2(eventUnknown)': 'N2',
    'Sleep stage 3(eventUnknown)': 'N3',
    'Sleep stage R(eventUnknown)': 'REM'
}
STAGES_OF_INTEREST = ('N2', 'N3')        # для тонического и фазического анализа

# ========== АРТЕФАКТЫ ==========
ARTIFACT_MARKERS = ['blockArtefact']     # маркеры, указывающие на артефактные участки

# ========== РЕСПИРАТОРНЫЕ СОБЫТИЯ ==========
RESP_KEYWORDS = [
    'Обструктивное апноэ', 'Центральное апноэ', 'Смешанное апноэ',
    'Обструктивное гипопноэ', 'Центральное гипопноэ', 'Смешанное гипопноэ'
]

# ========== ЧАСТОТНЫЕ ДИАПАЗОНЫ (Гц) ==========
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'sigma': (12, 15),
    'beta': (15, 30)
}
TOTAL_BAND = (0.5, 45)       # полный диапазон для спектрального анализа
GAMMA_BAND = (30, 45)        # гамма-диапазон для фазических признаков

# ========== ПАРАМЕТРЫ ВЕЛЧА (спектральная оценка) ==========
WELCH_NPERSEG_FACTOR = 256   # длина сегмента для БПФ (nperseg = min(256, длина данных))
WELCH_NOVERLAP_FACTOR = 128  # перекрытие между сегментами

# ========== МАСШТАБИРОВАНИЕ СИГНАЛА ==========
SCALE_TO_MICROVOLTS = 1e6    # вольты -> микровольты

# ========== ЭНТРОПИЯ ==========
INCLUDE_SAMPLE_ENTROPY = True   # добавлять ли Sample Entropy в тонические признаки
SAMPEN_M = 2                   # порядок энтропии (длина шаблона)
SAMPEN_R_FACTOR = 0.2          # коэффициент r = r_factor * std(signal) для SampEn

# ========== ТОНИЧЕСКИЙ АНАЛИЗ ==========
TONIC_BUFFER_SEC = 30.0        # запретная зона вокруг респираторных событий (сек)
POSITION_FILTER_MAX_START_SEC = 10.0   # секунд, начало события должно быть в пределах первых 10 с эпохи

# ========== ФАЗИЧЕСКИЙ АНАЛИЗ ==========
PHASIC_CHANNELS = ['C3', 'C4']   # отведения, для которых вычисляются фазические признаки
BASELINE_DURATION = 10.0         # секунд, длительность baseline-окна
BASELINE_LEAD_SEC = 60.0         # секунд, за сколько до начала события заканчивается baseline
EVENT_DURATION = 10.0            # секунд, длительность event-окна после окончания события

# ========== ОТВЕДЕНИЯ ЭЭГ ==========
STANDARD_CHANNELS = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']  # искомые стандартные отведения

# ========== API ====================
DEFAULT_API_URL = "https://med.smed96.synology.me/psgapp/api/"
DEFAULT_TOKEN = ""
TIMEOUT = 90   # таймаут для обычных запросов (сек)
MAX_RETRIES = 3
PAGE_SIZE = 2000
DOWNLOAD_TIMEOUT = 600              # 10 минут на передачу всего файла
DOWNLOAD_CHUNK_SIZE = 64 * 1024      # 64 KB на чанк (лучше для больших файлов)
DOWNLOAD_MAX_RETRIES = 3
BACKOFF_FACTOR = 2

QUALITY_ORDER = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1} # Порядок качества записи (для фильтрации)

# Отображение severity в читаемые метки
SEVERITY_MAP = {
    'no_impairment': 'Норма (<5)',
    'mild': 'Лёгкая (5-14.9)',
    'moderate': 'Умеренная (15-29.9)',
    'severe': 'Тяжёлая (≥30)',
    'central': 'Центральное',
    'mixed': 'Смешанное',
    'none': 'Нет',
    'unknown': 'Неизвестно'
}

# ========== ВРЕМЕННЫЕ РЯДЫ ГАММА-АКТИВНОСТИ ==========
EVENT_TIME_SERIES_WINDOW_SEC = 2.0      # секунд, окно для расчёта спектра
EVENT_TIME_SERIES_STEP_SEC = 1.0        # шаг скольжения
EVENT_TIME_SERIES_START_SEC = -60.0     # время относительно offset события (начало)
EVENT_TIME_SERIES_END_SEC = 30.0        # время относительно offset (конец)
EVENT_TIME_SERIES_BG_START = -60.0      # интервал фона (начало) для нормализации
EVENT_TIME_SERIES_BG_END = -30.0        # интервал фона (конец)
