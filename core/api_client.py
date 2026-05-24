# core/api_client.py
import logging
import os
import time
from typing import Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from core.cache import get_cached_data, save_cached_data
from core.config import (
    BACKOFF_FACTOR, DOWNLOAD_CHUNK_SIZE, DOWNLOAD_TIMEOUT,
    MAX_RETRIES, PAGE_SIZE, TIMEOUT
)

logger = logging.getLogger(__name__)


def normalize_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip('/')
    endpoint = endpoint.lstrip('/')
    return f"{base}/{endpoint}"


def request_with_retry(
    method: str,
    url: str,
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    timeout: int = TIMEOUT,
    max_retries: int = MAX_RETRIES,
    stream: bool = False,
    session: Optional[requests.Session] = None
) -> Optional[requests.Response]:
    """Выполняет запрос с повторными попытками при ошибках и таймаутах."""
    if session is None:
        session = requests.Session()

    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.debug(f"Попытка {attempt+1}/{max_retries}: {method} {url}")
            resp = session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=timeout,
                stream=stream
            )
            if resp.status_code == 429:
                wait = min(BACKOFF_FACTOR ** attempt, 60)
                logger.warning(f"Rate limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout (attempt {attempt+1}/{max_retries}): {e}")
            last_exception = e
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error (attempt {attempt+1}/{max_retries}): {e}")
            last_exception = e
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
            last_exception = e

        if attempt < max_retries - 1:
            wait = min(BACKOFF_FACTOR ** attempt, 30)
            time.sleep(wait)

    logger.error(f"All {max_retries} attempts failed: {last_exception}")
    return None

def paginated_get(
    url: str,
    headers: Dict,
    params: Optional[Dict] = None,
    page_size: int = PAGE_SIZE,
    timeout: int = TIMEOUT,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    max_workers: int = 3  # количество параллельных потоков
) -> List[Dict]:
    # Загружаем первую страницу, чтобы узнать total_pages
    query_params = dict(params or {})
    query_params['limit'] = page_size
    query_params['page'] = 1

    resp = request_with_retry("GET", url, headers=headers, params=query_params, timeout=timeout)
    if not resp:
        return []

    data = resp.json()
    total_pages = data.get('meta', {}).get('pages', 1)
    all_items = data.get('data', [])

    if total_pages <= 1:
        return all_items

    # Параллельная загрузка остальных страниц
    def fetch_page(page_num):
        if stop_check and stop_check():
            return []
        p_params = dict(params or {})
        p_params['limit'] = page_size
        p_params['page'] = page_num
        resp = request_with_retry("GET", url, headers=headers, params=p_params, timeout=timeout)
        if resp:
            return resp.json().get('data', [])
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {executor.submit(fetch_page, page): page for page in range(2, total_pages + 1)}
        for future in as_completed(future_to_page):
            if stop_check and stop_check():
                executor.shutdown(wait=False, cancel_futures=True)
                break
            page = future_to_page[future]
            try:
                items = future.result()
                if items:
                    all_items.extend(items)
                if progress_callback:
                    progress_callback(page, total_pages, len(items))
            except Exception as e:
                logger.error(f"Error loading page {page}: {e}")

    return all_items


# --- API методы (без изменений в логике, только добавлен timeout) ---

def get_studies(
    api_url: str,
    token: str,
    timeout: int = TIMEOUT,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None
) -> List[Dict]:
    headers = {"Authorization": f"Bearer {token}"}
    url = normalize_url(api_url, "studies")
    return paginated_get(url, headers, timeout=timeout, stop_check=stop_check, progress_callback=progress_callback)


def get_patients(
    api_url: str,
    token: str,
    timeout: int = TIMEOUT,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None
) -> List[Dict]:
    headers = {"Authorization": f"Bearer {token}"}
    url = normalize_url(api_url, "patients")
    return paginated_get(url, headers, timeout=timeout, stop_check=stop_check, progress_callback=progress_callback)


def get_clinical(
    api_url: str,
    token: str,
    timeout: int = TIMEOUT,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None
) -> List[Dict]:
    headers = {"Authorization": f"Bearer {token}"}
    url = normalize_url(api_url, "clinical")
    return paginated_get(url, headers, timeout=timeout, stop_check=stop_check, progress_callback=progress_callback)


def get_study_info(api_url: str, token: str, study_id: int, timeout: int = TIMEOUT) -> Optional[Dict]:
    headers = {"Authorization": f"Bearer {token}"}
    url = normalize_url(api_url, f"studies/{study_id}")
    resp = request_with_retry("GET", url, headers=headers, timeout=timeout)
    if resp:
        return resp.json()
    return None


def download_edf_file_via_study_id(
    api_url: str,
    token: str,
    study_id: int,
    output_path: str,
    timeout: int = DOWNLOAD_TIMEOUT,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> bool:
    headers = {"Authorization": f"Bearer {token}"}
    url = normalize_url(api_url, f"studies/{study_id}/download")
    session = requests.Session()
    try:
        response = session.get(url, headers=headers, stream=True, timeout=timeout)
        if response.status_code != 200:
            logger.error(f"HTTP {response.status_code} for study {study_id}")
            return False

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        progress_callback(downloaded, total_size)

        if os.path.getsize(output_path) == 0:
            os.remove(output_path)
            return False
        return True
    except Exception as e:
        logger.error(f"Download error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    finally:
        session.close()


# --- Методы с кэшированием (добавлен параметр timeout) ---

def get_epochs(
    api_url: str,
    token: str,
    patient_id: Optional[int] = None,
    study_ids: Optional[List[int]] = None,
    data_type: Optional[int] = None,
    has_apnea: Optional[bool] = None,
    epoch_stage: Optional[str] = None,
    page_size: int = PAGE_SIZE,
    timeout: int = TIMEOUT,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    use_cache: bool = True,
    cache_ttl: Optional[int] = None
) -> List[Dict]:
    headers = {"Authorization": f"Bearer {token}"}
    url = normalize_url(api_url, "epochs")
    params = {}
    if patient_id is not None:
        params['patient_id'] = patient_id
    if study_ids is not None:
        for sid in study_ids:
            params.setdefault('study_id[]', []).append(sid)
    if data_type is not None:
        params['data_type'] = data_type
    if has_apnea is not None:
        params['has_apnea'] = 1 if has_apnea else 0
    if epoch_stage in ('N2', 'N3'):
        params['epoch_stage'] = epoch_stage

    if use_cache:
        cached = get_cached_data(api_url, "epochs", params, ttl=cache_ttl)
        if cached is not None:
            logger.info(f"Загружено {len(cached)} эпох из кэша")
            return cached

    data = paginated_get(url, headers, params, page_size, timeout, stop_check, progress_callback)

    if use_cache and data:
        save_cached_data(api_url, "epochs", params, data)
    return data


def get_phasic_events(
    api_url: str,
    token: str,
    patient_id: Optional[int] = None,
    study_ids: Optional[List[int]] = None,
    channel: Optional[str] = None,
    page_size: int = PAGE_SIZE,
    timeout: int = TIMEOUT,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    use_cache: bool = True,
    cache_ttl: Optional[int] = None
) -> List[Dict]:
    headers = {"Authorization": f"Bearer {token}"}
    url = normalize_url(api_url, "phasic")
    params = {}
    if patient_id is not None:
        params['patient_id'] = patient_id
    if study_ids is not None:
        for sid in study_ids:
            params.setdefault('study_id[]', []).append(sid)
    if channel in ('C3', 'C4'):
        params['channel'] = channel

    if use_cache:
        cached = get_cached_data(api_url, "phasic", params, ttl=cache_ttl)
        if cached is not None:
            logger.info(f"Загружено {len(cached)} фазических событий из кэша")
            return cached

    data = paginated_get(url, headers, params, page_size, timeout, stop_check, progress_callback)

    if use_cache and data:
        save_cached_data(api_url, "phasic", params, data)
    return data


def get_event_time_series(
    api_url: str,
    token: str,
    patient_id: Optional[int] = None,
    study_ids: Optional[List[int]] = None,
    event_id: Optional[int] = None,
    channel: Optional[str] = None,
    time_from_offset_min: Optional[float] = None,
    time_from_offset_max: Optional[float] = None,
    page_size: int = PAGE_SIZE,
    timeout: int = TIMEOUT,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    use_cache: bool = True,
    cache_ttl: Optional[int] = None
) -> List[Dict]:
    headers = {"Authorization": f"Bearer {token}"}
    url = normalize_url(api_url, "timeseries")
    params = {}
    if patient_id is not None:
        params['patient_id'] = patient_id
    if study_ids is not None:
        for sid in study_ids:
            params.setdefault('study_id[]', []).append(sid)
    if event_id is not None:
        params['event_id'] = event_id
    if channel in ('C3', 'C4'):
        params['channel'] = channel
    if time_from_offset_min is not None:
        params['time_from_offset_min'] = time_from_offset_min
    if time_from_offset_max is not None:
        params['time_from_offset_max'] = time_from_offset_max

    if use_cache:
        cached = get_cached_data(api_url, "timeseries", params, ttl=cache_ttl)
        if cached is not None:
            logger.info(f"Загружено {len(cached)} временных точек из кэша")
            return cached

    data = paginated_get(url, headers, params, page_size, timeout, stop_check, progress_callback)

    if use_cache and data:
        save_cached_data(api_url, "timeseries", params, data)
    return data