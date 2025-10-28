import logging
import os
import torch
import whisper
import pandas as pd
from difflib import SequenceMatcher
import json
import time
import threading
from queue import Queue
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uuid
import shutil
from typing import List
from contextlib import asynccontextmanager
import datetime

# Получаем абсолютный путь к папке app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    global config, whisper_model, processor

    logger.info("Запуск Dialog Analyzer API")

    # Загрузка конфигурации
    config = load_config()

    # Создание папок с абсолютными путями
    os.makedirs(config["audio_folder"], exist_ok=True)
    os.makedirs(config["report_folder"], exist_ok=True)
    os.makedirs(config["reference_dialogs_folder"], exist_ok=True)

    # Загрузка модели Whisper
    device = "cuda" if torch.cuda.is_available() and config["use_cuda"] else "cpu"
    logger.info(f"Используем устройство: {device}")

    try:
        whisper_model = whisper.load_model(config["whisper_model"])
        logger.info("Whisper загружен успешно!")
    except Exception as e:
        logger.error(f"Ошибка загрузки Whisper: {e}")
        raise

    # Загрузка эталонных диалогов
    reference_dialogs = load_reference_dialogs()

    # Загрузка существующих отчетов
    existing_reports = load_all_reports_metadata()
    logger.info(f"Загружено отчетов из предыдущих сессий: {len(existing_reports)}")

    # Инициализация процессора
    processor = DialogProcessor(reference_dialogs)

    # Восстанавливаем результаты из метаданных
    for report_meta in existing_reports:
        processor.results.append({
            "task_id": report_meta["task_id"],
            "filename": report_meta["filename"],
            "dialog_name": report_meta["dialog_name"],
            "similarity": report_meta["similarity"],
            "error_count": report_meta["error_count"],
            "language": report_meta["language"],
            "status": "Есть ошибки" if report_meta["error_count"] > 0 else "OK"
        })

    logger.info("API готов к работе")

    yield

    logger.info("Выключение Dialog Analyzer API")


app = FastAPI(title="Dialog Analyzer API", version="1.0.0", lifespan=lifespan)

# Глобальные переменные
whisper_model = None
processor = None
config = None


class DialogProcessor:
    def __init__(self, reference_dialogs):
        self.processing_queue = Queue()
        self.is_processing = False
        self.results = []
        self.lock = threading.Lock()
        self.reference_dialogs = reference_dialogs
        self.current_tasks = {}

    def add_to_queue(self, file_path, task_id):
        self.processing_queue.put((file_path, task_id))
        self.current_tasks[task_id] = {
            "status": "queued",
            "filename": os.path.basename(file_path),
            "progress": 0
        }
        logger.info(f"Файл добавлен в очередь: {os.path.basename(file_path)}")

        if not self.is_processing:
            self.start_processing()

    def start_processing(self):
        self.is_processing = True
        processing_thread = threading.Thread(target=self._process_queue)
        processing_thread.daemon = True
        processing_thread.start()

    def _process_queue(self):
        while not self.processing_queue.empty():
            file_path, task_id = self.processing_queue.get()

            try:
                self.current_tasks[task_id]["status"] = "processing"
                self.current_tasks[task_id]["progress"] = 25
                logger.info(f"Начинаю обработку: {os.path.basename(file_path)}")

                # Транскрибация
                transcribed_text, language = transcribe_audio(file_path)
                self.current_tasks[task_id]["progress"] = 50

                # Поиск подходящего эталона
                reference_text, dialog_name, similarity_score = find_best_reference_match(transcribed_text,
                                                                                          self.reference_dialogs)
                self.current_tasks[task_id]["progress"] = 75

                # Анализ ошибок агента
                errors, recommendations, script_similarity = analyze_agent_errors(transcribed_text, reference_text)

                # Сохранение индивидуального отчета
                self._save_individual_report(file_path, task_id, transcribed_text, dialog_name,
                                             reference_text, errors, recommendations, language, script_similarity)

                # Сохранение результата для сводки
                result = {
                    "task_id": task_id,
                    "filename": os.path.basename(file_path),
                    "dialog_name": dialog_name,
                    "similarity": f"{script_similarity:.1%}" if reference_text else "N/A",
                    "error_count": len(errors),
                    "main_errors": "; ".join(errors[:2]) if errors else "Нет ошибок",
                    "recommendations": "; ".join(recommendations[:2]) if recommendations else "Все хорошо",
                    "language": language,
                    "status": "Есть ошибки" if errors else "OK",
                    "transcribed_text": transcribed_text,
                    "errors": errors,
                    "recommendations_list": recommendations
                }

                with self.lock:
                    self.results.append(result)

                self.current_tasks[task_id]["status"] = "completed"
                self.current_tasks[task_id]["progress"] = 100
                self.current_tasks[task_id]["result"] = result

                logger.info(f"Обработка завершена: {os.path.basename(file_path)}")

            except Exception as e:
                logger.error(f" Ошибка обработки {file_path}: {e}")
                self.current_tasks[task_id]["status"] = "error"
                self.current_tasks[task_id]["error"] = str(e)

            finally:
                self.processing_queue.task_done()
                time.sleep(1)

        self.is_processing = False

    def _save_individual_report(self, file_path, task_id, transcribed_text, dialog_name,
                                reference_text, errors, recommendations, language, similarity):
        """Сохранение индивидуального отчета в формате TXT"""
        try:
            filename = os.path.basename(file_path)

            # Формируем содержание отчета
            report_lines = [
                "АНАЛИЗ ДИАЛОГА АГЕНТА",
                f"Файл: {filename}",
                f"Дата анализа: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Язык: {language}",
                "",
                "ТРАНСКРИБИРОВАННЫЙ ТЕКСТ:",
                transcribed_text,
                "",
                "ОШИБКИ:"
            ]

            # Добавляем ошибки
            if errors:
                for error in errors:
                    report_lines.append(f"• {error}")
            else:
                report_lines.append("• Нет ошибок")

            report_lines.extend([
                "",
                "РЕКОМЕНДАЦИИ:"
            ])

            # Добавляем рекомендации
            if recommendations:
                for rec in recommendations:
                    report_lines.append(f"• {rec}")
            else:
                report_lines.append("• Все отлично!")

            report_lines.extend([
                "",
                f"ЭТАЛОННЫЙ СКРИПТ ({dialog_name}):",
                reference_text if reference_text else "Не найден"
            ])

            # Объединяем все строки
            report_content = "\n".join(report_lines)

            report_filename = f"report_{os.path.splitext(filename)[0]}_{task_id[:8]}.txt"
            report_path = os.path.join(config["report_folder"], report_filename)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            # Сохраняем метаданные
            save_report_metadata(
                task_id=task_id,
                filename=filename,
                dialog_name=dialog_name,
                similarity=f"{similarity:.1%}" if reference_text else "N/A",
                error_count=len(errors),
                language=language,
                report_filename=report_filename
            )

            logger.info(f"Сохранен отчет: {report_filename}")

        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")

    def get_task_status(self, task_id):
        return self.current_tasks.get(task_id, {"status": "not_found"})

    def get_all_results(self):
        with self.lock:
            return self.results.copy()


def save_report_metadata(task_id, filename, dialog_name, similarity, error_count, language, report_filename):
    """Сохраняет метаданные отчета в JSON файл"""
    try:
        metadata = {
            "task_id": task_id,
            "filename": filename,
            "dialog_name": dialog_name,
            "similarity": similarity,
            "error_count": error_count,
            "language": language,
            "created_at": datetime.datetime.now().isoformat(),
            "report_file": report_filename
        }

        metadata_file = f"metadata_{task_id}.json"
        metadata_path = os.path.join(config["report_folder"], metadata_file)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Ошибка сохранения метаданных: {e}")


def load_all_reports_metadata():
    """Загружает метаданные всех отчетов при запуске"""
    reports = []
    try:
        if not os.path.exists(config["report_folder"]):
            return reports

        for file_name in os.listdir(config["report_folder"]):
            if file_name.startswith('metadata_') and file_name.endswith('.json'):
                file_path = os.path.join(config["report_folder"], file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    reports.append(metadata)
    except Exception as e:
        logger.error(f"Ошибка загрузки метаданных: {e}")

    return reports


def find_config_file():
    """Search for config.json and return the path if found"""

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)

    possible_paths = [
        os.path.join(parent_dir, "config.json"),
        os.path.join(current_dir, "config.json"),
        "config.json",
        "../config.json",
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path

    return None


def get_default_config():
    return {
        "audio_folder": os.path.join(BASE_DIR, "audio"),
        "report_folder": os.path.join(BASE_DIR, "reports"),
        "reference_dialogs_folder": os.path.join(BASE_DIR, "etalons"),
        "whisper_model": "medium",
        "supported_languages": ["en", "es", "it", "de", "pl", "ru"],
        "use_cuda": True
    }


def load_config():
    config_path = os.getenv("CONFIG_PATH")

    if not config_path:
        config_path = find_config_file()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            for env_var, config_key in [
                ("AUDIO_FOLDER", "audio_folder"),
                ("REPORT_FOLDER", "report_folder"),
                ("REFERENCE_DIALOGS_FOLDER", "reference_dialogs_folder"),
                ("WHISPER_MODEL", "whisper_model"),
                ("USE_CUDA", "use_cuda")
            ]:
                if env_var in os.environ:
                    config[config_key] = os.environ[env_var]

            logger.info("✅ Конфиг загружен успешно")
            return config

        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка в формате JSON: {e}")
            raise
    else:
        logger.warning("⚠ Используем значения по умолчанию")
        return get_default_config()


def transcribe_audio(file_path: str):
    try:
        logger.info(f"🔊 Начинаю транскрибацию {os.path.basename(file_path)}...")
        result = whisper_model.transcribe(file_path)
        logger.info(f"✅ Транскрибация завершена")
        return result["text"].strip(), result.get("language", "unknown")
    except Exception as e:
        logger.error(f"❌ Ошибка транскрибации: {e}")
        return f"Ошибка транскрибации: {e}", "unknown"


def load_reference_dialogs():
    reference_dialogs = {}
    reference_dialogs_folder = config["reference_dialogs_folder"]

    print(f"🔍 Ищем эталоны в: {reference_dialogs_folder}")
    print(f"📁 Папка существует: {os.path.exists(reference_dialogs_folder)}")

    if not os.path.exists(reference_dialogs_folder):
        logger.warning(f"⚠ Папка с эталонными диалогами не найдена: {reference_dialogs_folder}")
        return reference_dialogs

    # Детальная проверка папки
    try:
        print(f"🔎 Проверяем права доступа к папке...")
        print(f"📊 Права папки: {oct(os.stat(reference_dialogs_folder).st_mode)[-3:]}")

        files = os.listdir(reference_dialogs_folder)
        print(f"📂 Файлы в папке: {files}")
        print(f"📊 Количество файлов: {len(files)}")

        # Проверяем каждый файл отдельно
        for file_name in files:
            file_path = os.path.join(reference_dialogs_folder, file_name)
            print(f"\n🔎 Анализируем файл: {file_name}")
            print(f"📁 Полный путь: {file_path}")
            print(f"📄 Файл существует: {os.path.exists(file_path)}")
            print(f"🔐 Права файла: {oct(os.stat(file_path).st_mode)[-3:]}")
            print(f"📏 Размер файла: {os.path.getsize(file_path)} байт")
            print(f"📝 Расширение: {os.path.splitext(file_name)[1]}")

            if file_name.endswith('.txt'):
                print(f"✅ Это TXT файл, пытаемся загрузить...")
                try:
                    # Пробуем открыть файл разными способами
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        dialog_name = os.path.splitext(file_name)[0]
                        reference_dialogs[dialog_name] = content
                        print(f"🎉 УСПЕХ! Загружен эталон: {dialog_name}")
                        print(f"📊 Размер содержимого: {len(content)} символов")
                        print(f"📝 Первые 100 символов: {repr(content[:100])}")

                except UnicodeDecodeError as e:
                    print(f"❌ Ошибка кодировки: {e}")
                    # Пробуем другие кодировки
                    encodings = ['utf-8-sig', 'latin-1', 'cp1251', 'cp1252', 'iso-8859-1']
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read().strip()
                            print(f"✅ Успешно с кодировкой {encoding}")
                            dialog_name = os.path.splitext(file_name)[0]
                            reference_dialogs[dialog_name] = content
                            break
                        except UnicodeDecodeError:
                            print(f"❌ Не удалось с кодировкой {encoding}")
                            continue

                except Exception as e:
                    print(f"❌ Ошибка загрузки: {e}")
            else:
                print(f"⏭ Пропускаем (не TXT файл)")

    except PermissionError as e:
        print(f"❌ Ошибка прав доступа: {e}")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")

    print(f"\n📊 ИТОГО: Загружено эталонов: {len(reference_dialogs)}")
    print(f"📋 Названия: {list(reference_dialogs.keys())}")

    return reference_dialogs


def find_best_reference_match(transcribed_text, reference_dialogs):
    if not reference_dialogs:
        return None, "Не найден", 0

    best_match = None
    best_score = 0
    best_dialog_name = ""

    for dialog_name, reference_text in reference_dialogs.items():
        similarity = SequenceMatcher(None, transcribed_text.lower(), reference_text.lower()).ratio()

        if similarity > best_score:
            best_score = similarity
            best_match = reference_text
            best_dialog_name = dialog_name

    return best_match, best_dialog_name, best_score


def analyze_agent_errors(transcribed_text, reference_text):
    errors = []
    recommendations = []

    if not reference_text:
        return ["Эталон не найден"], ["Добавить соответствующий эталонный диалог"], 0

    transcribed_lower = transcribed_text.lower()
    reference_lower = reference_text.lower()

    reference_lines = [line.strip() for line in reference_text.split('\n') if line.strip() and len(line.strip()) > 15]
    missing_key_phrases = []

    for line in reference_lines:
        if line.lower() not in transcribed_lower:
            missing_key_phrases.append(line)

    if missing_key_phrases:
        errors.append(f"Пропущены ключевые фразы ({len(missing_key_phrases)} шт.)")
        recommendations.append("Включить обязательные фразы из скрипта")

    pressure_keywords = [
        'must', 'have to', 'now', 'immediately', 'urgent', 'last chance',
        'limited time', 'hurry', 'no choice', 'only way', 'obliged'
    ]

    pressure_count = sum(1 for word in pressure_keywords if word in transcribed_lower)
    if pressure_count > 2:
        errors.append("Излишнее давление на клиента")
        recommendations.append("Снизить агрессивность, использовать более мягкие формулировки")

    professional_errors = []
    greetings = ['hello', 'hi', 'good morning', 'good afternoon', 'welcome', 'добрый', 'здравствуйте']
    farewells = ['thank you', 'goodbye', 'have a nice day', 'bye', 'спасибо', 'до свидания']

    has_greeting = any(greet in transcribed_lower for greet in greetings)
    has_farewell = any(farewell in transcribed_lower for farewell in farewells)

    if not has_greeting:
        professional_errors.append("Отсутствует приветствие")
    if not has_farewell:
        professional_errors.append("Отсутствует прощание")

    if professional_errors:
        errors.extend(professional_errors)
        recommendations.append("Соблюдать стандарты профессионального общения")

    similarity = SequenceMatcher(None, transcribed_lower, reference_lower).ratio()
    if similarity < 0.3:
        errors.append("Низкое соответствие эталонному скрипту")
        recommendations.append("Требуется дополнительное обучение по скрипту")

    return errors, recommendations, similarity


# API endpoints
@app.get("/")
async def root():
    return HTMLResponse(content=UI_HTML)


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        task_id = str(uuid.uuid4())
        file_path = os.path.join(config["audio_folder"], f"{task_id}_{file.filename}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        processor.add_to_queue(file_path, task_id)

        return {
            "task_id": task_id,
            "filename": file.filename,
            "status": "queued",
            "message": "Файл добавлен в очередь обработки"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка загрузки файла: {str(e)}"}
        )


@app.post("/analyze-batch")
async def analyze_audio_batch(files: List[UploadFile] = File(...)):
    task_ids = []

    for file in files:
        try:
            task_id = str(uuid.uuid4())
            file_path = os.path.join(config["audio_folder"], f"{task_id}_{file.filename}")

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            processor.add_to_queue(file_path, task_id)
            task_ids.append(task_id)

        except Exception as e:
            logger.error(f"❌ Ошибка обработки файла {file.filename}: {e}")

    return {
        "task_ids": task_ids,
        "message": f"{len(task_ids)} файлов добавлено в очередь"
    }


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    status = processor.get_task_status(task_id)

    if status.get("status") == "not_found":
        return JSONResponse(
            status_code=404,
            content={"error": "Задача не найдена"}
        )

    return status


@app.get("/results")
async def get_all_results():
    results = processor.get_all_results()
    return {"results": results}


@app.get("/download-report/{task_id}")
async def download_individual_report(task_id: str):
    """Скачать индивидуальный отчет по task_id"""
    try:
        # Ищем метаданные по task_id
        metadata_file = os.path.join(config["report_folder"], f"metadata_{task_id}.json")

        if not os.path.exists(metadata_file):
            return JSONResponse(
                status_code=404,
                content={"error": "Метаданные отчета не найдены"}
            )

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        report_path = os.path.join(config["report_folder"], metadata["report_file"])

        if not os.path.exists(report_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Файл отчета не найден"}
            )

        filename = f"report_{metadata['filename']}_{task_id[:8]}.txt"

        return FileResponse(
            report_path,
            filename=filename,
            media_type='text/plain'
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка загрузки отчета: {str(e)}"}
        )


@app.get("/list-reports")
async def list_reports():
    """Получить список всех доступных отчетов (из метаданных)"""
    try:
        # Загружаем метаданные вместо сканирования папки
        reports_metadata = load_all_reports_metadata()

        reports = []
        for metadata in reports_metadata:
            report_file_path = os.path.join(config["report_folder"], metadata["report_file"])
            if os.path.exists(report_file_path):
                file_stat = os.stat(report_file_path)
                reports.append({
                    "task_id": metadata["task_id"],
                    "filename": metadata["filename"],
                    "report_filename": metadata["report_file"],
                    "dialog_name": metadata["dialog_name"],
                    "similarity": metadata["similarity"],
                    "error_count": metadata["error_count"],
                    "language": metadata["language"],
                    "size": file_stat.st_size,
                    "created": metadata["created_at"]
                })

        # Сортируем по дате создания (новые сначала)
        reports.sort(key=lambda x: x["created"], reverse=True)

        return {"reports": reports}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка получения списка отчетов: {str(e)}"}
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": whisper_model is not None,
        "queue_size": processor.processing_queue.qsize() if processor else 0
    }


# Debug endpoints
@app.get("/debug/filesystem")
async def debug_filesystem():
    """Проверка файловой системы"""
    import subprocess

    # Проверяем различные пути
    paths_to_check = [
        "/app",
        "/app/etalons",
        "/Etalon",
        "/",
        os.getcwd()
    ]

    results = {}
    for path in paths_to_check:
        try:
            if os.path.exists(path):
                # Получаем список файлов
                files = os.listdir(path)
                results[path] = {
                    "exists": True,
                    "files": files,
                    "is_dir": os.path.isdir(path)
                }

                # Проверяем конкретный файл
                test_file = os.path.join(path, "good_scripts.txt")
                results[path]["good_scripts_exists"] = os.path.exists(test_file)
            else:
                results[path] = {"exists": False}
        except Exception as e:
            results[path] = {"error": str(e)}

    return results


@app.get("/debug/check-file")
async def debug_check_file():
    """Проверка конкретного файла good_scripts.txt"""
    file_path = "/app/etalons/good_scripts.txt"

    result = {
        "file_path": file_path,
        "exists": os.path.exists(file_path),
        "error": None,
        "content_preview": None,
        "size": 0,
        "permissions": None
    }

    if result["exists"]:
        try:
            # Получаем информацию о файле
            stat = os.stat(file_path)
            result["size"] = stat.st_size
            result["permissions"] = oct(stat.st_mode)[-3:]

            # Пробуем прочитать файл
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251', 'cp1252']

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read().strip()
                    result["encoding"] = encoding
                    result["content_preview"] = content[:200] + "..." if len(content) > 200 else content
                    result["content_length"] = len(content)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                result["error"] = "Не удалось прочитать ни в одной кодировке"

        except Exception as e:
            result["error"] = str(e)

    return result


# Обновленный HTML интерфейс
UI_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Анализатор диалогов</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .upload-section {
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
            border-radius: 5px;
        }
        .task {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .completed { border-color: #28a745; background: #f8fff9; }
        .error { border-color: #dc3545; background: #fff5f5; }
        .processing { border-color: #ffc107; background: #fffef0; }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover { background: #0056b3; }
        input[type="file"] { margin: 10px 0; padding: 10px; }
        .progress { 
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: #007bff;
            transition: width 0.3s;
        }
        .reports-section {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .report-item {
            border: 1px solid #ddd;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            background: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Анализатор диалогов агентов</h1>
        <p>Загрузите аудиофайлы для анализа соответствия эталонным скриптам</p>

        <div class="upload-section">
            <h3>📁 Загрузка аудиофайлов</h3>
            <input type="file" id="fileInput" multiple accept=".mp3,.wav,.m4a,.ogg,audio/*">
            <br>
            <button onclick="uploadFiles()">🚀 Начать анализ</button>
            <div id="uploadStatus"></div>
        </div>

        <div class="results-section">
            <h3>📊 Результаты анализа</h3>
            <button onclick="loadResults()">🔄 Обновить результаты</button>
            <div id="results"></div>
        </div>

        <div class="reports-section">
            <h3>📄 Отчеты</h3>
            <button onclick="listReports()">🔄 Показать список отчетов</button>
            <div id="reportsList"></div>
        </div>
    </div>

    <script>
        async function uploadFiles() {
            const fileInput = document.getElementById('fileInput');
            const statusDiv = document.getElementById('uploadStatus');

            if (!fileInput.files.length) {
                statusDiv.innerHTML = '<p style="color: red;">⚠️ Выберите файлы для анализа</p>';
                return;
            }

            const formData = new FormData();
            for (let file of fileInput.files) {
                formData.append('files', file);
            }

            try {
                statusDiv.innerHTML = '<p>📤 Загрузка файлов...</p>';
                const response = await fetch('/analyze-batch', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                if (response.ok) {
                    statusDiv.innerHTML = `<p style="color: green;">✅ ${result.message}</p>`;
                    monitorTasks(result.task_ids);
                } else {
                    statusDiv.innerHTML = `<p style="color: red;">❌ Ошибка: ${result.error || result.message}</p>`;
                }

            } catch (error) {
                statusDiv.innerHTML = `<p style="color: red;">❌ Ошибка соединения: ${error}</p>`;
            }
        }

        async function monitorTasks(taskIds) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<h4>📈 Статус задач:</h4>';

            for (let taskId of taskIds) {
                const taskDiv = document.createElement('div');
                taskDiv.id = `task-${taskId}`;
                taskDiv.className = 'task';
                resultsDiv.appendChild(taskDiv);

                updateTaskStatus(taskId);

                const interval = setInterval(() => {
                    updateTaskStatus(taskId).then(shouldContinue => {
                        if (!shouldContinue) {
                            clearInterval(interval);
                        }
                    });
                }, 2000);
            }
        }

        async function updateTaskStatus(taskId) {
            try {
                const response = await fetch(`/status/${taskId}`);
                const status = await response.json();

                const taskElement = document.getElementById(`task-${taskId}`);
                if (!taskElement) return false;

                let content = `<strong>${status.filename}</strong><br>`;

                if (status.status === 'completed') {
                    content += `✅ Завершено | Ошибок: ${status.result?.error_count} | Соответствие: ${status.result?.similarity}`;
                    content += ` <button onclick="downloadReport('${taskId}')">📄 Скачать отчет</button>`;
                    taskElement.className = 'task completed';
                } else if (status.status === 'error') {
                    content += `❌ Ошибка: ${status.error}`;
                    taskElement.className = 'task error';
                } else {
                    content += `⏳ ${status.status}...`;
                    taskElement.className = 'task processing';

                    if (status.progress) {
                        content += `<div class="progress"><div class="progress-bar" style="width: ${status.progress}%"></div></div>`;
                    }
                }

                taskElement.innerHTML = content;
                return status.status === 'queued' || status.status === 'processing';

            } catch (error) {
                console.error('Ошибка проверки статуса:', error);
                return false;
            }
        }

        async function downloadReport(taskId) {
            try {
                const response = await fetch(`/download-report/${taskId}`);
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `report_${taskId}.txt`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                } else {
                    alert('❌ Отчет недоступен');
                }
            } catch (error) {
                alert('❌ Ошибка скачивания: ' + error);
            }
        }

        async function listReports() {
            try {
                const response = await fetch('/list-reports');
                const data = await response.json();

                const reportsDiv = document.getElementById('reportsList');
                if (data.reports && data.reports.length > 0) {
                    let html = '<h4>Доступные отчеты:</h4>';
                    data.reports.forEach(report => {
                        html += `
                        <div class="report-item">
                            <strong>${report.filename}</strong><br>
                            Эталон: ${report.dialog_name} | 
                            Соответствие: ${report.similarity} | 
                            Ошибок: ${report.error_count}<br>
                            Язык: ${report.language} | 
                            Дата: ${new Date(report.created).toLocaleString()}<br>
                            <button onclick="downloadReport('${report.task_id}')" style="margin-top: 5px;">
                                📥 Скачать отчет
                            </button>
                        </div>`;
                    });
                    reportsDiv.innerHTML = html;
                } else {
                    reportsDiv.innerHTML = '<p>📭 Отчетов пока нет</p>';
                }
            } catch (error) {
                console.error('Ошибка загрузки списка отчетов:', error);
                reportsDiv.innerHTML = '<p style="color: red;">❌ Ошибка загрузки списка отчетов</p>';
            }
        }

        async function loadResults() {
            try {
                const response = await fetch('/results');
                const data = await response.json();

                const resultsDiv = document.getElementById('results');
                if (data.results && data.results.length > 0) {
                    resultsDiv.innerHTML = '<h4>📋 Все результаты:</h4>';

                    data.results.forEach(result => {
                        const resultDiv = document.createElement('div');
                        resultDiv.className = `task ${result.status === 'OK' ? 'completed' : 'error'}`;
                        resultDiv.innerHTML = `
                            <strong>${result.filename}</strong><br>
                            Эталон: ${result.dialog_name}<br>
                            Соответствие: ${result.similarity}<br>
                            Ошибки: ${result.error_count}<br>
                            Статус: ${result.status}
                            <button onclick="downloadReport('${result.task_id}')" style="margin-top: 5px;">
                                📥 Скачать отчет
                            </button>
                        `;
                        resultsDiv.appendChild(resultDiv);
                    });
                } else {
                    resultsDiv.innerHTML = '<p>📭 Результатов пока нет</p>';
                }
            } catch (error) {
                console.error('Ошибка загрузки результатов:', error);
            }
        }

        // Загружаем результаты при открытии страницы
        document.addEventListener('DOMContentLoaded', loadResults);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
