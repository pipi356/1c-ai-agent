import os
import json
import httpx
from io import BytesIO
from typing import Dict, List, Any
from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from docx import Document

from schemas import (
    ChatRequest, ChatResponse, ToolResultRequest,
    LLMDecision, ClientAction
)
from rag import init_qdrant_collection, search_knowledge, index_text_chunks

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")

app = FastAPI(title="1C Enterprise Local AI Orchestrator", version="1.0.0")

# Хранилище контекста активных сессий 1С в памяти
SESSIONS: Dict[str, Dict[str, Any]] = {}

SYSTEM_PROMPT = """Ты — универсальный встроенный ИИ-ассистент системы 1С:Предприятие.
Ты работаешь внутри клиент-серверного расширения 1С и помогаешь управлять ЛЮБЫМИ объектами конфигурации (справочники, документы, отчеты, обработки).

УНИВЕРСАЛЬНЫЕ ПРАВИЛА РАБОТЫ С ИНСТРУМЕНТАМИ:
1. ОТКРЫТИЕ ФОРМ И ОТЧЕТОВ ("open_form"):
   - Если пользователь просит открыть, показать или посмотреть любой объект (например, "открой склад", "покажи продажи", "отчет по остаткам"), ты ДОЛЖЕН вызвать инструмент "open_form".
   - Сам определи правильное полное имя метаданного в формате платформы 1С:
     * Справочники: "Справочник.<Имя>" (например, Справочник.Склады, Справочник.Контрагенты)
     * Документы: "Документ.<Имя>" (например, Документ.РеализацияТоваровУслуг)
     * Отчеты: "Отчет.<Имя>" (например, Отчет.ОстаткиТоваров)
   - Аргумент: `{"metadata_name": "Справочник.Склады"}`. Запрещено отвечать текстом!

2. СОЗДАНИЕ ОБЪЕКТОВ ("create_object"):
   - Если пользователь просит создать объект любого типа, вызывай "create_object", передавая `metadata_type` (например, "Справочник.Склады" или "Документ.ЗаказКлиента") и собранные реквизиты в `attributes`.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ 1С (tool_name):
  1) "find_entity": arguments = {"metadata_type": "Справочник.Контрагенты", "search_query": "строка"}
  2) "run_1c_report": arguments = {"report_name": "имя_отчета", "filters": {...}}
  3) "read_1c_file": arguments = {"owner_uid": "uid_объекта", "file_name": "имя"}
  4) "create_object": arguments = {"metadata_type": "Справочник.Имя", "attributes": {...}}
  5) "get_system_info": arguments = {}
  6) "open_form": arguments = {"metadata_name": "Справочник.Имя"}
"""

@app.on_event("startup")
def on_startup():
    try:
        init_qdrant_collection()
    except Exception as e:
        print(f"[Startup] Не удалось подключиться к Qdrant при старте: {e}")


async def call_ollama_structured(messages: List[Dict[str, str]]) -> LLMDecision:
    """Вызывает Ollama с жесткой фиксацией JSON-схемы ответа (Structured Outputs)."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "format": LLMDecision.model_json_schema(),
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192
        }
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return LLMDecision.model_validate_json(content)


def build_decision_response(session_id: str, decision: LLMDecision, sources: List[str]) -> ChatResponse:
    if decision.decision == "call_1c_tools" and decision.tool_calls:
        return ChatResponse(
            session_id=session_id,
            status="tool_call",
            tool_calls=decision.tool_calls,
            reply_text=decision.reply_text,
            sources=sources
        )
    elif decision.decision == "fill_current_form" and decision.fill_payload:
        return ChatResponse(
            session_id=session_id,
            status="completed",
            reply_text=decision.reply_text,
            client_action=ClientAction(
                action_type="fill_current_form",
                payload=decision.fill_payload
            ),
            sources=sources
        )
    else:
        return ChatResponse(
            session_id=session_id,
            status="completed",
            reply_text=decision.reply_text,
            sources=sources
        )


@app.post("/api/v1/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    # 1. Ищем релевантные инструкции в Qdrant
    search_query = req.message
    if req.form_context and req.form_context.metadata_name:
        search_query = f"{req.form_context.metadata_name} {req.form_context.synonym or ''}: {req.message}"

    rag_context, sources = await search_knowledge(search_query)

    # 2. Собираем контекст открытой формы 1С
    form_info = "Форма не открыта или контекст не передан."
    if req.form_context and req.form_context.is_open:
        form_info = json.dumps(req.form_context.model_dump(), ensure_ascii=False, indent=2)

    user_prompt = (
        f"[ПОЛЬЗОВАТЕЛЬ 1С]: {req.user.name} (Роли: {', '.join(req.user.roles)})\n\n"
        f"[КОНТЕКСТ ОТКРЫТОЙ ФОРМЫ 1С]:\n{form_info}\n\n"
        f"[БАЗА ЗНАНИЙ (ИНСТРУКЦИИ)]:\n{rag_context or 'Релевантных инструкций не найдено.'}\n\n"
        f"[ЗАПРОС ПОЛЬЗОВАТЕЛЯ]:\n{req.message}"
    )

    session = SESSIONS.setdefault(req.session_id, {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "sources": []
    })
    session["sources"] = sources
    session["messages"].append({"role": "user", "content": user_prompt})

    decision = await call_ollama_structured(session["messages"])
    session["messages"].append({"role": "assistant", "content": decision.model_dump_json()})

    return build_decision_response(req.session_id, decision, sources)


@app.post("/api/v1/chat/tool-result", response_model=ChatResponse)
async def tool_result(req: ToolResultRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        return ChatResponse(
            session_id=req.session_id,
            status="completed",
            reply_text="Сессия не найдена. Повторите запрос."
        )

    results_json = json.dumps([r.model_dump() for r in req.results], ensure_ascii=False, indent=2)
    tool_feedback = (
        f"[РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ ИНСТРУМЕНТОВ НА СЕРВЕРЕ 1С]:\n{results_json}\n\n"
        "Проанализируй полученные из 1С данные (включая найденные UID или данные отчета) и сформируй следующий шаг "
        "(например, fill_current_form, create_object или финальный ответ answer)."
    )
    session["messages"].append({"role": "user", "content": tool_feedback})

    decision = await call_ollama_structured(session["messages"])
    session["messages"].append({"role": "assistant", "content": decision.model_dump_json()})

    # Если был вызван create_object и 1С вернула созданный UID, открываем форму на клиенте
    for r in req.results:
        if r.tool_name == "create_object" and r.success and isinstance(r.data, dict) and r.data.get("uid"):
            return ChatResponse(
                session_id=req.session_id,
                status="completed",
                reply_text=decision.reply_text,
                client_action=ClientAction(
                    action_type="open_created_object",
                    payload=r.data
                ),
                sources=session.get("sources", [])
            )
        # Если был вызван open_form и 1С подтвердила успех, отдаем клиенту команду открыть форму
        if r.tool_name == "open_form" and r.success:
            payload_data = r.data if isinstance(r.data, dict) else {"metadata_name": r.arguments.get("metadata_name")}
            return ChatResponse(
                session_id=req.session_id,
                status="completed",
                reply_text=decision.reply_text,
                client_action=ClientAction(
                    action_type="open_form",
                    payload=payload_data
                ),
                sources=session.get("sources", [])
            )
    return build_decision_response(req.session_id, decision, session.get("sources", []))


@app.post("/api/v1/knowledge/upload")
async def upload_knowledge_file(file: UploadFile = File(...)):
    """Загрузка локальных инструкций (.txt, .md, .docx, .pdf) в базу знаний 1С."""
    raw = await file.read()
    filename = file.filename or "document"
    ext = filename.lower().split(".")[-1]

    text = ""
    if ext in ("txt", "md"):
        text = raw.decode("utf-8", errors="ignore")
    elif ext == "pdf":
        reader = PdfReader(BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        doc = Document(BytesIO(raw))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        return {"status": "error", "message": f"Формат .{ext} пока не поддерживается"}

    # Разбиваем на смысловые чанки по ~1000 символов с перекрытием 150 символов
    chunk_size, overlap = 1000, 150
    chunks = [
        text[i: i + chunk_size]
        for i in range(0, len(text), chunk_size - overlap)
        if text[i: i + chunk_size].strip()
    ]
    count = await index_text_chunks(filename, chunks)
    return {"status": "ok", "file": filename, "chunks_indexed": count}
