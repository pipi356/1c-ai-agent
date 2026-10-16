from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


# --- Модели входного запроса из 1С ---

class UserContext(BaseModel):
    name: str
    roles: List[str] = []


class FormAttribute(BaseModel):
    name: str
    title: Optional[str] = None
    type: str
    required: bool = False
    current_value: Optional[Any] = None


class TabularColumn(BaseModel):
    name: str
    title: Optional[str] = None
    type: str
    required: bool = False


class TabularSection(BaseModel):
    name: str
    columns: List[TabularColumn] = []
    row_count: int = 0
    sample_rows: List[Dict[str, Any]] = []


class FormContext(BaseModel):
    is_open: bool = False
    metadata_name: Optional[str] = None       # Например: "Документ.ПоступлениеТоваровУслуг"
    synonym: Optional[str] = None             # Например: "Поступление (акты, накладные, УПД)"
    object_ref_uid: Optional[str] = None
    is_new: bool = False
    attributes: List[FormAttribute] = []
    tabular_sections: List[TabularSection] = []
    form_errors: List[str] = []


class ChatRequest(BaseModel):
    session_id: str
    user: UserContext
    message: str
    form_context: Optional[FormContext] = None


# --- Модели вызова инструментов 1С и ответа ---

class ToolCallItem(BaseModel):
    call_id: str
    tool_name: Literal[
        "find_entity",
        "run_1c_report",
        "read_1c_file",
        "create_object",
        "open_form",
        "get_system_info",
        "run_1c_query"
    ]
    arguments: Dict[str, Any]


class ClientAction(BaseModel):
    action_type: Literal["fill_current_form", "open_created_object", "open_form", "none"]
    payload: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    session_id: str
    status: Literal["tool_call", "completed"]
    tool_calls: List[ToolCallItem] = []
    reply_text: Optional[str] = None
    client_action: Optional[ClientAction] = None
    sources: List[str] = []


# --- Модели возврата результата из 1С ---

class ToolExecutionResult(BaseModel):
    call_id: str
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None


class ToolResultRequest(BaseModel):
    session_id: str
    results: List[ToolExecutionResult]


# --- Жесткая схема ответа для LLM (Structured Output в Ollama) ---

class LLMDecision(BaseModel):
    decision: Literal["answer", "call_1c_tools", "fill_current_form"] = Field(
        description="Выбери 'call_1c_tools', если нужно найти УИД в 1С, запустить отчет, получить системную информацию (get_system_info) или создать объект. "
                    "Выбери 'fill_current_form', если все УИД уже известны и нужно заполнить открытую форму. "
                    "Выбери 'answer', если достаточно текстового ответа или подсказки."
    )
    reply_text: str = Field(
        description="Ответ пользователю на русском языке (понятный, с указанием шагов или реквизитов)."
    )
    tool_calls: List[ToolCallItem] = Field(
        default=[],
        description="Список вызовов к 1С, если decision == 'call_1c_tools'."
    )
    fill_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Данные для заполнения реквизитов (attributes) и табличных частей (tabular_sections) открытой формы."
    )
