# AI-agents_local_realization
ИИ-агенты интернет магазина электроники, задания с Agents Week Школы анализа данных (YSDA)

# AI Shopping Agents — ReAct, Memory & Multi-Agent System

Коллекция из трёх прогрессивных агентов для электронного магазина, реализованных на **LangChain + Ollama**. Каждый следующий агент расширяет возможности предыдущего: от простого ReAct-цикла до координируемой мультиагентной системы.

---


## 🔧 Требования

### Системные
- Python 3.10+
- [Ollama](https://ollama.com/) (локальный LLM-сервер)

### Языковая модель
```bash
# После установки Ollama скачайте модель (пример):
ollama pull llama3.2
# или любую другую, совместимую с OpenAI API:
# ollama pull mistral, qwen2.5, gemma2 и т.д.
```

>  В коде используется `MODEL_NAME = "gpt-oss:20b"` — замените на имя вашей модели.

### Python-зависимости
```bash
pip install langchain-openai langchain-core requests
```

---

##  Установка

```bash
# 1. Клонируйте репозиторий
git clone <your-repo-url>
cd <project-folder>

# 2. Запустите Ollama (в отдельном терминале)
ollama serve

# 3. Проверьте доступность модели
ollama list
```

---

## Агент 1: ReAct Tool-Calling

**Простой агент с циклом Reason → Act → Observe.**

### Возможности:
- Поиск товаров по запросу, категории, бренду, цене
- Добавление товаров в корзину
- Автоматическое планирование действий через tool calling

### Пример:
```python
state = ShopState()
tracer = ToolTracer()
response = run_shopping_agent(
    "Find wireless headphones under $150 and add the cheapest",
    state, TOOLS, tracer
)
print(response)
```

### Архитектура:
```
[User Query] 
    → [LLM + Tools Schema] 
    → [Tool Call?] → Execute → [ToolMessage] → Repeat
    → [Final Answer]
```

---

## Агент 2: Memory Agent

**ReAct-агент с краткосрочной и долгосрочной памятью.**

### Типы памяти:
| Тип | Хранилище | Назначение |
|-----|-----------|-----------|
|  Short-term | `history: list[Message]` | Контекст текущего диалога |
|  Long-term | `user_profile.json` | Предпочтения пользователя между сессиями |

### Новые возможности:
-  Сохранение предпочтений: имя, бренд, бюджет, цвет, категория
-  Продолжение диалога: агент "помнит" результаты предыдущих запросов
-  Инструмент `update_profile()` для динамического обновления профиля

### Пример:
```python
# Сессия 1: сохраняем профиль
history = []
run_memory_agent("I prefer Sony, budget $200", ..., history=history)

# Сессия 2: агент использует сохранённые предпочтения
history = []  # новая сессия, но профиль загружен из файла
run_memory_agent("What headphones do you recommend?", ..., history=history)
```

---

## Агент 3: Multi-Agent System

**Координируемая цепочка из 4 специализированных агентов + оркестратор.**

### Агенты:
| Агент | Роль | Инструменты | Вывод |
|-------|------|------------|-------|
|  `RetrieverAgent` | Поиск товаров | `search_products` | `ctx.candidates` |
|  `ProsAgent` | Анализ преимуществ | нет (только LLM) | `ctx.pros` |
|  `ConsAgent` | Анализ недостатков | нет (только LLM) | `ctx.cons` |
|  `RankerAgent` | Выбор лучшего | нет (логика) | `ctx.best` |
|  `CoordinatorAgent` | Оркестрация | все + `add_to_cart` | финальный ответ |

### Общий контекст:
```python
@dataclass
class AgentContext:
    query: str
    max_price: float | None
    candidates: list[dict]
    pros: dict[str, str]      # id → описание плюсов
    cons: dict[str, str]      # id → описание минусов
    best: dict | None         # выбранный товар
    cart_result: dict | None
```

### Пример:
```python
result = CoordinatorAgent().run(
    "Find the best wireless mouse under $120 and add to cart",
    state, TOOLS
)
print(result.response)
# → Best choice: Logitech MX Master 3S ($109, 4.8★)
#   Pros: ...
#   Cons: ...
```

### Архитектура:
```
Coordinator
   │
   ├──▶ Retriever ──▶ [candidates]
   │
   ├──▶ ProsAgent ──▶ [pros]
   │
   ├──▶ ConsAgent ──▶ [cons]
   │
   ├──▶ Ranker ─────▶ [best]
   │
   └──▶ [response + optional cart]
```

---

## Запуск тестов

В конце notebook'а предусмотрены автотесты для каждой задачи:

```python
# Task 1
assert _t1a.called("search_products")
print("OK 1.A")

# Task 2
assert _prof2a.get("brand") == "Sony"
print("OK 2.A")

# Task 3
assert _res3a.context.best["id"] == "p6"
print("OK 3.A")
```

Просто выполните ячейки последовательно — тесты проверят корректность реализации.

---

## Настройка модели

Откройте ячейку с конфигурацией и укажите свои параметры:

```python
# Для Ollama (локально):
OPENAI_API_KEY = "ollama"
OPENAI_API_BASE = "http://localhost:11434/v1"
MODEL_NAME = "llama3.2"  # ← ваша модель

# Для YandexGPT / других совместимых с OpenAI API:
# OPENAI_API_BASE = "https://llm.api.aistudio.yandex.ru/v1"
# OPENAI_API_KEY = "ваш_ключ"
```

---

## Технологии

- **[LangChain](https://python.langchain.com/)** — фреймворк для работы с LLM
- **[Ollama](https://ollama.com/)** — локальный запуск открытых моделей
- **ReAct Pattern** — Reason + Act цикл для агентного ИИ
- **Function Calling** — структурированный вызов инструментов через LLM

---


