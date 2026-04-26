# ╔══════════════════════════════════════════════════════════════╗
# ║          TEMPLATE — DO NOT MODIFY THIS CELL                  ║
# ╚══════════════════════════════════════════════════════════════╝
# %pip install -q langchain-openai langchain-core

OPENAI_API_KEY="ollama"
OPENAI_API_BASE="http://localhost:11434/v1"

import os, json, copy
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

MODEL_NAME = "gpt-oss:20b"#"gpt-oss-20b"
#os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "YOUR_KEY_HERE")
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
llm = ChatOpenAI(model=MODEL_NAME, temperature=0, request_timeout=300)
#print(llm)

def test_ollama_connection():
    import requests
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    # 1. Проверка, что сервер Ollama запущен
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"📦 Доступные модели: {[m['name'] for m in models]}")
        else:
            print("⚠️ Сервер ответил, но с ошибкой:", resp.status_code)
    except requests.ConnectionError:
        print("❌ Не удалось подключиться к Ollama. Запустите: ollama serve")
        return False
    
    # 2. Проверка через LangChain
    try:
        llm = ChatOpenAI(
            model="gpt-oss:20b",#"llama3.2",  # ← замените на вашу модель
            temperature=0,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            request_timeout=300 
        )
        response = llm.invoke([HumanMessage(content="Назови столицу России.")])
        print(f"✅ Ответ: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при вызове модели: {e}")
        return False

def llm_chat(messages: list, tools: list | None = None) -> AIMessage:
    """
    Sends the message history to the LLM and returns the model response.

    Parameters:
      messages — list of dialog messages. Each message is a LangChain object:
                   SystemMessage(content="...")   — instruction for the model (agent role)
                   HumanMessage(content="...")    — message from the user
                   AIMessage(...)                 — previous model response
                   ToolMessage(content="...", tool_call_id="...") — tool result

      tools   — list of tool descriptions (OpenAI function calling schema or LangChain tools).

    Returns AIMessage:
      msg.content    — text response (str)
      msg.tool_calls — list of tool calls:
                         "name" — tool name
                         "args" — arguments (already parsed dict)
                         "id"   — unique call identifier
    """
    if tools:
        return llm.bind_tools(tools).invoke(messages)
    return llm.invoke(messages)


# Product catalog
CATALOG = [
    {"id": "p1",  "name": "Sony WH-1000XM5",            "category": "headphones", "brand": "Sony",     "price": 349, "color": "black",    "rating": 4.8, "tags": ["wireless", "noise-cancelling", "premium"]},
    {"id": "p2",  "name": "Sony WH-CH720N",              "category": "headphones", "brand": "Sony",     "price": 129, "color": "blue",     "rating": 4.4, "tags": ["wireless", "budget", "noise-cancelling"]},
    {"id": "p3",  "name": "Bose QuietComfort Ultra",     "category": "headphones", "brand": "Bose",     "price": 379, "color": "white",    "rating": 4.7, "tags": ["wireless", "noise-cancelling", "premium"]},
    {"id": "p4",  "name": "Apple AirPods Pro 2",         "category": "earbuds",    "brand": "Apple",    "price": 249, "color": "white",    "rating": 4.6, "tags": ["wireless", "noise-cancelling", "ios"]},
    {"id": "p5",  "name": "Anker Soundcore Liberty 4 NC","category": "earbuds",    "brand": "Anker",    "price": 99,  "color": "black",    "rating": 4.3, "tags": ["wireless", "budget", "noise-cancelling"]},
    {"id": "p6",  "name": "Logitech MX Master 3S",       "category": "mouse",      "brand": "Logitech", "price": 109, "color": "graphite", "rating": 4.8, "tags": ["wireless", "productivity", "premium"]},
    {"id": "p7",  "name": "Logitech Pebble 2",           "category": "mouse",      "brand": "Logitech", "price": 34,  "color": "white",    "rating": 4.2, "tags": ["wireless", "budget", "portable"]},
    {"id": "p8",  "name": "Keychron K2",                 "category": "keyboard",   "brand": "Keychron", "price": 89,  "color": "black",    "rating": 4.5, "tags": ["wireless", "mechanical", "compact"]},
    {"id": "p9",  "name": "NuPhy Air75",                 "category": "keyboard",   "brand": "NuPhy",    "price": 139, "color": "gray",     "rating": 4.6, "tags": ["wireless", "mechanical", "low-profile"]},
    {"id": "p10", "name": "Amazon Kindle Paperwhite",    "category": "ereader",    "brand": "Amazon",   "price": 149, "color": "black",    "rating": 4.7, "tags": ["reading", "portable", "gift"]},
]


@dataclass
class ShopState:
    """Session state: cart and last search results."""
    cart: list = field(default_factory=list)
    last_results: list = field(default_factory=list)


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: Any = None


class ToolTracer:
    """Collects all tool calls."""
    def __init__(self):
        self.calls: list[ToolCallRecord] = []

    def record(self, name: str, args: dict, result: Any = None) -> None:
        self.calls.append(ToolCallRecord(name=name, args=args, result=result))

    def called(self, name: str) -> bool:
        return any(c.name == name for c in self.calls)

    def get_calls(self, name: str) -> list:
        return [c for c in self.calls if c.name == name]

    def print_trace(self) -> None:
        print("=== Tool Call Trace ===")
        for i, c in enumerate(self.calls, 1):
            print(f"  {i}. {c.name}({json.dumps(c.args, ensure_ascii=False)[:80]})")
            if c.result is not None:
                print(f"     -> {json.dumps(c.result, ensure_ascii=False)[:100]}")
        print("=====================")


class ShopTools:
    """Shop logic — search and add to cart."""
    def __init__(self, catalog):
        self.catalog = catalog

    def search_products(self, query: str = "", category: str | None = None,
                        brand: str | None = None, max_price: float | None = None,
                        sort_by: str | None = None) -> list:
        results = []
        q_words = query.lower().split() if query else []
        for item in self.catalog:
            hay = f"{item['name']} {item['category']} {item['brand']} {' '.join(item['tags'])}".lower()
            if q_words and not all(w in hay for w in q_words): continue
            if category and item["category"] != category: continue
            if brand and item["brand"].lower() != brand.lower(): continue
            if max_price is not None and item["price"] > float(max_price): continue
            results.append(copy.deepcopy(item))
        if sort_by == "price_asc": results.sort(key=lambda x: x["price"])
        elif sort_by == "rating_desc": results.sort(key=lambda x: -x["rating"])
        return results

    def add_to_cart(self, state: ShopState, product_id: str, quantity: int = 1) -> dict:
        product = next((p for p in self.catalog if p["id"] == product_id), None)
        if not product:
            return {"ok": False, "error": f"Product {product_id} not found"}
        existing = next((r for r in state.cart if r["product_id"] == product_id), None)
        if existing:
            existing["quantity"] += quantity
        else:
            state.cart.append({"product_id": product_id, "name": product["name"],
                                "price": product["price"], "quantity": quantity})
        return {"ok": True, "cart_size": len(state.cart)}


@dataclass
class AgentContext:
    """Shared context passed between agents in Task 3."""
    query: str
    max_price: float | None = None
    candidates: list[dict] = field(default_factory=list)
    pros: dict[str, str] = field(default_factory=dict)   # product_id -> pros description
    cons: dict[str, str] = field(default_factory=dict)   # product_id -> cons description
    best: dict | None = None
    cart_result: dict | None = None


TOOLS = ShopTools(CATALOG)
print("Template loaded.")
print(f"  Model: {MODEL_NAME}")
print(f"  Catalog: {len(CATALOG)} products")
print(f"  Utilities: AgentContext, ToolTracer, ShopTools, convert_to_openai_tool")
print(f"  LangChain: HumanMessage, SystemMessage, AIMessage, ToolMessage")

#test_ollama_connection()

#msg = llm.invoke([HumanMessage(content="Reply exactly: CONNECTED")])

#print("🔹 Статус:", msg.content.strip())
#print("🔹 Tool calls:", bool(msg.tool_calls))

def search_products(
    query: str = "",
    category: str | None = None,
    brand: str | None = None,
    max_price: float | None = None,
    sort_by: str | None = None,
) -> list:
    """Searches the product catalog. All words in `query` must match name, category, brand, or tags (logical AND).

    Args:
        query: Keywords for text search across name, category, brand, and tags. 
               More words narrow the results. Leave empty for no text filter.
        category: Product category filter (e.g., "headphones", "mouse"). Optional.
        brand: Brand filter (e.g., "Sony", "Logitech"). Case-insensitive. Optional.
        max_price: Maximum allowed price. Products with higher price are excluded. Optional.
        sort_by: Sort order. Use "price_asc" for cheapest first, or "rating_desc" for highest rated first. Optional.

    Returns:
        List of matching product dictionaries.
    """
    pass


#{"id": "p1",
#"name": "Sony WH-1000XM5",
#"category": "headphones",
#"brand": "Sony",
#"price": 349,
#"color": "black",
#"rating": 4.8,
#"tags": ["wireless", "noise-cancelling", "premium"]},
# hay = f"{item['name']} {item['category']} {item['brand']} {' '.join(item['tags'])}".lower()


def add_to_cart(product_id: str, quantity: int = 1) -> dict:
    """Adds a product to the user's shopping cart.

    Args:
        product_id: The unique ID of the product from the catalog (e.g., 'p1', 'p7').
        quantity: Number of items to add. Defaults to 1.

    Returns:
        Dictionary with 'ok' (bool) and 'cart_size' (int) on success, or 'error' (str) if product not found.
    """
    pass

    # YOUR CODE HERE: add a docstring
    #...

# YOUR CODE HERE: generate the schema
SHOP_TOOLS_SCHEMA = [
    convert_to_openai_tool(search_products),
    convert_to_openai_tool(add_to_cart),
]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2. Memory Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROFILE_PATH = Path("user_profile.json")
# Recommended profile fields:
#   name       — user name
#   brand      — preferred brand
#   max_price  — maximum price
#   color      — preferred color
#   category   — category of interest

def load_profile(path: Path = PROFILE_PATH) -> dict:
    """Loads profile from JSON. Returns {} if file does not exist."""
    # YOUR CODE HERE
    restored_dict = {}
    if path.exists():
        restored_dict = json.loads(path.read_text(encoding="utf-8"))
    return restored_dict
    #raise NotImplementedError

def save_profile(profile: dict, path: Path = PROFILE_PATH) -> None:
    """
    Saves the profile dict to a file as JSON.
    Args:
        profile: Dictionary with fields user_profile.
        path: user_profile path.

    """
    # YOUR CODE HERE
    json_string = json.dumps(profile, ensure_ascii=False, indent=2)
    path.write_text(json_string, encoding="utf-8")
    
    #raise NotImplementedError

# def update_profile(key: str, val: str, prof: dict, path:  Path = PROFILE_PATH):
#     """
#     SUMMARY.

#     Parameters
#     ----------
#     key : key in user_profile. May be one of the  "name"— user name, "brand"— preferred brand, "max_price"— maximum price
#     "color"— preferred color or "category" — category of interest"
#     val : value of the key

#     Returns:
#         JSON of the user profile

#     """
#     prof = load_profile(path)
#     prof[key] = val
#     save_profile(prof, path)
#     return f"Пользователя зовут {prof.get("name", None)}, он предпочитает бренд {prof.get("brand", None)}\
#         покупает товары дешевле {prof.get("max_price", None)}, любит {prof.get("color", None)} цвет\
#             и интересуется категорией {prof.get("category", None)}"
    



def run_memory_agent(
    user_message: str,
    state: ShopState,
    tools: ShopTools,
    tracer: ToolTracer,
    history: list,
    profile_path: Path = PROFILE_PATH,
) -> tuple:
    """
    Memory agent. Extends run_shopping_agent with long-term and short-term memory.

    Long-term memory:
      - Loads profile from file (load_profile) on each run
      - Passes profile to agent via SystemMessage
      - update_profile tool updates the profile on disk when preferences are first mentioned

    Short-term memory:
      - history contains the full message history from previous turns (including ToolMessages)
      - This allows the agent to "see" the results of past searches in the next turn
      - Added to the query before calling the LLM

    Returns (response: str, updated_history: list).
    Hint: save ALL messages to history (HumanMessage, AIMessage, ToolMessage),
    so the agent knows what was found in the next turn.
    """
    
    def update_profile(key: str, val: str):
        """
        Updates a user preference in the long-term memory profile.

        Args:
            key : key in user_profile. May be one of: "name", "brand", "max_price", "color", "category".
            val: The value to save for the given key.

        Returns:
            Dictionary with "ok": True and the updated profile.

        """
        current_profile = load_profile(path=profile_path)
        current_profile[key] = val
        save_profile(current_profile, path=profile_path)
        return {"ok": True, "profile": current_profile}
        
    
    
    SHOP_TOOLS_SCHEMA_WITH_MEMORY = SHOP_TOOLS_SCHEMA + [
        convert_to_openai_tool(update_profile)
        # YOUR CODE HERE — SHOP_TOOLS_SCHEMA + update_profile tool
        # update_profile: takes key (recommended: name | brand | max_price | color | category)
        #                 and value — saves a user preference to the profile
    ]
    # YOUR CODE HERE
    instr = "Ты агент по продаже электронной техники в магазине. Твоя задача найти по запросу пользователя товар\
        в каталоге и положить его в корзину. У тебя есть два инструмента для этого. Не придумывай никаких товаров, которые\
        не были найдены поиском. Не придумывай id для товаров. Если товара нет, который подходил критериям поиска,\
        то не добавляй ничего. Также иногда пользователь может уточнять данные по себе или находить данные по его\
            предпочтениям."
    instr = "You're an agent selling electronic equipment in a store. Your task is to find the product according to the user's request\
        in the catalog and put it in the basket. You have two tools for that. Don't invent any products that\
        were not found by the search. Don't create product IDs. If there is no product that fits the search criteria,\
        then do not add anything. Also, sometimes the user can refine the data on himself or find the data on his\
        preferences."
    instr = (
        "You are a helpful shopping assistant for an electronics store.\n\n"
        "RULES:\n"
        "1. If the user shares personal preferences (name, brand, max_price, color, category) "
        "WITHOUT explicitly asking to search or buy, call `update_profile` for each preference. "
        "After updating, reply confirming you saved them. DO NOT search for products.\n"
        "2. Only call `search_products` or `add_to_cart` when the user explicitly asks to find, "
        "compare, or purchase items.\n"
        "3. Always use the provided tools. Never invent product IDs, prices, or availability.\n"
        "4. Respond concisely in the user's language."
    )    
        
    #history.append(SystemMessage(content=instr))
    prof = load_profile(profile_path)
    # prof_info = "Также ты знаешь какую-то информацию о пользователе. Если ничего не знаешь, то будет пометка None. При обращении ты можешь понимать,\
    #     что некоторая информация изменилась по пользоввателю и ты можешь обновить её используя специальные инструмент. Вот информация о пользователе: "
    prof_info = "You also know some information about the user. If you don't know anything,\
        it will be marked None. When contacting you, you may understand\
        that some information has changed about the user and you can update\
            it using a special tool. Here is the user information: "
    # prof_info = prof_info + f"Пользователя зовут {prof.get("name", None)}, он предпочитает бренд {prof.get("brand", None)}\
    #     покупает товары дешевле {prof.get("max_price", None)}, любит {prof.get("color", None)} цвет\
    #         и интересуется категорией {prof.get("category", None)}"
    prof_info = prof_info + f"The user's name is {prof.get("name", None)}, and he prefers the brand {prof.get("brand", None)}\
        buys goods cheaper than {prof.get("max_price", None)}, likes {prof.get("color", None)} color\
            and is interested in the category {prof.get("category", None)}"
    # "brand"— preferred brand, "max_price"— maximum price
    # "color"— preferred color or "category" — category of interest
    # if prof == {}:
    #     prof_info = f"Ничего не известно об этом пользователе"
    # else:
    #     prof_info = f"Пользователя зовут {prof.get("name", None)}"
    # history.append(SystemMessage(content=json.dumps(prof, ensure_ascii=False, indent=2)))
    #history.append(SystemMessage(content=prof_info))
    #history.append(HumanMessage(content=user_message))
    
    profile_lines = []
    if prof.get("name"): profile_lines.append(f"Name: {prof['name']}")
    if prof.get("brand"): profile_lines.append(f"Preferred brand: {prof['brand']}")
    if prof.get("max_price"): profile_lines.append(f"Max budget: ${prof['max_price']}")
    if prof.get("color"): profile_lines.append(f"Preferred color: {prof['color']}")
    if prof.get("category"): profile_lines.append(f"Interested in: {prof['category']}")

    prof_context = "\n".join(profile_lines) if profile_lines else "No saved preferences yet."
    
    if not history:
        history.append(SystemMessage(content=instr))
        history.append(SystemMessage(content=f"USER PROFILE (long-term memory):\n{prof_context}"))
        history.append(HumanMessage(content=user_message))
    else:
        # Если история уже есть (повторный turn), просто добавляем новый запрос
        history.append(HumanMessage(content=user_message))
    
    while True:
        msg = llm_chat(history, SHOP_TOOLS_SCHEMA_WITH_MEMORY)
        history.append(msg)
        if not msg.tool_calls:
            return msg.content, history
        #tool_req = msg.tool_calls[0]
        #print(len(msg.tool_calls))
        for tool_req in msg.tool_calls:
            match tool_req["name"]:
                case "search_products":
                    resfind = tools.search_products(**tool_req["args"])
                    state.last_results = resfind
                    tracer.record(tool_req["name"], tool_req["args"], resfind) 
                    history.append(ToolMessage(content=json.dumps(resfind, ensure_ascii=False), tool_call_id=tool_req["id"]))
                    continue
                case "add_to_cart":
                    rescart = tools.add_to_cart(state, **tool_req["args"])
                    tracer.record(tool_req["name"], tool_req["args"], rescart) 
                    history.append(ToolMessage(content = json.dumps(rescart, ensure_ascii=False), tool_call_id=tool_req["id"]))
                    if rescart["ok"]:
                        print("Товар добавили")
                    else:
                        print("Какая-то ошибка")
                    continue
                case "update_profile":
                    result = update_profile(**tool_req["args"])
                    tracer.record(tool_req["name"], tool_req["args"], result) 
                    history.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tool_req["id"]))
                    continue
    
    raise NotImplementedError


# [2.A] Saving preferences
_p2a = Path("_demo_profile_2a.json")
if _p2a.exists(): _p2a.unlink()
_s2a = ShopState(); _t2a = ToolTracer(); _h2a = []
_r2a, _h2a = run_memory_agent(
    "My name is Anna, I prefer Sony and my budget is 200 dollars",
    _s2a, TOOLS, _t2a, _h2a, _p2a
)
_prof2a = load_profile(_p2a)
assert _t2a.called("update_profile") and _prof2a.get("brand") == "Sony"
print("OK 2.A")

# [2.B] New session uses profile (history=[])
_p2b = Path("_demo_profile_2b.json")
save_profile({"name": "Boris", "brand": "Logitech", "max_price": "150"}, _p2b)
_s2b = ShopState(); _t2b = ToolTracer(); _h2b = []
_r2b, _ = run_memory_agent("What is my name and what is my budget?", _s2b, TOOLS, _t2b, _h2b, _p2b)
assert "Boris" in _r2b
print("OK 2.B")

# [2.C] Short-term memory — turn 2 remembers turn 1
_p2c = Path("_demo_profile_2c.json")
if _p2c.exists(): _p2c.unlink()
_s2c = ShopState(); _h2c = []
_, _h2c = run_memory_agent(
    "Find wireless headphones under 150 dollars", _s2c, TOOLS, ToolTracer(), _h2c, _p2c
)
assert len(_h2c) >= 2
_t2c2 = ToolTracer()
_, _h2c = run_memory_agent(
    "Add the first one found to cart", _s2c, TOOLS, _t2c2, _h2c, _p2c
)
assert _t2c2.called("add_to_cart") and len(_s2c.cart) == 1
print(f"OK 2.C: added '{_s2c.cart[0]['name']}'")

