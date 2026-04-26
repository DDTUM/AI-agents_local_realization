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

#***********************************************************************************************************
# ╔══════════════════════════════════════════════════════════════╗
# ║               YOUR CODE — THREE TASKS                        ║
# ╚══════════════════════════════════════════════════════════════╝

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1. Tool-Calling Agent (ReAct loop)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1.1. Define SHOP_TOOLS_SCHEMA — tool descriptions for the LLM.
#
# Below are stub functions with signatures but no descriptions.
# The LLM needs to understand what each tool does and what its parameters mean.
#
# Task: add a docstring (description + Args) to each function.
# The convert_to_openai_tool() function from the template will generate the JSON schema automatically.
# For docstring format details, see Google-style docstrings.

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

#print(SHOP_TOOLS_SCHEMA[0])
#print("***")
#print(json.dumps(SHOP_TOOLS_SCHEMA, indent=1))

# Проверка работы llm
#msg = llm_chat([HumanMessage(content="Reply with exactly one word: CONNECTED")])
#print("🔹 Ответ модели:", msg.content.strip())
#print("🔹 Есть tool_calls:", bool(msg.tool_calls))


"""
llm_chat(messages: list, tools: list | None = None) -> AIMessage
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
import pprint
# 1.2. Implement run_shopping_agent — a ReAct shop agent.
def run_shopping_agent(user_message: str, state: ShopState, tools: ShopTools, tracer: ToolTracer) -> str:
    """
    ReAct shop agent. Receives a user message and iteratively:
      1. Calls the LLM with the history and tool schema.
      2. If the LLM returns tool_calls — executes each tool:
           search_products -> saves result to state.last_results, records in tracer
           add_to_cart     -> adds product to state.cart, records in tracer
         Adds a ToolMessage with the result to the history and repeats the loop.
      3. If tool_calls is empty — returns the text response from the LLM.
    """
    # YOUR CODE HERE
    dialogue = []
    instr = "Ты агент по продаже электронной техники в магазине. Твоя задача найти по запросу пользователя товар в каталоге и положить его в корзину.\
    У тебя есть два инструмента. Не придумывай никаких товаров, которые не были найдены поиском. Не придумывай id для товаров. Если товара нет, который\
    подходил критериям поиска, то не добавляй ничего."
    instr = "You're an agent selling electronic equipment in a store. Your task is to find the product in the catalog at the user's request and put it in the basket.\
    You have two tools. Do not invent any products that were not found by the search. Don't create product IDs. If there is no product, which\
    If it matches the search criteria, then don't add anything."
    dialogue.append(SystemMessage(content=instr))
    dialogue.append(HumanMessage(content=user_message))
    while True:
        msg = llm_chat(dialogue, SHOP_TOOLS_SCHEMA)
        dialogue.append(msg)
        # print("Ответ:")
        # pprint.pprint(msg.content)
        # print("Инструмент")
        # pprint.pprint(msg.tool_calls)
        # print("***")
        #dialogue.append(msg.tool_calls)
        if not msg.tool_calls:
            return msg.content
        tool_req = msg.tool_calls[0]
        #print(len(msg.tool_calls))
        
        match tool_req["name"]:
            case "search_products":
                #resfind = TOOLS.search_products(tool_req["args"]["query"], tool_req["args"]["category"],\
                #                                tool_req["args"]["brand"], tool_req["args"]["max_price"],\
                #                                    tool_req["args"]["sort_by"])
                resfind = tools.search_products(**tool_req["args"])
                # (self, query: str = "", category: str | None = None,
                #                    brand: str | None = None, max_price: float | None = None,
                #                    sort_by: str | None = None)
                state.last_results = resfind
                tracer.record(tool_req["name"], tool_req["args"], resfind) 
                #ToolMessage(content="...", tool_call_id=tool_req["id"])
                dialogue.append(ToolMessage(content=json.dumps(resfind, ensure_ascii=False), tool_call_id=tool_req["id"]))
                continue
            case "add_to_cart":
                #rescart = TOOLS.add_to_cart(state, tool_req["args"]["product_id"], tool_req["args"]["quantity"]) # должно быть два что добваить и сколько
                rescart = tools.add_to_cart(state, **tool_req["args"])
                #(self, state: ShopState, product_id: str, quantity: int = 1) -> dict:
                tracer.record(tool_req["name"], tool_req["args"], rescart) 
                #return {"ok": False, "error": f"Product {product_id} not found"}
                #return {"ok": True, "cart_size": len(state.cart)}
                dialogue.append(ToolMessage(content = json.dumps(rescart, ensure_ascii=False), tool_call_id=tool_req["id"]))
                if rescart["ok"]:
                    print("Товар добавили")
                else:
                    print("Какая-то ошибка")
                continue
        

    raise NotImplementedError



# [1.A] Search with price filter
_s1a = ShopState(); _t1a = ToolTracer()
_r1a = run_shopping_agent("Find wireless headphones under 150 dollars", _s1a, TOOLS, _t1a)
_t1a.print_trace()
assert _t1a.called("search_products"), "FAIL: search_products was not called"
assert all(p["price"] <= 150 for p in _s1a.last_results)
print("OK 1.A")

# [1.B] Search + add the cheapest
_s1b = ShopState(); _t1b = ToolTracer()
_r1b = run_shopping_agent(
    "Find a wireless mouse under 120 dollars and add the cheapest one to cart",
    _s1b, TOOLS, _t1b
)
assert _t1b.called("search_products") and _t1b.called("add_to_cart")
assert len(_s1b.cart) == 1 and _s1b.cart[0]["product_id"] == "p7"
print("OK 1.B")

# [1.C] Best keyboard
_s1c = ShopState(); _t1c = ToolTracer()
_r1c = run_shopping_agent(
    "Find a wireless keyboard with the best rating and add it to cart",
    _s1c, TOOLS, _t1c
)
assert _t1c.called("search_products") and _t1c.called("add_to_cart")
added = next(p for p in CATALOG if p["id"] == _s1c.cart[0]["product_id"])
assert added["category"] == "keyboard"
print(f"OK 1.C: '{added['name']}' (rating {added['rating']})")
