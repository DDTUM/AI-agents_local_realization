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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3. Multi-Agent System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Implement a system of four agents + an orchestrator.
# Goal — find the best product and honestly describe its pros and cons.
# Agents work in a chain via a shared AgentContext object (defined in the template).
#
# RetrieverAgent (LLM + tools)
#   Searches for up to 5 relevant products via search_products.
#   Fills ctx.candidates and ctx.max_price.
#   Important: only pass the search tool (not add_to_cart).
#
# ProsAgent (LLM, no tools)
#   For each product in ctx.candidates, writes 1-2 sentences of pros.
#   Fills ctx.pros (dict: product_id -> pros string).
#   Records an "analyze_pros" call in tracer.
#
# ConsAgent (LLM, no tools)
#   For each product in ctx.candidates, writes 1-2 sentences of cons.
#   Fills ctx.cons (dict: product_id -> cons string).
#   Records an "analyze_cons" call in tracer.
#
# RankerAgent (no LLM — logic only)
#   Picks the best product from ctx.candidates:
#     - Filters by ctx.max_price (if set)
#     - Among remaining: highest rating; if tied — lowest price
#   Records a "rank_candidates" call in tracer. Fills ctx.best.
#
# CoordinatorAgent (orchestrator)
#   Runs agents in a chain, maintains a trace list.
#   Trace keys: "delegate_retriever", "delegate_pros", "delegate_cons",
#               "delegate_ranker", "delegate_cart".
#   No CartAgent needed — if the user asks to add to cart,
#   CoordinatorAgent does it itself via tools.add_to_cart after ranking.
#   Returns AgentResult with response, trace, and context.
#   The response should include: product name, price, rating, pros and cons.

@dataclass
class AgentResult:
    response: str
    trace: list
    context: AgentContext
    
RetrieverAgent_Tools = [
    convert_to_openai_tool(search_products),
]



class RetrieverAgent:
    # RetrieverAgent (LLM + tools)
    #   Searches for up to 5 relevant products via search_products.
    #   Fills ctx.candidates and ctx.max_price.
    #   Important: only pass the search tool (not add_to_cart).

    def run(self, ctx: AgentContext, state: ShopState, tools: ShopTools, tracer: ToolTracer) -> AgentContext:
        """Searches for products via LLM+tools. Fills ctx.candidates and ctx.max_price."""
        # YOUR CODE HERE
        dialogue = []
        instr = "Ты агент по продаже электронной техники в магазине. Твоя задача найти по запросу пользователя товары в каталоге.\
            У тебя есть инструмент для поиска. Не придумывай никаких товаров, которые не были найдены поиском. \
            Не придумывай id для товаров. Если товара нет, который\
            подходил критериям поиска, то не добавляй ничего. Тебе надо найти до 5 товаров, подходящих критерию поиска."
        instr = "You're an agent selling electronic equipment in a store. Your task is to find the products in the catalog\
            according to the user's request.\
            You have a search tool. Do not invent any products that were not found by the search. \
            Don't invent product IDs. If there is no product, which\
            If it matches the search criteria, then don't add anything. You need to find up to 5 products that match the search criteria."
        dialogue.append(SystemMessage(content=instr))
        dialogue.append(HumanMessage(content=ctx.query))
        candidates = []
        max_price = None
        while True:
                msg = llm_chat(dialogue, RetrieverAgent_Tools)
                dialogue.append(msg)
                if not msg.tool_calls:
                    ctx.candidates = candidates[:5]
                    if max_price is not None:
                        # ctx.candidates = candidates.copy()
                        ctx.max_price = max_price
                    return ctx
                for tool_req in msg.tool_calls:
                #tool_req = msg.tool_calls[0]
                #print(len(msg.tool_calls))
                    match tool_req["name"]:
                        case "search_products":
                            resfind = tools.search_products(**tool_req["args"])
                            state.last_results = resfind
                            candidates = candidates + resfind
                            max_price = tool_req["args"].get("max_price", None)
                            tracer.record(tool_req["name"], tool_req["args"], resfind) 
                            dialogue.append(ToolMessage(content=json.dumps(resfind, ensure_ascii=False), tool_call_id=tool_req["id"]))
                            continue
                        # case "add_to_cart":
                        #     rescart = tools.add_to_cart(state, **tool_req["args"])
                        #     tracer.record(tool_req["name"], tool_req["args"], rescart) 
                        #     dialogue.append(ToolMessage(content = json.dumps(rescart, ensure_ascii=False), tool_call_id=tool_req["id"]))
                        #     if rescart["ok"]:
                        #         print("Товар добавили")
                        #     else:
                        #         print("Какая-то ошибка")
                        #     continue
        
        # raise NotImplementedError

# ProsAgent_Tools = [
#     convert_to_openai_tool(search_products),
#     convert_to_openai_tool(add_to_cart),
# ]
class ProsAgent:
    def run(self, ctx: AgentContext, tracer: ToolTracer) -> AgentContext:
        """Finds pros for each product via LLM. Fills ctx.pros."""
        # ProsAgent (LLM, no tools)
        #   For each product in ctx.candidates, writes 1-2 sentences of pros.
        #   Fills ctx.pros (dict: product_id -> pros string).
        #   Records an "analyze_pros" call in tracer.
        # YOUR CODE HERE
        for goods in ctx.candidates:
            dialogue = []
            instr = "You're an agent selling electronic equipment in a store. Тебе известны некоторые параметры товара которые лежат в JSON.\
                Твоя задача написать 1-2 предложения с описанием преимуществ данного товара. Вот параметры:"
            instr = "You're an agent selling electronic equipment in a store. You know some of the product parameters that are in JSON.\
                Your task is to write 1-2 sentences describing the advantages of this product. Here are the parameters of product:"
            instr = instr + str(json.dumps(goods, ensure_ascii=False, indent=2))
            dialogue.append(SystemMessage(content=instr))
            #dialogue.append(SystemMessage(content=str(goods)))
            #dialogue.append(HumanMessage(content=ctx.query))
            msg = llm_chat(dialogue)
            dialogue.append(msg)
            ctx.pros[goods["id"]] = msg.content.strip()
            tracer.record("analyze_pros", {"product_id": goods["id"]}, msg.content)
                #pros: dict[str, str] = field(default_factory=dict)   # product_id -> pros description
        return ctx
        
        
        # raise NotImplementedError


class ConsAgent:
    def run(self, ctx: AgentContext, tracer: ToolTracer) -> AgentContext:
        """Finds cons for each product via LLM. Fills ctx.cons."""
        # YOUR CODE HERE
        for goods in ctx.candidates:
            dialogue = []
            instr = "You're an agent selling electronic equipment in a store. Тебе известны некоторые параметры товара которые лежат в JSON.\
                Твоя задача написать 1-2 предложения с описанием преимуществ данного товара. Вот параметры:"
            instr = "You're an agent selling electronic equipment in a store. You know some of the product parameters that are in JSON.\
                Your task is to write 1-2 sentences describing the disadvantages of this product. Here are the parameters of product:"
            instr = instr + str(json.dumps(goods, ensure_ascii=False, indent=2))
            dialogue.append(SystemMessage(content=instr))
            #dialogue.append(SystemMessage(content=str(goods)))
            #dialogue.append(HumanMessage(content=ctx.query))
            msg = llm_chat(dialogue)
            dialogue.append(msg)
            ctx.cons[goods["id"]] = msg.content.strip()
            tracer.record("analyze_cons", {"product_id": goods["id"]}, msg.content)
               
        return ctx
        #raise NotImplementedError

    
    # class AgentContext:
    #     """Shared context passed between agents in Task 3."""
    #     query: str
    #     max_price: float | None = None
    #     candidates: list[dict] = field(default_factory=list)
    #     pros: dict[str, str] = field(default_factory=dict)   # product_id -> pros description
    #     cons: dict[str, str] = field(default_factory=dict)   # product_id -> cons description
    #     best: dict | None = None
    #     cart_result: dict | None = None
    
class RankerAgent:
    def run(self, ctx: AgentContext, tracer: ToolTracer) -> AgentContext:
        """Picks the best product from ctx.candidates considering ctx.max_price. Fills ctx.best."""
        # YOUR CODE HERE
        # RankerAgent (no LLM — logic only)
        #   Picks the best product from ctx.candidates:
        #     - Filters by ctx.max_price (if set)
        #     - Among remaining: highest rating; if tied — lowest price
        #   Records a "rank_candidates" call in tracer. Fills ctx.best.
        
        if ctx.max_price is not None:#ctx.getattr("max_price") != None:
            ctx.candidates = [p for p in ctx.candidates if p["price"] <= ctx.max_price]
            #rank_list = [ (ctx.candidates[i], ctx.cons[ctx.candidates[i]["id"]], ctx.cons[ctx.candidates[i]["id"]])    for i in range(len(ctx.candidates))]
            #rank_list = [p for p in rank_list if p[0]["price"] <= ctx.max_price]
            #rank_list = [p for p in candidates if p[0]["price"] <= ctx.max_price]
            #rank_list1 = list(zip(*rank_list) )
            #[ctx.candidates.copy(), ctx.cons, ctx.pros]
            #ctx.candidates = [rank_list[0]]
            # for i in range(len(ctx.candidates)):
            #     goods = ctx.candidates[i]
            #     if goods["price"] > ctx.max_price:
            #         ctx.candidates.pop(i)
        if not ctx.candidates:
            ctx.best = None
            tracer.record("rank_candidates", {"filtered_count": 0, "max_price": ctx.max_price}, None)
            return ctx
        
        if ctx.candidates:
            #rank_list = [ctx.candidates.copy(), ctx.cons, ctx.pros]
            ctx.candidates.sort(key=lambda x: (-x["rating"], x["price"]))
            ctx.best = ctx.candidates[0]
            tracer.record("rank_candidates", {"best_id": ctx.best["id"], "rating": ctx.best["rating"]}, ctx.best)
        
        return ctx
        
        # raise NotImplementedError

# CoordinatorAgent (orchestrator)
#   Runs agents in a chain, maintains a trace list.
#   Trace keys: "delegate_retriever", "delegate_pros", "delegate_cons",
#               "delegate_ranker", "delegate_cart".
#   No CartAgent needed — if the user asks to add to cart,
#   CoordinatorAgent does it itself via tools.add_to_cart after ranking.
#   Returns AgentResult with response, trace, and context.
#   The response should include: product name, price, rating, pros and cons.
class CoordinatorAgent:
    def __init__(self):
        self.retriever = RetrieverAgent()
        self.pros_agent = ProsAgent()
        self.cons_agent = ConsAgent()
        self.ranker = RankerAgent()

    def run(self, user_message: str, state: ShopState, tools: ShopTools) -> AgentResult:
        """Orchestrates agents. Returns AgentResult with response, trace, and context."""
        # YOUR CODE HERE
        ctx = AgentContext(query=user_message)
        tracer =  ToolTracer()
        trace = []
        
        trace.append("delegate_retriever")
        ctx = self.retriever.run(ctx, state, tools, tracer)
        trace.append("delegate_pros")
        ctx = self.pros_agent.run(ctx, tracer)
        trace.append("delegate_cons")
        ctx = self.cons_agent.run(ctx, tracer)
        trace.append("delegate_ranker")
        ctx = self.ranker.run(ctx, tracer)
        
        if ctx.best is None:
            return AgentResult(
                response="No suitable products found within your criteria.",
                trace=trace,
                context=ctx
            )
        
        keywords = "add put place take"
        if ctx.best is not None:
            if any([i in user_message.lower() for i in keywords.split()]):
                rescart = tools.add_to_cart(state, ctx.best["id"])
                tracer.record("add_to_cart", {"product_id": ctx.best["id"]}, rescart)
                trace.append("delegate_cart")
                # if rescart["ok"]:
                #     print("Товар добавили")
                # else:
                #     print("Какая-то ошибка")
            #product name, price, rating, pros and cons
            # response = (ctx.best["name"], ctx.best["price"], ctx.best["rating"], ctx.pros[ctx.best["id"]], ctx.cons[ctx.best["id"]])
            response = (
            f"Best choice: {ctx.best['name']} (${ctx.best['price']}, {ctx.best['rating']})\n"
            f"Pros: {ctx.pros.get(ctx.best['id'], 'N/A')}\n"
            f"Cons: {ctx.cons.get(ctx.best['id'], 'N/A')}"
            )
            #context = ""
            
        
        return AgentResult(response=response, trace=trace, context=ctx)
        # raise NotImplementedError





_s3a = ShopState()
_res3a = CoordinatorAgent().run(
    "Find the best wireless mouse under 120 dollars and add it to cart", _s3a, TOOLS
)
assert "delegate_retriever" in _res3a.trace
assert "delegate_pros" in _res3a.trace and "delegate_cons" in _res3a.trace
assert "delegate_ranker" in _res3a.trace and "delegate_cart" in _res3a.trace
assert len(_s3a.cart) == 1 and _s3a.cart[0]["product_id"] == "p6"
assert _res3a.context.best is not None and _res3a.context.best["id"] == "p6"
assert len(_res3a.context.pros) > 0 and len(_res3a.context.cons) > 0
print("OK 3.A")

# [3.B] Search only, no add to cart
_s3b = ShopState()
_res3b = CoordinatorAgent().run("Find a wireless keyboard", _s3b, TOOLS)
assert "delegate_retriever" in _res3b.trace
assert "delegate_pros" in _res3b.trace and "delegate_cons" in _res3b.trace
assert "delegate_ranker" in _res3b.trace
assert "delegate_cart" not in _res3b.trace and len(_s3b.cart) == 0
assert _res3b.context.best is not None
print("OK 3.B")

# [3.C] RankerAgent — price tie-break with equal rating
_ctx3c = AgentContext(query="test", candidates=[
    {"id": "x1", "name": "A", "price": 200, "rating": 4.8},
    {"id": "x2", "name": "B", "price": 150, "rating": 4.8},
    {"id": "x3", "name": "C", "price": 100, "rating": 4.5},
])
_tr3c = ToolTracer()
_ctx3c = RankerAgent().run(_ctx3c, _tr3c)
assert _ctx3c.best["id"] == "x2" and _tr3c.called("rank_candidates")
print("OK 3.C")

# [3.D] RankerAgent respects ctx.max_price
_ctx3d = AgentContext(
    query="mouse under 120 dollars",
    max_price=120.0,
    candidates=[
        {"id": "expensive", "name": "Super Mouse",  "price": 200, "rating": 4.9},
        {"id": "p6",        "name": "MX Master 3S", "price": 109, "rating": 4.8},
        {"id": "p7",        "name": "Pebble 2",      "price": 34,  "rating": 4.2},
    ],
)
_tr3d = ToolTracer()
_ctx3d = RankerAgent().run(_ctx3d, _tr3d)
assert _ctx3d.best is not None and _ctx3d.best["id"] == "p6"
print("OK 3.D: context passed correctly, max_price is respected")

