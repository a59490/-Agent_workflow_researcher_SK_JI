"""
Convo Orchestrator (CLI)

English-first async assistant built with Semantic Kernel, Azure OpenAI, and Tavily.
It implements an Orchestrator + Researcher pattern with a predictable JSON decision
contract, a thread-safe HTML logger, and a tiny per-chat memory.

Key ideas:
- Keep the orchestration logic simple and explicit (no hidden tool routing).
- Ask the Orchestrator to produce machine-readable JSON when deciding on research.
- Perform Tavily calls outside the model (direct plugin functions), then feed
  results back to the Orchestrator to compose the final answer.
- Always sanitize external content when logging to HTML and guard file writes
  with an asyncio lock.

This file is intentionally well-commented for learning and maintenance.
"""
import os
import sys
import re
import html
import json
import asyncio
import uuid
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from typing import Annotated, Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
import aioconsole

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import kernel_function
from tavily import TavilyClient

# =========================
# Console UI (Rich)
# =========================
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.theme import Theme

    _theme = Theme(
        {
            "info": "bold cyan",
            "success": "bold green",
            "warning": "bold yellow",
            "error": "bold red",
            "muted": "dim",
            "title": "bold blue",
            "section": "blue",
        }
    )
    console = Console(theme=_theme)
    console_err = Console(stderr=True, theme=_theme)

    def log_info(msg: str) -> None:
        console.print(f"[info]ℹ️ {msg}[/info]")

    def log_success(msg: str) -> None:
        console.print(f"[success]✅ {msg}[/success]")

    def log_warning(msg: str) -> None:
        console.print(f"[warning]⚠️ {msg}[/warning]")

    def log_error(msg: str) -> None:
        console_err.print(f"[error]❌ {msg}[/error]")

    def log_note(msg: str) -> None:
        console.print(f"[muted]{msg}[/muted]")

    def log_rule(title: str) -> None:
        console.rule(f"[title]{title}[/title]")

    def log_section(title: str, content: str, *, markdown: bool = False, border: str = "section") -> None:
        body = Markdown(content) if markdown else Text.from_markup(content)
        console.print(Panel(body, title=title, title_align="left", border_style=border))

except Exception:
    # Fallback if rich is unavailable
    console = None
    console_err = None

    def log_info(msg: str) -> None:
        print(f"ℹ️ {msg}")

    def log_success(msg: str) -> None:
        print(f"✅ {msg}")

    def log_warning(msg: str) -> None:
        print(f"⚠️ {msg}")

    def log_error(msg: str) -> None:
        print(f"❌ {msg}", file=sys.stderr)

    def log_note(msg: str) -> None:
        print(msg)

    def log_rule(title: str) -> None:
        print(f"\n==== {title} ====\n")

    def log_section(title: str, content: str, *, markdown: bool = False, border: str = "section") -> None:
        print(f"\n[{title}]\n{content}\n")


# =========================
# Configuration & Utils
# =========================
load_dotenv()

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.getenv("LOG_FILE", "interaction_log.html")
_log_lock = asyncio.Lock()

# Tasteful defaults (caps)
MAX_FACTS = 50
MAX_NOTES = 200

# Routing keywords (fallback, lightweight)
EXTERNAL_KEYWORDS = {
    "today", "now", "yesterday", "news", "release", "version",
    "price", "quote", "rate", "exchange", "stock",
    "law", "decree", "regulation", "hours", "open", "closed",
    "forecast", "weather", "result", "score", "where to buy",
    "available", "availability", "download", "changelog", "docs",
    "latest", "current", "recent",
}

LOCAL_PATTERNS = [
    r"\b(my\s+name\s+is|i\s+am\s+called|call\s+me)\b",
    r"\b(my\s+preferenc|preferences)\b",
    r"\b(conversation\s+summary|recap|what\s+i\s+said)\b",
    r"\b(memory|remember\s+this|store\s+this)\b",
]


def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        log_error(f"Missing environment variable: {var}")
    return val or ""


def _all_env_ok(vars_: List[str]) -> bool:
    missing = [v for v in vars_ if not os.getenv(v)]
    if missing:
        log_warning("The following environment variables are missing:")
        for v in missing:
            log_error(f" - {v}")
        return False
    return True


# =========================
# Tavily (async wrappers)
# =========================
async def tavily_search_async(client: TavilyClient, *, query: str, max_results: int = 3, timeout: float = 12.0) -> Dict[str, Any]:
    """Runs Tavily.search in a non-blocking way with a timeout."""

    def _call():
        return client.search(query=query, max_results=max_results, search_depth="advanced")

    return await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout)


async def tavily_extract_async(client: TavilyClient, *, url: str, timeout: float = 15.0) -> Dict[str, Any]:
    """Runs Tavily.extract in a non-blocking way with a timeout."""

    def _call():
        return client.extract(url)

    return await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout)


# =========================
# Plugins
# =========================
class RespondPotato:
    @kernel_function(description="Answers potato-related questions.")
    async def respond_to_potato(self, question: Annotated[str, "Potato-related question"]) -> str:
        if "potato" in question.lower():
            return (
                "Potatoes are versatile: you can boil, bake, mash, or fry them. "
                "They are rich in starch and pair well with many dishes."
            )
        return "This plugin only answers potato-related questions."


class SearchOnline:
    def __init__(self, tavily_client: TavilyClient):
        self._client = tavily_client

    @kernel_function(description="Use Tavily to search the web (use for all questions that are not about potatoes).")
    async def search_online(self, query: Annotated[str, "Search query"]) -> str:
        log_info("SearchOnline plugin active")
        try:
            log_info(f"🔎 Searching online for: {query}")
            response = await tavily_search_async(self._client, query=query, max_results=3)
            results = response.get("results") or []
            if isinstance(results, list) and results:
                bullets = []
                for item in results:
                    title = item.get("title") or "Untitled"
                    url = item.get("url") or ""
                    content = item.get("content") or ""
                    bullets.append(f"- **{title}**\n {content}\n Source: {url}".strip())
                output = "\n\n".join(bullets)
                log_section("🔎 SearchOnline Results", output, markdown=True)
                return output

            log_warning("SearchOnline did not find useful results.")
            return "No results were found for the search."
        except asyncio.TimeoutError:
            log_warning("⏱️ SearchOnline timed out.")
            return "The search service took too long to respond."
        except Exception as e:
            log_error(f"SearchOnline error: {e}")
            return "Could not retrieve online information at this time."


class ScrapeURL:
    def __init__(self, tavily_client: TavilyClient):
        self._client = tavily_client

    @kernel_function(description="Extracts full content from a URL using Tavily.")
    async def scrape_url(self, url: Annotated[str, "URL to extract"]) -> str:
        log_info(f"🌐 ScrapeURL plugin active for: {url}")
        try:
            response = await tavily_extract_async(self._client, url=url)
            content = response.get("content") or ""
            if content:
                log_success(f"Content extracted successfully from {url} (size: {len(content)} chars)")
                return content

            log_warning(f"No content found at {url}")
            return "No content was found on the provided page."
        except asyncio.TimeoutError:
            log_warning("⏱️ ScrapeURL timed out.")
            return "Extraction took too long and was aborted."
        except Exception as e:
            log_error(f"Error extracting {url}: {e}")
            return f"An error occurred while extracting content: {e}"


class LogToHTML:
    """Thread-safe HTML logger with a single header/footer."""

    _header = """<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Interaction Log</title>
      <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f8; color: #333; margin: 20px; }
        .log-entry { background-color: #fff; border: 1px solid #d1d9e6; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .log-entry h3 { color: #2c3e50; margin-bottom: 15px; }
        .log-entry p { margin: 8px 0; line-height: 1.6; }
        .label { font-weight: bold; color: #34495e; }
      </style>
    </head>
    <body>
    """

    _footer = "\n</body>\n</html>\n"

    def __init__(self, log_file: str = LOG_FILE):
        self._log_file = log_file

    async def _ensure_header(self) -> None:
        """Ensures the file has a header and an open body tag."""
        if not os.path.exists(self._log_file) or os.path.getsize(self._log_file) == 0:
            async with _log_lock:
                if not os.path.exists(self._log_file) or os.path.getsize(self._log_file) == 0:
                    await asyncio.to_thread(self._write_text, self._header)

    def _write_text(self, text: str) -> None:
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(text)

    @kernel_function(description="Logs the interaction to an HTML file.")
    async def log_interaction(
        self,
        user_query: Annotated[str, "User question"],
        initial_response: Annotated[str, "Initial response from orchestrator"],
        researcher_called: Annotated[str, "Whether researcher was called (YES/NO)"],
        researcher_response: Annotated[str, "Researcher response"],
        final_response: Annotated[str, "Final response from orchestrator"],
    ) -> str:
        await self._ensure_header()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Sanitize all fields
        uq = html.escape(user_query, quote=True)
        ir = html.escape(initial_response, quote=True)
        rc = html.escape(researcher_called, quote=True)
        rr = html.escape(researcher_response, quote=True)
        fr = html.escape(final_response, quote=True)

        entry = (
            f'\n<div class="log-entry">\n'
            f'  <h3>Interaction {timestamp}</h3>\n'
            f'  <p><span class="label">Question:</span> {uq}</p>\n'
            f'  <p><span class="label">Initial (Orchestrator):</span> {ir}</p>\n'
            f'  <p><span class="label">Researcher Called:</span> {rc}</p>\n'
            f'  <p><span class="label">Researcher Response:</span> {rr}</p>\n'
            f'  <p><span class="label">Final (Orchestrator):</span> {fr}</p>\n'
            f'</div>\n'
        )
        async with _log_lock:
            await asyncio.to_thread(self._write_text, entry)

        return "Interaction logged."

    def close_file(self) -> None:
        """Closes the HTML with a footer."""
        try:
            with open(self._log_file, "ab+") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size > 0:
                    with open(self._log_file, "rb+") as f2:
                        f2.seek(0, os.SEEK_END)
                        f2.write(self._footer.encode("utf-8"))
        except Exception as e:
            log_warning(f"Could not close HTML: {e}")


# =========================
# Memory (per-chat)
# =========================
@dataclass
class ChatMemory:
    session_id: str
    facts: Dict[str, str] = field(default_factory=dict)
    notes: deque = field(default_factory=lambda: deque(maxlen=MAX_NOTES))

    def to_json(self) -> str:
        return json.dumps(
            {"session_id": self.session_id, "facts": self.facts, "notes": list(self.notes)},
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "ChatMemory":
        data = json.loads(s)
        mem = cls(session_id=data["session_id"])
        mem.facts = data.get("facts", {})
        mem.notes = deque(data.get("notes", []), maxlen=MAX_NOTES)
        return mem

    def prune(self) -> None:
        if len(self.facts) > MAX_FACTS:
            overflow = len(self.facts) - MAX_FACTS
            for k in list(self.facts.keys())[:overflow]:
                self.facts.pop(k, None)


class MemoryPlugin:
    """Simple memory plugin: get/set/append and export."""

    def __init__(self, memory: ChatMemory):
        self._memory = memory

    @kernel_function(description="Get the entire memory for this chat as JSON.")
    async def get_memory(self) -> str:
        return self._memory.to_json()

    @kernel_function(description="Save (or replace) a fact in chat memory.")
    async def remember_fact(self, key: Annotated[str, "key"], value: Annotated[str, "value"]) -> str:
        k = key.strip()
        v = value.strip()
        if not k:
            return "Ignored: empty key."
        self._memory.facts[k] = v
        self._memory.prune()
        return f"OK: stored '{k}'"

    @kernel_function(description="Append a free-form note to chat memory.")
    async def remember_note(self, note: Annotated[str, "free-form note"]) -> str:
        n = (note or "").strip()
        if not n:
            return "Ignored: empty note."
        self._memory.notes.append(n)
        return "OK: note stored"

    @kernel_function(description="Remove a key from memory (if exists).")
    async def forget(self, key: Annotated[str, "key to remove"]) -> str:
        self._memory.facts.pop(key.strip(), None)
        return "OK: removed (if existed)"


# =========================
# Routing & Brief
# =========================
def _has_url(text: str) -> bool:
    t = text.lower()
    return "http://" in t or "https://" in t or "www." in t


def needs_research_and_why(user_message: str, memory: ChatMemory) -> Tuple[bool, str]:
    """Very simple fallback router (kept for contingency only)."""
    t = user_message.lower().strip()
    if _has_url(t):
        return True, "url"
    if re.search(r"\b(my\s+name\s+is|i\s+am\s+called|call\s+me)\b", t) and memory.facts.get("name"):
        return False, "has_memory:name"
    for pat in LOCAL_PATTERNS:
        if re.search(pat, t):
            return False, f"local:{pat}"
    for kw in EXTERNAL_KEYWORDS:
        if kw in t:
            return True, f"kw:{kw}"
    return False, "default"


def _rewrite_query_for_web(q: str, *, locale_hint="United States", lang_hint="en-US") -> str:
    t = q.strip()
    t = re.sub(r"\b(i|my|mine|me|our|ours)\b", " ", t, flags=re.I)
    t = re.sub(r"[!?]+", " ", t).strip()
    t = re.sub(r"\s{2,}", " ", t)
    return f"{t} — location: {locale_hint}, language: {lang_hint}. Context: {CURRENT_DATE}."


def build_research_brief(user_message: str, memory: ChatMemory) -> Dict[str, Any]:
    msg = user_message.strip()
    t = msg.lower()
    if any(w in t for w in ["price", "quote", "exchange", "rate", "stock"]):
        intent = "price/quote"
    elif any(w in t for w in ["law", "decree", "regulation", "rule", "standard"]):
        intent = "law/regulation"
    elif any(w in t for w in ["news", "latest", "today", "now", "release", "version"]):
        intent = "news/update"
    elif _has_url(t):
        intent = "summarize_url"
    else:
        intent = "general_fact"

    # Build two queries: one as-is, one stripped
    q1 = _rewrite_query_for_web(msg)
    q2 = _rewrite_query_for_web(re.sub(r"\b(what|which|when|where|why|how)\b", "", t, flags=re.I).strip())

    brief = {
        "intent": intent,
        "queries": [q1, q2],
        "notes": {"locale": "United States", "lang": "en-US", "date": CURRENT_DATE},
    }
    return brief


# =========================
# Agents & Kernel
# =========================
def build_agents(memory_plugin: Optional[Any] = None) -> Dict[str, Any]:
    # Env validation
    if not _all_env_ok(["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT", "TAVILY_API_KEY"]):
        raise SystemExit(1)

    azure_chat_service = AzureChatCompletion(
        service_id="azure_chat",
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
    )

    # Kernels
    orchestrator_kernel = Kernel()
    orchestrator_kernel.add_service(azure_chat_service)

    researcher_kernel = Kernel()
    researcher_kernel.add_service(azure_chat_service)

    logger_kernel = Kernel()
    logger_kernel.add_service(azure_chat_service)

    # Tavily
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Plugins
    potato_plugin = RespondPotato()
    search_plugin = SearchOnline(tavily_client)
    scrape_plugin = ScrapeURL(tavily_client)
    logger_plugin = LogToHTML(LOG_FILE)

    # Agents
    researcher_agent = ChatCompletionAgent(
        kernel=researcher_kernel,
        description="Agent that fetches information.",
        instructions=(
            "You are a research agent. "
            "You MUST use the Potato plugin for potato questions. "
            "For other questions, follow the orchestrator's instructions strictly. "
            f"Current date: {CURRENT_DATE}. "
            "Always respond in English."
        ),
        plugins=[potato_plugin, search_plugin, scrape_plugin],
    )

    orchestrator_plugins = []
    if memory_plugin is not None:
        orchestrator_plugins.append(memory_plugin)

    orchestrator_agent = ChatCompletionAgent(
        kernel=orchestrator_kernel,
        description="Agent that coordinates and summarizes information.",
        instructions=(
            "You are the orchestrator. Answer the user's question in a detailed, concise, and structured way. "
            f"Current date: {CURRENT_DATE}. "
            "If the question involves current events, external data, or specific entities (e.g., banks, laws, products), "
            "you may call the researcher. "
            "Always respond in English."
        ),
        plugins=orchestrator_plugins or None,
    )

    logger_agent = ChatCompletionAgent(
        kernel=logger_kernel,
        description="Agent that logs interactions.",
        instructions="You are a logging agent. Use the LogToHTML plugin to save each interaction into an HTML file.",
        plugins=[logger_plugin],
    )

    # Return agents and plugin instances (for direct calls)
    return {
        "researcher_agent": researcher_agent,
        "orchestrator_agent": orchestrator_agent,
        "logger_agent": logger_agent,
        "logger_plugin": logger_plugin,
        "search_plugin": search_plugin,
        "scrape_plugin": scrape_plugin,
    }


# =========================
# Interaction helpers
# =========================
async def get_response(agent: ChatCompletionAgent, message: str) -> str:
    """Aggregate the full streamed response from the agent."""
    parts: List[str] = []
    try:
        async for resp in agent.invoke(message):
            if resp and getattr(resp, "content", None):
                parts.append(str(resp.content))
    except Exception as e:
        return f"[Error generating response: {e}]"
    return "".join(parts).strip()


def _is_yes_no(answer: str) -> Optional[bool]:
    if not answer:
        return None
    m = re.fullmatch(r"\s*(yes|no)\s*", answer.strip().lower())
    if not m:
        return None
    return m.group(1) == "yes"


def _maybe_store_name(umsg: str, chat_memory: ChatMemory) -> None:
    m = re.search(r"^\s*my\s+name\s+is\s+([A-Za-z' -]{2,})\s*$", umsg, re.I)
    if m:
        name = m.group(1).strip().title()
        chat_memory.facts["name"] = name


def _maybe_store_location(umsg: str, chat_memory: ChatMemory) -> None:
    m = re.search(r"\b(i\s+(live|am)\s+in)\s+([A-Za-z' -]{2,})\b", umsg, re.I)
    if m:
        city = m.group(3).strip().title()
        chat_memory.facts["city"] = city


async def decide_research(orchestrator_agent: ChatCompletionAgent, memory_context: str, user_message: str) -> Dict[str, Any]:
    """Ask the orchestrator to decide if web research is needed (machine-readable JSON)."""
    schema = """{
      "needs_research": true,
      "reason": "short string",
      "category": "events|news|price|law|howto|code|general",
      "confidence_local": 0.0,
      "queries": ["string"],
      "geoscope": "string (e.g., NYC, USA)",
      "ask_user_confirmation": false
    }"""

    prompt = (
        "Decide if you can answer confidently without fetching external data.\n"
        "Criteria for NEEDING research: volatile facts/current events, schedules, prices/quotes, laws/rules, availability, local/'near me', news/releases.\n"
        "If the question is tutorial/explanatory about stable knowledge (programming, math, theory), DO NOT research.\n"
        "Return ONLY valid JSON matching this schema, no extra text:\n"
        + schema
        + "\nMemory context and user message below.\n\n"
        + memory_context
        + "\n\nUser message:\n"
        + user_message
    )

    raw = await get_response(orchestrator_agent, prompt)

    # Extract JSON robustly (remove code fences, etc.)
    text = raw.strip()
    # Strip ```json fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    # Find the first {...} block if needed
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except Exception:
        # Conservative fallback
        data = {
            "needs_research": False,
            "reason": "fallback-parse",
            "category": "general",
            "confidence_local": 0.6,
            "queries": [],
            "geoscope": "",
            "ask_user_confirmation": False,
            "raw": raw,
        }
    return data


async def safe_log(
    logger_agent: ChatCompletionAgent,
    *,
    user_message: str,
    orchestrator_response: str,
    researcher_called: bool,
    researcher_text: str,
    final_text: str,
) -> None:
    log_prompt = (
        "Log this interaction to HTML (use the LogToHTML plugin):\n\n"
        f"User Question: {user_message}\n"
        f"Initial Orchestrator Response: {orchestrator_response}\n"
        f"Researcher Called: {'YES' if researcher_called else 'NO'}\n"
        f"Researcher Response: {researcher_text}\n"
        f"Final Orchestrator Response: {final_text}\n"
    )
    try:
        async for log_resp in logger_agent.invoke(log_prompt):
            if log_resp and getattr(log_resp, "content", None):
                log_note(f"📝 Logger: {log_resp.content}")
                break
    except Exception as e:
        log_warning(f"Failed to log interaction: {e}")


# =========================
# Export & Reset
# =========================
async def export_and_reset(chat_memory: ChatMemory, logger_plugin: LogToHTML) -> None:
    # 1) Export memory JSON
    mem_json_path = f"memory_{chat_memory.session_id}.json"
    with open(mem_json_path, "w", encoding="utf-8") as f:
        f.write(chat_memory.to_json())
    log_success(f"🧠 Memory exported to {mem_json_path}")

    # 2) Export memory pretty text (optional)
    mem_txt_path = f"memory_{chat_memory.session_id}.txt"
    lines = ["CHAT MEMORY", f"session_id: {chat_memory.session_id}", ""]
    if chat_memory.facts:
        lines.append("[FACTS]")
        for k, v in chat_memory.facts.items():
            lines.append(f"- {k}: {v}")
    if chat_memory.notes:
        lines.append("\n[NOTES]")
        for n in chat_memory.notes:
            lines.append(f"- {n}")
    with open(mem_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log_success(f"🧾 Memory exported to {mem_txt_path}")

    # 3) Close HTML log neatly
    try:
        logger_plugin.close_file()
        log_note("📁 HTML log closed.")
    except Exception as e:
        log_warning(f"Could not close HTML log: {e}")

    # 4) Reset (in-memory)
    chat_memory.facts.clear()
    chat_memory.notes.clear()


# =========================
# Main loop (CLI)
# =========================
MAX_RESEARCHER_ATTEMPTS = 3  # Max tries with different queries


async def orchestrator_flow() -> None:
    # Per-chat memory
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + str(uuid.uuid4())[:8]
    chat_memory = ChatMemory(session_id=session_id)

    # Tasteful defaults (seed)
    chat_memory.facts.setdefault("language", "en-US")
    chat_memory.facts.setdefault("units", "SI")
    chat_memory.facts.setdefault("style", "concise-structured")

    memory_plugin = MemoryPlugin(chat_memory)

    # Build agents WITH memory plugin (and exposed plugins)
    agents = build_agents(memory_plugin=memory_plugin)
    researcher_agent: ChatCompletionAgent = agents["researcher_agent"]
    orchestrator_agent: ChatCompletionAgent = agents["orchestrator_agent"]
    logger_agent: ChatCompletionAgent = agents["logger_agent"]
    logger_plugin: LogToHTML = agents["logger_plugin"]

    # Direct-access plugins (bulletproof)
    search_plugin: SearchOnline = agents["search_plugin"]
    scrape_plugin: ScrapeURL = agents["scrape_plugin"]

    log_success(f"🚀 Assistant started (session {session_id}). Type your question (or 'exit' to quit).")

    while True:
        try:
            user_message = await aioconsole.ainput("🧑‍💬 You: ")
        except (EOFError, KeyboardInterrupt):
            log_info("👋 Shutting down...")
            await export_and_reset(chat_memory, logger_plugin)
            break

        if user_message.strip().lower() in {"exit", "quit", "end"}:
            await export_and_reset(chat_memory, logger_plugin)
            log_info("👋 Bye!")
            break

        # Capture name and location early
        _maybe_store_location(user_message, chat_memory)
        _maybe_store_name(user_message, chat_memory)

        log_rule("New Interaction")

        # Memory context injected on every turn
        memory_context = (
            "Memory context (for this chat):\n"
            f"{chat_memory.to_json()}\n\n"
            "If helpful, use it. If you discover new persistent facts (preferences, goals, constraints), "
            "call the memory plugin to store them."
        )

        # 0) Orchestrator decision (auto-routing)
        decision = await decide_research(orchestrator_agent, memory_context, user_message)
        need_research = bool(decision.get("needs_research"))
        confidence = float(decision.get("confidence_local", 0.5))
        queries_from_llm = decision.get("queries") or []
        geo = (decision.get("geoscope") or chat_memory.facts.get("city") or "United States")
        decision_pretty = json.dumps(decision, ensure_ascii=False, indent=2)
        log_section("🧭 Orchestrator Decision", f"[code]{decision_pretty}[/code]", markdown=False)
        orchestrator_initial = decision_pretty  # for logging

        # Optional confirmation if confidence is medium
        if 0.4 <= confidence <= 0.6 and decision.get("ask_user_confirmation", False):
            try:
                yn = await aioconsole.ainput("🔎 Should I search the web? (yes/no): ")
            except (EOFError, KeyboardInterrupt):
                yn = "no"
            yn_bool = _is_yes_no(yn)
            if yn_bool is not None:
                need_research = yn_bool

        researcher_called = False
        researcher_text = ""
        final_text = ""

        if not need_research:
            # Local response (no researcher)
            final_text = await get_response(
                orchestrator_agent,
                f"{memory_context}\n\nUser:\n{user_message}"
            )
            log_section("✅ Orchestrator (Final Answer)", final_text)
        else:
            # Build queries with geo/language context
            language = chat_memory.facts.get("language", "en-US")

            def _rew(q: str) -> str:
                return _rewrite_query_for_web(q, locale_hint=geo, lang_hint=language)

            if queries_from_llm:
                queries = [_rew(q) for q in queries_from_llm]
            else:
                brief = build_research_brief(user_message, chat_memory)
                queries = [_rew(q) for q in brief["queries"]]

            log_section("🔎 Built Query(ies)", "\n".join(f"- {q}" for q in queries))

            # Direct plugin calls + manual concatenation
            researcher_called = True
            urls_pattern = re.compile(r"https?://[^\s)]+", re.I)

            for idx, q in enumerate(queries[:MAX_RESEARCHER_ATTEMPTS], start=1):
                log_info(f"📡 Calling Researcher (query {idx})...")

                # 1) SearchOnline (always first)
                rr_search = await search_plugin.search_online(q)
                researcher_text += f"\n[SearchOnline — Query {idx}]\n{rr_search}\n"
                log_section("🔬 Researcher — SearchOnline", rr_search)

                # Stop condition: found something useful
                if rr_search and "No results were found" not in rr_search and "could not retrieve" not in rr_search.lower():
                    # 2) (Optional) Scrape 1–2 relevant URLs detected in the text
                    urls = urls_pattern.findall(rr_search)
                    filtered_urls = [u for u in urls if len(u) > 20 and not re.search(r"news\.google|/home", u)]
                    for u in filtered_urls[:2]:
                        rr_scrape = await scrape_plugin.scrape_url(u)
                        researcher_text += f"\n[ScrapeURL] {u}\n{rr_scrape}\n"
                        log_section("🔬 Researcher — ScrapeURL", f"URL: {u}\n\n{rr_scrape[:1200]}…")
                    break

            # Final response based on researcher findings
            combined_message = (
                f"{memory_context}\n\n"
                f"User question: {user_message}\n\n"
                f"Base your answer ONLY on what the researcher found (you may structure and clarify):\n"
                f"{researcher_text}"
            )
            final_text = await get_response(orchestrator_agent, combined_message)
            log_section("✅ Orchestrator (Final Answer)", final_text)

        # 4) Auto memory extraction (silent)
        memory_extraction_prompt = (
            "Analyze the conversation above and, if there are persistent facts useful for this chat "
            "(preferences, goals, constraints, stable user data), store them using calls to the memory plugin.\n"
            "Examples:\n"
            "- remember_fact('language', 'en-US')\n"
            "- remember_fact('topic', 'personal finance')\n"
            "- remember_note('user prefers bullet points')\n"
            "If there is nothing to store, do nothing."
        )
        _ = await get_response(orchestrator_agent, memory_extraction_prompt)

        # 5) Log
        await safe_log(
            logger_agent,
            user_message=user_message,
            orchestrator_response=orchestrator_initial,
            researcher_called=researcher_called,
            researcher_text=researcher_text,
            final_text=final_text,
        )


def main():
    try:
        asyncio.run(orchestrator_flow())
    except KeyboardInterrupt:
        log_info("👋 Shutting down...")


if __name__ == "__main__":
    main()
