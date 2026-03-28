import json
import os
import re
import ctypes
import tempfile
import threading
from datetime import datetime
import speech_recognition as sr
import requests

IS_COMMAND_STAGED_BEFORE_EXECUTION = False
IS_TTS_OFFLINE = False  # True = pyttsx3/espeak, False = gTTS

# Suppress ALSA warnings/errors from C library
_ERROR_HANDLER = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)


def _null_error_handler(*_):
    pass


_asound = ctypes.cdll.LoadLibrary("libasound.so.2")
_c_error_handler = _ERROR_HANDLER(_null_error_handler)  # prevent garbage collection
_asound.snd_lib_error_set_handler(_c_error_handler)

from ddgs import DDGS
from bs4 import BeautifulSoup

# -- TTS setup --

if IS_TTS_OFFLINE:
    import pyttsx3

    _tts_engine = pyttsx3.init()
else:
    from gtts import gTTS


def _clean_for_speech(text):
    """Clean text for TTS: remove markdown formatting and non-Latin characters."""
    text = re.sub(r"\*+", "", text)  # strip * and **
    text = re.sub(r"_+", "", text)   # strip _ and __
    text = re.sub(r"#+\s*", "", text)  # strip markdown headers
    text = re.sub(r"`+", "", text)   # strip backticks
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # [link](url) -> link
    # text = re.sub(r"[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]+", " ", text)  # non-Latin
    return text


def _speak_blocking(text):
    if IS_TTS_OFFLINE:
        clean = _clean_for_speech(text)
        if not clean.strip():
            return
        _tts_engine.say(clean)
        _tts_engine.runAndWait()
    else:
        clean = _clean_for_speech(text)
        if not clean.strip():
            return
        tts = gTTS(text=clean, lang="en")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        tts.save(tmp_path)
        os.system(f'mpv --no-terminal "{tmp_path}" 2>/dev/null')
        os.unlink(tmp_path)


def speak(text, blocking=True):
    """Speak text. Set blocking=False to run in a background thread."""
    if not text:
        return
    if blocking:
        _speak_blocking(text)
    else:
        threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()


# -- Tool definitions for Ollama --

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns top results with title, snippet, and URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch the full text content of a web page given its URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the page to fetch",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# -- Tool execution --


def execute_web_search(query):
    results = DDGS().text(query, max_results=5)
    output = []
    for r in results:
        output.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}")
    return "\n\n".join(output) if output else "No results found."


def execute_fetch_page(url):
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text[:4000] if len(text) > 4000 else text


def execute_get_datetime():
    return datetime.now().strftime("%A, %B %d, %Y — %H:%M:%S")


TOOL_EXECUTORS = {
    "web_search": lambda args: execute_web_search(args["query"]),
    "fetch_page": lambda args: execute_fetch_page(args["url"]),
    "get_datetime": lambda _: execute_get_datetime(),
}

# -- Voice input --


WAKE_WORD = "archer"


def listen_for_wake_word(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=30)
        except sr.WaitTimeoutError:
            return False

    try:
        text = recognizer.recognize_google(audio).lower()
        if WAKE_WORD in text:
            return True
    except (sr.UnknownValueError, sr.RequestError):
        pass
    return False


def listen_for_command(recognizer, microphone):
    recognizer.pause_threshold = 3.0  # seconds of silence before stopping
    recognizer.phrase_threshold = 0.3
    with microphone as source:
        print("Listening... (Speak now)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=None, phrase_time_limit=60)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
        return None
    except sr.RequestError:
        print("System is down/No internet.")
        return None


SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a voice assistant named archer. The user speaks to you through a microphone "
        "and your responses are displayed as text and read aloud via TTS.\n\n"
        "RESPONSE LENGTH: Keep all responses under 100 words by default. Be direct "
        "and concise — no filler, no unnecessary detail. ONLY if the user says the "
        "word 'detailed' in their request, give a thorough and unlimited-length "
        "response. Otherwise, always be brief.\n\n"
        "You have access to the following tools:\n"
        "- web_search(query): Search the web using DuckDuckGo. Use this when the "
        "user asks about current events, recent information, or anything you're "
        "unsure about.\n"
        "- fetch_page(url): Fetch the full text content of a web page. Use this "
        "after a web search when the user needs detailed information from a "
        "specific page.\n"
        "- get_datetime(): Get the current date and time. Use this when the user "
        "asks what day, date, or time it is.\n\n"
        "When a query involves current events, real-time information, recent news, "
        "today's date, or anything where timeliness matters, ALWAYS call "
        "get_datetime() first to know the current date, then use web_search() "
        "with the current year/date to get up-to-date results. Never rely on "
        "your training data for time-sensitive questions.\n\n"
        "Use tools when needed. Do not make up information you can look up.\n\n"
        "IMPORTANT: When you retrieve information using tools, you MUST include "
        "the actual data and details in your response. Never say things like "
        "'I've shared the information' or 'here are the results' without "
        "actually stating the information. The user cannot see tool results — "
        "they can only see your final response.\n\n"
        "IMPORTANT: Always respond in English only. Never use Chinese characters "
        "or any non-Latin script in your responses."
    ),
}

# -- Ollama interaction with tool loop --


def ask_ollama(prompt, conversation_history):
    if not conversation_history:
        conversation_history.append(SYSTEM_PROMPT)
    conversation_history.append({"role": "user", "content": prompt})

    print("  [processing...]", flush=True)

    for round_num in range(5):  # max 5 tool rounds
        content, tool_calls = _stream_response(conversation_history)

        if not tool_calls:
            break

        # Execute each tool call
        conversation_history.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
        )
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            print(f"  [tool: {fn_name}({json.dumps(fn_args)})]", flush=True)

            try:
                result = TOOL_EXECUTORS[fn_name](fn_args)
            except Exception as e:
                result = f"Error: {e}"

            conversation_history.append(
                {
                    "role": "tool",
                    "content": str(result),
                }
            )
    else:
        # Exhausted tool rounds — do one final streaming response without tools
        content, _ = _stream_response(conversation_history, use_tools=False)

    print()
    speak(content, blocking=False)
    conversation_history.append({"role": "assistant", "content": content})


def _stream_response(conversation_history, use_tools=True):
    """Stream a response from Ollama, printing tokens as they arrive.
    Returns (full_text, tool_calls_list)."""
    payload = {
        "model": "glm-5:cloud",
        "messages": conversation_history,
        "stream": True,
    }
    if use_tools:
        payload["tools"] = TOOLS

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    full_content = []
    tool_calls = []
    started_printing = False

    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        msg = chunk.get("message", {})

        # Collect tool calls
        if "tool_calls" in msg:
            tool_calls.extend(msg["tool_calls"])

        # Stream text tokens
        token = msg.get("content", "")
        if token:
            if not started_printing:
                print("\nAssistant: ", end="", flush=True)
                started_printing = True
            print(token, end="", flush=True)
            full_content.append(token)

    if started_printing:
        print()

    return "".join(full_content), tool_calls


# -- UI --


def confirm_input(text):
    while True:
        print(f'\nYou said: "{text}"')
        print("[c]onfirm  [r]edo  [e]dit  [q]uit")
        choice = input("> ").strip().lower()

        if choice in ("c", "confirm"):
            return text
        elif choice in ("r", "redo"):
            return "REDO"
        elif choice in ("e", "edit"):
            edited = input("Edit your message: ").strip()
            if edited:
                return edited
            print("Empty input, try again.")
        elif choice in ("q", "quit", "exit"):
            return "QUIT"
        else:
            print("Invalid choice.")


def main():
    tts_mode = "offline (pyttsx3)" if IS_TTS_OFFLINE else "online (gTTS)"
    print('Voice Assistant ready. Say "archer" to activate.')
    print(f"Tools: web_search, fetch_page, get_datetime | TTS: {tts_mode}")
    print("Press Ctrl+C to quit.\n")

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    conversation_history = []

    try:
        while True:
            print("Waiting for wake word...", end="\r", flush=True)
            if not listen_for_wake_word(recognizer, microphone):
                continue

            print("Wake word detected!        ")
            speak("Ready to listen.", blocking=False)

            user_text = listen_for_command(recognizer, microphone)
            if user_text is None:
                continue

            if IS_COMMAND_STAGED_BEFORE_EXECUTION:
                result = confirm_input(user_text)

                if result == "QUIT":
                    print("Goodbye!")
                    raise KeyboardInterrupt
                if result == "REDO":
                    continue

                final_text = result
            else:
                final_text = user_text

            try:
                ask_ollama(final_text, conversation_history)
            except requests.ConnectionError:
                print("Could not connect to Ollama. Is it running? (ollama serve)")
            except requests.HTTPError as e:
                print(f"Ollama error: {e}")

            print()  # blank line before next wake word
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
