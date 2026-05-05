import json
import os
import re
import ctypes
import subprocess
import tempfile
import time
import multiprocessing as mp
from datetime import datetime
import speech_recognition as sr
import requests

# -- Config --

IS_TTS_OFFLINE = False
WAKE_PHRASE = "harold"
STOP_WORDS = {"stop", "cancel", "shut up", "be quiet", "enough", "terminate"}

# Suppress ALSA warnings/errors from C library
_ERROR_HANDLER = ctypes.CFUNCTYPE(
	None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)

def _null_error_handler(*_):
	pass

_asound = ctypes.cdll.LoadLibrary("libasound.so.2")
_c_error_handler = _ERROR_HANDLER(_null_error_handler)
_asound.snd_lib_error_set_handler(_c_error_handler)


# ============================================================
# LISTENER PROCESS — owns the mic, always listening
# ============================================================

def listener_process(listener_q, active_flag):
	"""Listens for wake word (PocketSphinx offline) or commands (Google online)."""
	from pocketsphinx import LiveSpeech

	recognizer = sr.Recognizer()
	microphone = sr.Microphone()
	recognizer.pause_threshold = 0.8
	recognizer.phrase_threshold = 0.3
	recognizer.non_speaking_duration = 0.5

	while True:
		try:
			if not active_flag.is_set():
				for phrase in LiveSpeech(keyphrase=WAKE_PHRASE, kws_threshold=1e-20):
					listener_q.put(("wake", str(phrase)))
					break
			else:
				with microphone as source:
					recognizer.adjust_for_ambient_noise(source, duration=0.3)
					audio = recognizer.listen(source, timeout=None, phrase_time_limit=30)
				try:
					text = recognizer.recognize_google(audio)
					listener_q.put(("speech", text))
				except sr.UnknownValueError:
					pass
				except sr.RequestError as e:
					listener_q.put(("error", str(e)))
		except Exception as e:
			listener_q.put(("error", str(e)))
			time.sleep(1)


# ============================================================
# SPEAKER PROCESS — owns audio output, plays TTS
# ============================================================

def _clean_for_speech(text):
	text = re.sub(r"\*+", "", text)
	text = re.sub(r"_+", "", text)
	text = re.sub(r"#+\s*", "", text)
	text = re.sub(r"`+", "", text)
	text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
	return text


def speaker_process(speaker_q, speaking_flag):
	"""Reads from speaker_q and plays TTS. Supports cancel signals."""
	cache_dir = os.path.join(tempfile.gettempdir(), "va_tts_cache")
	os.makedirs(cache_dir, exist_ok=True)
	current_proc = None

	def play_audio(mp3_path):
		nonlocal current_proc
		current_proc = subprocess.Popen(
			["ffplay", "-nodisp", "-autoexit", mp3_path],
			stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
		)
		current_proc.wait()
		current_proc = None

	def tts_to_file(text):
		if IS_TTS_OFFLINE:
			import pyttsx3
			engine = pyttsx3.init()
			clean = _clean_for_speech(text)
			if not clean.strip():
				return None
			path = os.path.join(cache_dir, f"tts_{os.getpid()}_{time.monotonic_ns()}.mp3")
			engine.save_to_file(clean, path)
			engine.runAndWait()
			return path
		else:
			from gtts import gTTS
			clean = _clean_for_speech(text)
			if not clean.strip():
				return None
			path = os.path.join(cache_dir, f"tts_{os.getpid()}_{time.monotonic_ns()}.mp3")
			gTTS(text=clean, lang="en").save(path)
			return path

	ready_path = os.path.join(cache_dir, "ready_to_listen.mp3")
	if not os.path.exists(ready_path):
		from gtts import gTTS
		gTTS(text="Ready to listen.", lang="en").save(ready_path)

	while True:
		try:
			msg = speaker_q.get()
			if msg is None:
				break

			cmd, data = msg

			if cmd == "speak":
				speaking_flag.set()
				path = tts_to_file(data)
				if path:
					play_audio(path)
					try:
						os.unlink(path)
					except OSError:
						pass
				speaking_flag.clear()

			elif cmd == "ready":
				play_audio(ready_path)

			elif cmd == "cancel":
				if current_proc and current_proc.poll() is None:
					current_proc.terminate()
					current_proc = None
				while not speaker_q.empty():
					try:
						speaker_q.get_nowait()
					except Exception:
						break
				speaking_flag.clear()

		except Exception as e:
			print(f"  [speaker error: {e}]", flush=True)


# ============================================================
# TOOL DEFINITIONS & EXECUTION
# ============================================================

from ddgs import DDGS
from bs4 import BeautifulSoup

TOOLS = [
	{
		"type": "function",
		"function": {
			"name": "web_search",
			"description": "Search the web using DuckDuckGo. Returns top results with title, snippet, and URL.",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {"type": "string", "description": "The search query"}
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
					"url": {"type": "string", "description": "The URL of the page to fetch"}
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
			"parameters": {"type": "object", "properties": {}},
		},
	},
]


def execute_web_search(query):
	results = DDGS().text(query, max_results=50)
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

# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt():
	now = datetime.now().strftime("%A, %B %d, %Y — %H:%M:%S")
	return {
		"role": "system",
		"content": (
			f"You are a voice assistant named 'gio'. The user speaks to you through a microphone "
			f"and your responses are displayed as text and read aloud via TTS.\n\n"
			f"CURRENT DATE AND TIME: {now}\n"
			f"Always use this date when the user asks about 'current', 'today', 'now', 'recent', "
			f"or 'latest' information. When searching the web, include the correct year ({datetime.now().year}) "
			f"in your search queries. NEVER guess or use dates from your training data.\n\n"
			f"RESPONSE LENGTH: Keep all responses under 100 words by default. Be direct "
			f"and concise — no filler, no unnecessary detail. ONLY if the user says the "
			f"word 'detailed' in their request, give a thorough and unlimited-length "
			f"response. Otherwise, always be brief.\n\n"
			f"You have access to the following tools:\n"
			f"- web_search(query): Search the web using DuckDuckGo. Use this when the "
			f"user asks about current events, recent information, or anything you're "
			f"unsure about.\n"
			f"- fetch_page(url): Fetch the full text content of a web page. Use this "
			f"after a web search when the user needs detailed information from a "
			f"specific page.\n"
			f"- get_datetime(): Get the current date and time. Use this to double-check "
			f"the exact current time if precision matters.\n\n"
			f"Use tools when needed. Do not make up information you can look up. "
			f"Minimize tool rounds — prefer answering from search snippets rather than "
			f"fetching full pages unless the snippets are insufficient. "
			f"Use at most 2 tool rounds before giving your answer.\n\n"
			f"IMPORTANT: When you retrieve information using tools, you MUST include "
			f"the actual data and details in your response. Never say things like "
			f"'I've shared the information' or 'here are the results' without "
			f"actually stating the information. The user cannot see tool results — "
			f"they can only see your final response.\n\n"
			f"IMPORTANT: Always respond in English only. Never use Chinese characters "
			f"or any non-Latin script in your responses.\n\n"
			f"FORMATTING: Your responses are read aloud by a text-to-speech engine. "
			f"Never use markdown formatting — no bold (**), italics (_), headers (#), "
			f"bullet points (-/*), numbered lists, backticks, or any other markup. "
			f"Write in plain, natural sentences as if you were speaking out loud. "
			f"Use short paragraphs and conversational transitions instead of lists."
		),
	}

# ============================================================
# OLLAMA INTERACTION
# ============================================================


def stream_response(conversation_history, use_tools=True):
	"""Stream a response from Ollama. Returns (full_text, tool_calls_list)."""
	payload = {
		"model": "gemini-3-flash-preview",
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

		if "tool_calls" in msg:
			tool_calls.extend(msg["tool_calls"])

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


def drain_listener_queue(listener_q):
	"""Discard all stale events from the listener queue."""
	while not listener_q.empty():
		try:
			listener_q.get_nowait()
		except Exception:
			break


def ask_ollama(prompt, conversation_history, listener_q, speaker_q, speaking_flag):
	"""Run the Ollama tool loop. Checks listener_q for cancel commands."""
	if not conversation_history:
		conversation_history.append(build_system_prompt())
	conversation_history.append({"role": "user", "content": prompt})

	drain_listener_queue(listener_q)
	print("  [processing...]", flush=True)
	cancelled = False

	for round_num in range(10):
		if check_for_cancel(listener_q):
			print("  [cancelled by user]", flush=True)
			cancelled = True
			break

		t_llm = time.monotonic()
		content, tool_calls = stream_response(conversation_history)
		print(f"  [{time.monotonic() - t_llm:.1f}s llm round {round_num + 1}]", flush=True)

		if not tool_calls:
			break

		conversation_history.append({
			"role": "assistant",
			"content": content,
			"tool_calls": tool_calls,
		})
		for tc in tool_calls:
			fn_name = tc["function"]["name"]
			fn_args = tc["function"]["arguments"]
			t_tool = time.monotonic()
			try:
				result = TOOL_EXECUTORS[fn_name](fn_args)
			except Exception as e:
				result = f"Error: {e}"
			print(f"  [tool: {fn_name}({json.dumps(fn_args)}) {time.monotonic() - t_tool:.1f}s]", flush=True)
			conversation_history.append({"role": "tool", "content": str(result)})
	else:
		if not cancelled:
			t_llm = time.monotonic()
			content, _ = stream_response(conversation_history, use_tools=False)
			print(f"  [{time.monotonic() - t_llm:.1f}s llm final]", flush=True)

	if not cancelled:
		if not content:
			content = "Sorry, I wasn't able to come up with a response."
			print(f"\nAssistant: {content}")
		print()
		speaker_q.put(("speak", content))
		wait_for_speaker(listener_q, speaker_q, speaking_flag)
		conversation_history.append({"role": "assistant", "content": content})
	else:
		speaker_q.put(("cancel", None))
		conversation_history.append({"role": "assistant", "content": "(cancelled)"})


def wait_for_speaker(listener_q, speaker_q, speaking_flag):
	"""Wait for speaker to finish, checking for stop words."""
	while speaking_flag.is_set():
		try:
			event_type, text = listener_q.get(timeout=0.3)
		except Exception:
			continue

		if event_type == "speech" and text:
			words = text.lower().split()
			if any(w in words for w in STOP_WORDS):
				print("  [cancelled by user]", flush=True)
				speaker_q.put(("cancel", None))
				return
	drain_listener_queue(listener_q)


def check_for_cancel(listener_q):
	"""Non-blocking check if a stop word was heard."""
	while not listener_q.empty():
		try:
			event_type, text = listener_q.get_nowait()
			if event_type == "speech" and text:
				words = text.lower().split()
				if any(w in words for w in STOP_WORDS):
					return True
		except Exception:
			break
	return False


# ============================================================
# APP (MAIN PROCESS) — state machine
# ============================================================

# States
IDLE = "idle"
LISTENING = "listening"


def main():
	listener_q = mp.Queue()
	speaker_q = mp.Queue()
	speaking_flag = mp.Event()
	active_flag = mp.Event()

	listener_proc = mp.Process(target=listener_process, args=(listener_q, active_flag), daemon=True)
	speaker_proc = mp.Process(target=speaker_process, args=(speaker_q, speaking_flag), daemon=True)
	listener_proc.start()
	speaker_proc.start()

	tts_mode = "offline (pyttsx3)" if IS_TTS_OFFLINE else "online (gTTS)"
	print('Voice Assistant ready. Say "gio" to activate.')
	print(f"Tools: web_search, fetch_page, get_datetime | TTS: {tts_mode}")
	print('Say "stop" or "cancel" to interrupt a response.')
	print("Press Ctrl+C to quit.\n")

	state = IDLE
	conversation_history = []
	last_speech_time = 0
	LISTEN_TIMEOUT = 15

	try:
		while True:
			if state == IDLE:
				active_flag.clear()
				print("Waiting for wake word...", end="\r", flush=True)

			try:
				event_type, text = listener_q.get(timeout=1.0)
			except Exception:
				if state == LISTENING and (time.monotonic() - last_speech_time) > LISTEN_TIMEOUT:
					print("  [no speech — returning to idle]", flush=True)
					state = IDLE
				continue

			if event_type == "error":
				print(f"  [listener error: {text}]", flush=True)
				continue

			if event_type == "wake":
				print(f"Wake word detected! (heard: {text})")
				active_flag.set()
				drain_listener_queue(listener_q)
				speaker_q.put(("ready", None))
				state = LISTENING
				last_speech_time = time.monotonic()
				continue

			if event_type != "speech" or not text:
				continue

			last_speech_time = time.monotonic()
			words = text.lower().split()

			if state == LISTENING:
				if any(w in words for w in STOP_WORDS):
					continue

				print(f"You said: {text}")

				try:
					ask_ollama(text, conversation_history, listener_q, speaker_q, speaking_flag)
				except requests.ConnectionError:
					print("Could not connect to Ollama. Is it running? (ollama serve)")
				except requests.HTTPError as e:
					print(f"Ollama error: {e}")
				except Exception as e:
					print(f"Error: {e}")

				last_speech_time = time.monotonic()
				print("\nListening for follow-up...", flush=True)

	except KeyboardInterrupt:
		print("\nGoodbye!")
	finally:
		speaker_q.put(None)
		listener_proc.terminate()
		speaker_proc.join(timeout=3)


if __name__ == "__main__":
	main()
