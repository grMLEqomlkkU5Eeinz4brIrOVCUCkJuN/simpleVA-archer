import sys
import json
import ctypes
from datetime import datetime
import speech_recognition as sr
import requests

# Suppress ALSA warnings/errors from C library
_ERROR_HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                  ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def _null_error_handler(*_):
	pass
_asound = ctypes.cdll.LoadLibrary("libasound.so.2")
_c_error_handler = _ERROR_HANDLER(_null_error_handler)  # prevent garbage collection
_asound.snd_lib_error_set_handler(_c_error_handler)
import pyttsx3
from ddgs import DDGS
from bs4 import BeautifulSoup

tts_engine = pyttsx3.init()


def speak(text):
	if text:
		tts_engine.say(text)
		tts_engine.runAndWait()

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


def listen_to_me():
	r = sr.Recognizer()

	with sr.Microphone() as source:
		print("Listening... (Speak now)")
		r.adjust_for_ambient_noise(source, duration=1)
		audio = r.listen(source)

	try:
		text = r.recognize_google(audio)
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
		"You are a voice assistant. The user speaks to you through a microphone "
		"and your responses are displayed as text. Keep your answers concise and "
		"conversational since the user is speaking, not typing.\n\n"
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
		"they can only see your final response."
	),
}

# -- Ollama interaction with tool loop --


def ask_ollama(prompt, conversation_history):
	if not conversation_history:
		conversation_history.append(SYSTEM_PROMPT)
	conversation_history.append({"role": "user", "content": prompt})

	for _ in range(5):  # max 5 tool rounds
		response = requests.post(
			"http://localhost:11434/api/chat",
			json={
				"model": "glm-5:cloud",
				"messages": conversation_history,
				"tools": TOOLS,
				"stream": False,
			},
		)
		response.raise_for_status()
		msg = response.json()["message"]

		tool_calls = msg.get("tool_calls")
		if not tool_calls:
			break

		# Execute each tool call
		conversation_history.append(msg)
		for tc in tool_calls:
			fn_name = tc["function"]["name"]
			fn_args = tc["function"]["arguments"]
			print(f'  [tool: {fn_name}({json.dumps(fn_args)})]', flush=True)

			try:
				result = TOOL_EXECUTORS[fn_name](fn_args)
			except Exception as e:
				result = f"Error: {e}"

			conversation_history.append({
				"role": "tool",
				"content": str(result),
			})
	else:
		# Exhausted tool rounds, use last response as-is
		conversation_history.append(msg)
		content = msg.get("content", "")
		print(f"\nAssistant: {content}\n")
		speak(content)
		return

	# Print the final response
	reply = msg.get("content", "")
	print(f"\nAssistant: {reply}\n")
	speak(reply)

	conversation_history.append({"role": "assistant", "content": reply})


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


def wait_for_space():
	print("Press [SPACE] to speak, [q] to quit...", flush=True)
	import termios
	import tty
	fd = sys.stdin.fileno()
	old_settings = termios.tcgetattr(fd)
	try:
		tty.setraw(fd)
		ch = sys.stdin.read(1)
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
	if ch == " ":
		return True
	if ch in ("q", "\x03"):  # q or Ctrl+C
		return False
	return None


def main():
	print("Voice Assistant ready.")
	print("Tools: web_search, fetch_page, get_datetime")
	conversation_history = []

	while True:
		action = wait_for_space()
		if action is False:
			print("Goodbye!")
			break
		if action is None:
			continue

		user_text = listen_to_me()
		if user_text is None:
			continue

		result = confirm_input(user_text)

		if result == "QUIT":
			print("Goodbye!")
			break
		if result == "REDO":
			continue

		try:
			ask_ollama(result, conversation_history)
		except requests.ConnectionError:
			print("Could not connect to Ollama. Is it running? (ollama serve)")
		except requests.HTTPError as e:
			print(f"Ollama error: {e}")


if __name__ == "__main__":
	main()
