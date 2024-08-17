import curses
from rag_llama3.RAG import RAG as rag
from rag_llama3.TextExtractor import TextExtractor as te
from rag_llama3.VectorDB import VectorDB as vdb
import sys
import threading
import time

class ChatWindow:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.messages = []
        self.stop_chat = False
    
    def run_chat(self):
        curses.wrapper(self._chat_loop)

    def _chat_loop(self, stdscr):
        curses.curs_set(1)
        stdscr.nodelay(1)
        stdscr.timeout(100)

        height, width = stdscr.getmaxyx()
        chat_win = curses.newwin(height - 3, width, 0, 0)
        input_win = curses.newwin(3, width, height - 3, 0)

        chat_win.scrollok(True)
        chat_win.idlok(True)

        input_win.addstr(1, 0, "Type your query (or 'quit' to exit): ")
        input_win.refresh()

        # Start threads for input processing and chat window updating
        input_thread = threading.Thread(target=self._process_input, args=(stdscr, input_win))
        chat_thread = threading.Thread(target=self._update_chat_window, args=(chat_win,))

        input_thread.start()
        chat_thread.start()

        input_thread.join()
        chat_thread.join()

    def _process_input(self, stdscr, input_win):
        while not self.stop_chat:
            ch = stdscr.getch()
            if ch == curses.KEY_BACKSPACE or ch == 127:
                input_win.addstr(1, len("Type your query (or 'quit' to exit): ") + len(input_win.instr(1, len("Type your query (or 'quit' to exit): ")).decode()), ' ')
                input_win.refresh()
            elif ch == 10:
                message = input_win.instr(1, len("Type your query (or 'quit' to exit): ")).decode().strip()
                if message.lower() == 'quit':
                    self.stop_chat = True
                    break
                if message:
                    self.messages.append(f"You: {message}")
                    response = self.rag_system.stream_answer(message)
                    self.messages.append(f"Bot: {response}")
                    input_win.clear()
                    input_win.addstr(1, 0, "Type your query (or 'quit' to exit): ")
                    input_win.refresh()
            elif ch != -1:
                input_win.addch(1, len("Type your query (or 'quit' to exit): ") + len(input_win.instr(1, len("Type your query (or 'quit' to exit): ")).decode()), ch)
                input_win.refresh()

    def _update_chat_window(self, chat_win):
        while not self.stop_chat:
            chat_win.clear()
            for i, msg in enumerate(self.messages):
                chat_win.addstr(i, 0, msg)
            chat_win.refresh()
            time.sleep(0.5)