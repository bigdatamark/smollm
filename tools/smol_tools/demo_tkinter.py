# [MODULE DOC] macOS Tkinter desktop demo ("Smol Tools") that runs at import: creates a hidden Tk root with
# Segoe UI
# fonts and a TextPopupApp, then enters the Tk main loop. Global hotkeys (pynput): F1 draft-reply rewriter,
# F2 summarize the currently selected text (copied via Cmd+C), F5 chat window with saved history, F10 tool
# agent. Uses tkmacosx buttons and pyperclip; model generation runs on daemon threads and UI updates are
# scheduled with root.after.
import tkinter as tk
from tkmacosx import Button
import time
import threading
import queue
from pynput import keyboard
from smol_tools.summarizer import SmolSummarizer
from smol_tools.rewriter import SmolRewriter
from pynput.keyboard import Key, Controller
import pyperclip
from smol_tools.agent import SmolToolAgent
from smol_tools.chatter import SmolChatter
from smol_tools.titler import SmolTitler
import os
import getpass

class TextPopupApp:
    # [CLASS DOC] Main application object. Holds the Tk root, the list of open popups, the system username,
    # and instances of
    # SmolSummarizer, SmolRewriter, SmolTitler, SmolToolAgent and SmolChatter (all sharing one cached model).
    def __init__(self, root):
        # [FUNCTION DOC] Hides the root window, initializes state (last_text, active_popups, last_summary),
        # constructs all tools
        # (loading/warming up the model; slow), creates a keyboard Controller, registers and starts
        # GlobalHotKeys
        # for F1/F2/F5/F10, and stores getpass.getuser() as username.
        # KNOWN ISSUE: hotkey callbacks run on the pynput listener thread and create Tk widgets directly,
        # which is
        # not thread-safe in Tkinter.
        self.root = root
        self.root.withdraw()  # Start with the window hidden
        self.last_text = ""
        self.active_popups = []
        self.last_summary = ""
        
        # Initialize tools
        self.summarizer = SmolSummarizer()
        self.rewriter = SmolRewriter()
        self.titler = SmolTitler()
        self.agent = SmolToolAgent()
        self.chatter = SmolChatter()
        
        self.keyboard_controller = Controller()
        
        # Replace the keyboard listener with GlobalHotKeys
        self.keyboard_listener = keyboard.GlobalHotKeys({
            "<F1>": lambda: self.show_draft_input(
                self.root.winfo_x(),
                self.root.winfo_y(),
                self.root.winfo_width()
            ),
            "<F2>": self.generate_summary_from_selected_text,
            "<F5>": self.show_chat_window,
            "<F10>": self.show_agent_input,
        })
        self.keyboard_listener.start()
        
        self.username = getpass.getuser()  # Get system username

    def generate_summary_from_selected_text(self):
        # [FUNCTION DOC] F2 handler: grabs the selected text through the clipboard and, if non-empty, opens
        # the summary window.
        selected_text = self.get_selected_text()
        if selected_text:
            # Directly generate summary instead of showing confirmation popup
            self.generate_summary_direct(selected_text)

    # New method to directly show summary window
    def generate_summary_direct(self, text):
        # [FUNCTION DOC] Opens a 600x600 "Summary Chat" popup centred on the mouse (clamped to screen): a
        # read-only chat display
        # with name tags, a text input and "Ask Question" button (Enter also submits) wired to
        # process_summary_question. Shows the user's request with a 100-char preview, then starts a daemon
        # thread
        # running summarize on the text.
        summary_popup = tk.Toplevel(self.root)
        summary_popup.withdraw()  # Hide the window initially
        self.active_popups.append(summary_popup)
        summary_popup.title("Summary Chat")
        summary_popup.configure(bg='#f6f8fa')
        
        # Set minimum window size
        summary_popup.minsize(600, 600)
        
        # Main container
        container = tk.Frame(summary_popup, bg='#f6f8fa')
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Chat display area (top)
        chat_frame = tk.Frame(
            container,
            bg='white',
            highlightbackground='#e1e4e8',
            highlightthickness=1,
            bd=0,
            relief=tk.FLAT
        )
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Add chat display with scrollbar
        chat_display = tk.Text(
            chat_frame, 
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=0,
            bg='white',
            fg='#24292e',
            font=('Segoe UI', 12),
            padx=15,
            pady=12
        )
        scrollbar = tk.Scrollbar(chat_frame, command=chat_display.yview)
        chat_display.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure text tags
        chat_display.tag_configure("assistant_name", foreground="#E57373", font=('Segoe UI', 12, 'bold'))
        chat_display.tag_configure("user_name", foreground="#7986CB", font=('Segoe UI', 12, 'bold'))
        chat_display.config(state='disabled')
        
        # Input area (bottom)
        input_container = tk.Frame(
            container,
            bg='white',
            highlightbackground='#e1e4e8',
            highlightthickness=1,
            bd=0,
            relief=tk.FLAT
        )
        input_container.pack(fill=tk.X)
        
        # Text input
        chat_input = tk.Text(
            input_container, 
            height=3, 
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=0,
            bg='white',
            fg='#24292e',
            font=('Segoe UI', 12),
            padx=15,
            pady=12,
            insertwidth=2,  # Width of cursor
            insertbackground='#0066FF',  # Color of cursor matching our theme
            insertofftime=500,  # Cursor blink off time in milliseconds
            insertontime=500   # Cursor blink on time in milliseconds
        )
        chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Send button
        send_btn = Button(
            input_container,
            text="Ask Question",
            command=lambda: self.process_summary_question(
                text, chat_input.get("1.0", "end-1c").strip(),
                chat_display, chat_input
            ),
            font=('Segoe UI', 12),
            bg='#0066FF',
            fg='white',
            activebackground='#0052CC',
            activeforeground='white',
            borderless=True,
            focuscolor='',
            height=32,
            padx=15
        )
        send_btn.pack(side=tk.RIGHT, padx=15, pady=8)
        
        # Bind Enter key
        chat_input.bind("<Return>", lambda e: [
            self.process_summary_question(
                text, chat_input.get("1.0", "end-1c").strip(),
                chat_display, chat_input
            ),
            "break"
        ][1])
        
        # Display initial summary request
        preview = text[:100] + "..." if len(text) > 100 else text
        self.update_summary_chat(chat_display, self.username, f"Please summarize this text: {preview}")
        
        # Position window
        summary_popup.update_idletasks()
        popup_width = 600
        popup_height = 600
        
        # Get mouse position and screen dimensions
        mouse_x = self.root.winfo_pointerx()
        mouse_y = self.root.winfo_pointery()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate position
        x = min(max(mouse_x - popup_width//2, 0), screen_width - popup_width)
        y = min(max(mouse_y - popup_height//2, 0), screen_height - popup_height)
        
        summary_popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
        summary_popup.deiconify()
        
        def summarize(input_text):
            # [FUNCTION DOC] Thread body: adds an empty assistant line, then streams summarizer.process
            # output, appending only the newly
            # added suffix of the cumulative text to the display via root.after. Exceptions are printed.
            try:
                # First message from the model
                self.root.after(0, lambda: self.update_summary_chat(
                    chat_display, self.summarizer.name, ""))
                
                current_response = ""
                for output in self.summarizer.process(input_text):
                    # Only send the new part of the response
                    if output.startswith(current_response):
                        new_text = output[len(current_response):]
                        if new_text:  # Only update if there's new text
                            current_response = output
                            self.root.after(0, lambda t=new_text: chat_display.config(state='normal') or 
                                chat_display.insert("end-1c", t) or 
                                chat_display.config(state='disabled'))
            except Exception as e:
                print(e)
        
        threading.Thread(target=lambda: summarize(text), daemon=True).start()

    def update_summary_chat(self, chat_display: tk.Text, sender: str, message: str):
        # [FUNCTION DOC] Appends a new line "<sender>: <message>" to a summary chat display, colouring the
        # sender as assistant if it
        # equals the summarizer name, else as user; scrolls to the end and keeps the widget read-only.
        """Update the summary chat display with new message"""
        chat_display.config(state='normal')
        
        # Add the message with appropriate styling
        chat_display.insert(tk.END, "\n")  # Add spacing
        chat_display.insert(tk.END, sender, 
                        "assistant_name" if sender == self.summarizer.name else "user_name")
        chat_display.insert(tk.END, f": {message}")
        
        chat_display.see(tk.END)
        chat_display.config(state='disabled')

    def process_summary_question(self, original_text: str, question: str, 
                               chat_display: tk.Text, chat_input: tk.Text):
        # [FUNCTION DOC] Ignores blank questions; otherwise clears the input, shows the question, and starts
        # a daemon thread
        # (process_question) that answers it about original_text.
        """Process a follow-up question about the summarized text"""
        if not question.strip():
            return
            
        # Clear input
        chat_input.delete("1.0", tk.END)
        
        # Display user question
        self.update_summary_chat(chat_display, self.username, question)
        
        def process_question():
            # [FUNCTION DOC] Thread body: like summarize, but streams summarizer.process(original_text,
            # question=question).
            try:
                # First message from the model
                self.root.after(0, lambda: self.update_summary_chat(
                    chat_display, self.summarizer.name, ""))

                current_response = ""
                for output in self.summarizer.process(original_text, question=question):
                    # Only send the new part of the response
                    if output.startswith(current_response):
                        new_text = output[len(current_response):]
                        if new_text:  # Only update if there's new text
                            current_response = output
                            self.root.after(0, lambda t=new_text: chat_display.config(state='normal') or 
                                chat_display.insert("end-1c", t) or 
                                chat_display.config(state='disabled'))
            except Exception as e:
                print(e)

        threading.Thread(target=process_question, daemon=True).start()

    def get_selected_text(self):
        # [FUNCTION DOC] HELPER: simulates Cmd+C with pynput, waits 0.1 s and returns the clipboard contents
        # (overwrites the user's
        # clipboard; macOS-specific).
        # Copy selected text to clipboard
        with self.keyboard_controller.pressed(Key.cmd):
            self.keyboard_controller.tap('c')
        
        # Small delay to ensure clipboard is updated
        time.sleep(0.1)
        
        # Get text from clipboard
        return pyperclip.paste()

    def destroy_active_popups(self):
        # [FUNCTION DOC] Destroys every tracked popup (ignoring errors for already-closed ones) and clears
        # the list. Unused here.
        # Destroy all active popups
        for popup in self.active_popups:
            try:
                popup.destroy()
            except:
                pass  # Popup might already be destroyed
        self.active_popups = []

    def show_draft_input(self, summary_x, summary_y, summary_width):
        # [FUNCTION DOC] F1 handler: opens a "Draft Reply" popup with two columns: "Your Reply" input with a
        # "Smol Improvement?"
        # button (calls generate_improved_text) and a read-only "Improved Reply" with a "Copy" button
        # (copies to the
        # clipboard). Widget height is derived from last_summary line count (15-40). Positions the popup
        # centred
        # horizontally on summary_x + summary_width/2 and above or below summary_y depending on screen half.
        # KNOWN ISSUE: last_summary is never updated anywhere, so the height is always the minimum.
        draft_popup = tk.Toplevel(self.root)
        draft_popup.withdraw()  # Hide initially
        self.active_popups.append(draft_popup)
        draft_popup.title("Draft Reply")
        draft_popup.configure(bg='#f6f8fa')
        
        # Create frame for the two columns
        columns_frame = tk.Frame(draft_popup, bg='#f6f8fa')
        columns_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Calculate required height based on summary content
        num_lines = len(self.last_summary.split('\n'))
        line_height = max(num_lines * 1.5, 15)
        widget_height = min(line_height, 40) 
        
        # Column 1: Draft Input
        input_frame = tk.Frame(
            columns_frame,
            bg='white',
            highlightbackground='#e1e4e8',
            highlightthickness=1,
            bd=0,
            relief=tk.FLAT
        )
        input_frame.pack(side=tk.LEFT, padx=5, fill='both', expand=True)
        tk.Label(input_frame, text="Your Reply", bg='white', fg='#24292e', font=('Segoe UI', 12)).pack(padx=15, pady=(12,0), anchor='w')
        text_input = tk.Text(
            input_frame, 
            height=widget_height, 
            width=30, 
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=0,
            selectbackground='#e1e4e8',
            selectforeground='#24292e',
            insertwidth=2,  # Width of cursor
            insertbackground='#0066FF',  # Color of cursor matching our theme
            insertofftime=500,  # Cursor blink off time in milliseconds
            insertontime=500   # Cursor blink on time in milliseconds

        )
        text_input.pack(fill='both', expand=True, padx=15, pady=12)
        text_input.config(bg='white', fg='#24292e', font=('Segoe UI', 12))
        
        # Column 2: Improved Text
        improved_frame = tk.Frame(
            columns_frame,
            bg='white',
            highlightbackground='#e1e4e8',
            highlightthickness=1,
            bd=0,
            relief=tk.FLAT
        )
        improved_frame.pack(side=tk.LEFT, padx=5, fill='both', expand=True)
        tk.Label(improved_frame, text="Improved Reply", bg='white', fg='#24292e', font=('Segoe UI', 12)).pack(padx=15, pady=(12,0), anchor='w')
        improved_text = tk.Text(
            improved_frame, 
            height=widget_height, 
            width=30, 
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=0,
            selectbackground='#e1e4e8',
            selectforeground='#24292e'
        )
        improved_text.pack(fill='both', expand=True, padx=15, pady=12)
        improved_text.config(state='disabled', bg='white', fg='#586069', font=('Segoe UI', 12))
        
        # Add Copy button using tkmacosx
        copy_btn = Button(
            improved_frame, 
            text="Copy",
            command=lambda: pyperclip.copy(improved_text.get("1.0", "end-1c")),
            font=('Segoe UI', 14),
            bg='#0066FF',
            fg='white',
            activebackground='#0052CC',
            activeforeground='white',
            borderless=True,
            focuscolor='',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        copy_btn.pack(pady=(0, 12))
        
        # Add "Smol Improvement?" button using tkmacosx
        improve_btn = Button(
            input_frame, 
            text="Smol Improvement?",
            command=lambda: self.generate_improved_text(
                text_input.get("1.0", "end-1c"),
                improved_text),
            font=('Segoe UI', 14),
            bg='#0066FF',
            fg='white',
            activebackground='#0052CC',
            activeforeground='white',
            borderless=True,
            focuscolor='',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        improve_btn.pack(pady=(0, 12))
        
        # Position window relative to the summary window's position
        draft_popup.update_idletasks()
        screen_width = draft_popup.winfo_screenwidth()
        screen_height = draft_popup.winfo_screenheight()
        popup_width = draft_popup.winfo_width()
        if popup_width == 1:
            popup_width = 800
        popup_height = draft_popup.winfo_height()
        
        # Calculate center point of the summary window
        summary_center_x = summary_x + summary_width//2
        
        # Center the new window on the same point, but ensure it stays on screen
        new_x = max(min(summary_center_x - popup_width//2, screen_width - popup_width), 0)
        # Position vertically based on screen space available
        if summary_y > screen_height / 2:
            new_y = max(summary_y - popup_height - 10, 0)  # 10px gap above summary
        else:
            new_y = min(summary_y + 10, screen_height - popup_height)  # 10px gap below summary
        
        draft_popup.geometry(f"+{new_x}+{new_y}")
        draft_popup.deiconify()

    def generate_improved_text(self, text, improved_text_widget):
        # [FUNCTION DOC] Looks up the improve button through Tk's auto-generated child names, disables it
        # ("Generating..."), shows a
        # placeholder in the output widget and starts a daemon thread running improve on text.
        # KNOWN ISSUE: the button lookup depends on internal widget names ('!frame2', '!button') and is
        # fragile;
        # after generation the button text is restored to "Copy" instead of "Smol Improvement?".
        # Get reference to the improve button
        improve_btn = improved_text_widget.master.master.children['!frame2'].children['!button']
        
        # Disable the button and change text to show processing
        improve_btn.configure(
            state='disabled',
            text="Generating...",
            bg='#A8A8A8',  # Grayed out color
        )
        
        # Update the improve function
        improved_text_widget.config(state='normal')
        improved_text_widget.delete("1.0", tk.END)
        improved_text_widget.insert("1.0", "Generating improvement...")
        improved_text_widget.config(state='disabled')
        
        def improve(input_text):
            # [FUNCTION DOC] Thread body: streams rewriter.process output, replacing the widget text with
            # each cumulative output via
            # root.after; re-enables the button at the end or on error (then re-raises the exception in the
            # thread).
            try:
                for output in self.rewriter.process(input_text):
                    self.root.after(0, lambda t=output: self.update_improved_text(improved_text_widget, t))
                
                # Re-enable button and restore original state after generation is complete
                self.root.after(0, lambda: improve_btn.config(
                    state='normal',
                    text="Copy",
                    bg='#0066FF'
                ))
            except Exception as e:
                # Make sure to re-enable button even if there's an error
                self.root.after(0, lambda: improve_btn.config(
                    state='normal',
                    text="Copy",
                    bg='#0066FF'
                ))
                raise e
        
        threading.Thread(target=lambda: improve(text), daemon=True).start()

    def update_improved_text(self, text_widget, new_text):
        # [FUNCTION DOC] Replaces a read-only Text widget's content with new_text.
        text_widget.config(state='normal')
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", new_text)
        text_widget.config(state='disabled')

    def show_agent_input(self):
        # [FUNCTION DOC] F10 handler: opens a centred "SmolAgent" popup with a request input, read-only
        # response area and Submit
        # button that runs process_agent_request.
        # Create new popup for agent input
        agent_popup = tk.Toplevel(self.root)
        self.active_popups.append(agent_popup)
        agent_popup.title("SmolAgent")
        
        # Create input area
        input_frame = tk.Frame(agent_popup)
        input_frame.pack(padx=10, pady=5, fill='both', expand=True)
        
        tk.Label(input_frame, text="What would you like me to do?").pack()
        text_input = tk.Text(input_frame, height=4, width=50, wrap=tk.WORD)
        text_input.pack(pady=5)
        
        # Create output area
        output_frame = tk.Frame(agent_popup)
        output_frame.pack(padx=10, pady=5, fill='both', expand=True)
        
        tk.Label(output_frame, text="Response:").pack()
        output_text = tk.Text(output_frame, height=8, width=50, wrap=tk.WORD)
        output_text.pack(pady=5)
        output_text.config(state='disabled')
        
        def process_agent_request():
            # [FUNCTION DOC] Reads the query, shows "Processing request..." and starts a daemon thread
            # running run_agent.
            query = text_input.get("1.0", "end-1c")
            output_text.config(state='normal')
            output_text.delete("1.0", tk.END)
            output_text.insert("1.0", "Processing request...\n")
            output_text.config(state='disabled')
            
            def run_agent():
                # [FUNCTION DOC] Thread body: collects each agent.process output and displays all of them
                # joined by newlines.
                # Exceptions are not caught (the thread dies silently apart from the traceback).
                full_response = []
                for response in self.agent.process(query):
                    full_response.append(response)
                    self.root.after(0, lambda t="\n".join(full_response): self.update_agent_output(output_text, t))
            
            threading.Thread(target=run_agent, daemon=True).start()
        
        # Add Submit button
        submit_btn = tk.Button(agent_popup, text="Submit", command=process_agent_request)
        submit_btn.pack(pady=5)
        
        # Position window
        agent_popup.update_idletasks()
        screen_width = agent_popup.winfo_screenwidth()
        screen_height = agent_popup.winfo_screenheight()
        popup_width = agent_popup.winfo_width()
        popup_height = agent_popup.winfo_height()
        x = (screen_width - popup_width) // 2
        y = (screen_height - popup_height) // 2
        agent_popup.geometry(f"+{x}+{y}")

    def update_agent_output(self, text_widget, new_text):
        # [FUNCTION DOC] Replaces the agent output widget text with new_text (read-only afterwards).
        text_widget.config(state='normal')
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", new_text)
        text_widget.config(state='disabled')

    def show_chat_window(self):
        # [FUNCTION DOC] F5 handler: opens an 800x800 "SmolChat" window with a left history panel ("New
        # Chat" button and a listbox
        # of saved chat ids sorted by file modification time, newest first; selecting loads the chat) and a
        # main
        # area with the chat display, input and Send button (Enter also sends). Stores listbox and new-chat
        # button
        # in self.chat_controls.
        # KNOWN ISSUE: the saved-chat path "saved_chats" is hard-coded instead of using chatter.chats_dir.
        chat_window = tk.Toplevel(self.root)
        self.active_popups.append(chat_window)
        chat_window.title("SmolChat")
        
        # Configure the chat window to be resizable
        chat_window.geometry("800x800")
        chat_window.minsize(600, 600)
        
        # Create split view with history panel
        history_panel = tk.Frame(chat_window, width=200, padx=5, pady=10)
        history_panel.pack(side=tk.LEFT, fill=tk.Y)
        history_panel.pack_propagate(False)  # Maintain width
        
        # Create main chat area
        main_frame = tk.Frame(chat_window, padx=10, pady=10)
        main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create chat display first
        chat_display = tk.Text(main_frame, wrap=tk.WORD)
        chat_display.pack(fill=tk.BOTH, expand=True)
        chat_display.config(state='disabled')
        
        # Add input field and send button
        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        chat_input = tk.Text(input_frame, height=3, wrap=tk.WORD)
        chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        send_btn = tk.Button(input_frame, text="Send", 
                           command=lambda: self.process_chat_message(
                               chat_input.get("1.0", "end-1c").strip(),
                               chat_display))
        send_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Bind Enter key to send message
        chat_input.bind("<Return>", lambda e: [
            self.process_chat_message(
                chat_input.get("1.0", "end-1c").strip(),
                chat_display),
            "break"  # Prevent the default newline behavior
        ][1])
        
        # Now add the New Chat button (after chat_display is created)
        new_chat_btn = tk.Button(history_panel, text="New Chat", 
                               command=lambda: self.start_new_chat(chat_display))
        new_chat_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Add listbox for chat history
        history_label = tk.Label(history_panel, text="Previous Chats")
        history_label.pack()
        chat_listbox = tk.Listbox(history_panel, height=20)
        chat_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Get and sort chats by modification time (newest first)
        saved_chats = self.chatter.get_saved_chats()
        sorted_chats = sorted(
            saved_chats,
            key=lambda x: os.path.getmtime(os.path.join("saved_chats", f"chat_{x}.json")),
            reverse=True
        )
        
        # Populate chat history with sorted chats
        for chat_id in sorted_chats:
            chat_listbox.insert(tk.END, chat_id)
        
        # Bind selection event
        chat_listbox.bind('<<ListboxSelect>>', 
                         lambda e: self.load_selected_chat(chat_listbox, chat_display))
        
        # Add scrollbar to listbox
        history_scrollbar = tk.Scrollbar(history_panel, command=chat_listbox.yview)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        chat_listbox.config(yscrollcommand=history_scrollbar.set)
        
        # Store references to UI elements that need to be disabled during chat
        self.chat_controls = {
            'listbox': chat_listbox,
            'new_chat_btn': new_chat_btn
        }

        # Add text tags with softer colors
        chat_display.tag_configure("assistant_name", foreground="#E57373")  # Soft red
        chat_display.tag_configure("user_name", foreground="#7986CB")      # Soft blue

    def load_selected_chat(self, listbox: tk.Listbox, chat_display: tk.Text):
        # [FUNCTION DOC] Loads the chat selected in the listbox into the chatter and redraws the display.
        selection = listbox.curselection()
        if selection:
            chat_id = listbox.get(selection[0])
            self.chatter.load_chat(chat_id)
            self.display_chat_history(chat_display)

    def start_new_chat(self, chat_display):
        # [FUNCTION DOC] If the current chat has messages and was modified, saves it: under its existing id,
        # or (no id) under a
        # title generated by the titler from the "role: content" history (first 50 chars, slashes replaced
        # by "-").
        # Then starts a new chat, clears the display and refreshes the listbox (sorted by mtime).
        # KNOWN ISSUE: title generation runs synchronously on the UI thread, freezing the window while it
        # runs.
        # Only save the current chat if it has been modified since loading
        if self.chatter.has_current_chat() and self.chatter.is_chat_modified():
            # Get full chat history as a single string
            chat_history = "\n".join([f"{msg.role}: {msg.content}" 
                                    for msg in self.chatter.get_chat_history()])
            
            # If we're continuing an existing chat, use its ID
            current_chat_id = self.chatter.get_current_chat_id()
            if current_chat_id:
                # Save using existing ID
                self.chatter.save_current_chat(current_chat_id, overwrite=True)
            else:
                # Generate new title for new chat
                summary = ""
                for chunk in self.titler.process(chat_history):
                    summary = chunk
                
                summary_title = summary[:50].strip().replace("/", "-").replace("\\", "-")
                self.chatter.save_current_chat(summary_title, overwrite=True)
        
        # Start new chat
        self.chatter.start_new_chat()
        
        # Clear and update display
        self.display_chat_history(chat_display)
        
        # Update the chat history listbox with sorted chats
        listbox = self.chat_controls['listbox']
        listbox.delete(0, tk.END)
        saved_chats = self.chatter.get_saved_chats()
        sorted_chats = sorted(
            saved_chats,
            key=lambda x: os.path.getmtime(os.path.join("saved_chats", f"chat_{x}.json")),
            reverse=True
        )
        for chat_id in sorted_chats:
            listbox.insert(tk.END, chat_id)

    def process_chat_message(self, message: str, chat_display: tk.Text):
        # [FUNCTION DOC] Ignores blank messages. Disables history controls, writes "<username>: <message>"
        # and the assistant name
        # prefix, clears the input (found via Tk internal child names), resets current_response and starts a
        # daemon
        # thread running chat_response.
        if not message.strip():  # Skip empty messages
            return
            
        # Disable chat controls while processing
        self.chat_controls['listbox'].config(state='disabled')
        self.chat_controls['new_chat_btn'].config(state='disabled')
            
        chat_display.config(state='normal')
        
        # Add extra newline before user message for spacing
        chat_display.insert(tk.END, "")  # Start new line
        chat_display.insert(tk.END, self.username, "user_name")  # Add colored username
        chat_display.insert(tk.END, f": {message}\n")  # Add message
        
        chat_display.insert(tk.END, "\n")  # Add spacing
        chat_display.insert(tk.END, self.chatter.name, "assistant_name")  # Add colored AI name
        chat_display.insert(tk.END, ": ")  # Add separator
        
        # Clear the input field (get its reference from chat_display's master)
        input_frame = chat_display.master.children['!frame']
        chat_input = input_frame.children['!text']
        chat_input.delete("1.0", tk.END)
        
        # Initialize an empty string to store the full response
        self.current_response = ""
        
        chat_display.see(tk.END)
        chat_display.config(state='disabled')
        
        def chat_response():
            # [FUNCTION DOC] Thread body: streams chatter.process(message), appending only new suffix text
            # to the display, then two
            # newlines; always re-enables chat controls at the end (exceptions propagate in the thread).
            try:
                for chunk in self.chatter.process(message):
                    # Only send the new part of the response
                    if chunk.startswith(self.current_response):
                        new_text = chunk[len(self.current_response):]
                        if new_text:  # Only update if there's new text
                            self.current_response = chunk
                            self.root.after(0, lambda t=new_text: self.update_chat_display(chat_display, t))
                self.root.after(0, lambda t="\n\n": self.update_chat_display(chat_display, t))
            finally:
                # Re-enable chat controls after response is complete
                self.root.after(0, self.enable_chat_controls)
        
        threading.Thread(target=chat_response, daemon=True).start()

    def enable_chat_controls(self):
        # [FUNCTION DOC] Re-enables the history listbox and New Chat button.
        """Re-enable chat controls after response is complete"""
        self.chat_controls['listbox'].config(state='normal')
        self.chat_controls['new_chat_btn'].config(state='normal')

    def update_chat_display(self, chat_display: tk.Text, new_text: str):
        # [FUNCTION DOC] Appends new_text to the chat display, scrolls to the end and keeps it read-only.
        chat_display.config(state='normal')
        chat_display.insert(tk.END, new_text)
        chat_display.see(tk.END)
        chat_display.config(state='disabled')

    def display_chat_history(self, chat_display: tk.Text):
        # [FUNCTION DOC] Clears the chat display and redraws every message of the chatter history as
        # "<username or model name>:
        # <content>" with coloured name tags.
        chat_display.config(state='normal')
        chat_display.delete("1.0", tk.END)
        
        # Configure text tags with softer colors
        chat_display.tag_configure("assistant_name", foreground="#E57373")  # Soft red
        chat_display.tag_configure("user_name", foreground="#7986CB")      # Soft blue
        
        for message in self.chatter.get_chat_history():
            if message.role == "user":
                chat_display.insert(tk.END, "\n")  # Add spacing
                chat_display.insert(tk.END, self.username, "user_name")  # Change "You" to username
                chat_display.insert(tk.END, f": {message.content}\n")
            else:
                chat_display.insert(tk.END, "\n")  # Add spacing
                chat_display.insert(tk.END, self.chatter.name, "assistant_name")
                chat_display.insert(tk.END, f": {message.content}\n")
        
        chat_display.config(state='disabled')
        chat_display.see(tk.END)

# Run the app
root = tk.Tk()

# Set default font size for all tkinter widgets
default_font = ('Segoe UI', 14)  # Changed from TkDefaultFont to Segoe UI
root.option_add("*Font", default_font)
root.option_add("*Entry.Font", default_font)
root.option_add("*Text.Font", default_font)
root.option_add("*Button.Font", default_font)
root.option_add("*Label.Font", default_font)

app = TextPopupApp(root)
root.mainloop()
