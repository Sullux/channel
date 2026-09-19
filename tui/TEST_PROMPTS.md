# Test Prompts

These are just some informal test prompt ideas for easy copy/pasting into the TUI.

## Reading

---

If I were to ask you to read and summarize a large text file for me (> 10 GB, plain English, no JSON or other markup), how would you go about it? What low-level tools would you use and how would you use them?

Keep your response brief and to the point. If you have multiple steps, try not to convey too much detail on each step. If I need more information about a step, I am happy to ask follow up questions.

What do you think?

---

What specific Python libraries would you use for this?

---

Do you see how the simple approach of `less` with a `space` loop would work?

---

Ok. Think about this approach for a moment and tell me what you think:

1. You use `terminal_write` to send `less big-file.txt` to your terminal.
2. You use `terminal_read` to read the first page of text.
3. You use `terminal_key` to send a `space` key to proceed to the next page of text.
4. You use `terminal_read` again to read the next page.
5. You repeat until the page indicates the end of the file.
6. You use `terminal_key` to send a `q` key to exit less.

Think about these steps. Think about whether this process would meet the brief. Would this allow you to read the full contents of a long text file in such a way that your built-in memory would capture it for later? Would this work well with your limited context window since each terminal page of text would only contain a few hundred tokens? What do you think about this approach in general?
