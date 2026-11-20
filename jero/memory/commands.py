class MemoryCommandHandler:
    def __init__(self, memory):
        self.memory = memory

    def handle(self, message):
        text = message.strip()

        if not text.lower().startswith("remember that "):
            return None

        memory_text = text[len("remember that "):].strip()

        if not memory_text:
            return None

        key = f"memory_{self.memory_count() + 1}"

        self.memory.add_memory(
            key,
            memory_text,
        )

        return memory_text

    def memory_count(self):
        cursor = self.memory.connection.execute(
            "SELECT COUNT(*) FROM memories"
        )

        return cursor.fetchone()[0]