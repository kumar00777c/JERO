from PySide6.QtCore import QObject, Signal, Slot


class GenerationWorker(QObject):
    chunk = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, brain, message, attachments=None):
        super().__init__()
        self.brain = brain
        self.message = message
        self.attachments = list(attachments or [])

    @Slot()
    def run(self):
        try:
            self.brain.generate_stream(
                self.message,
                self.chunk.emit,
                attachments=self.attachments,
            )
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
