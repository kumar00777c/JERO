from abc import ABC, abstractmethod


class Brain(ABC):

    @abstractmethod
    def generate(self, message: str) -> str:
        pass