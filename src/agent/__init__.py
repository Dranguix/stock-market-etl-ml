"""Marco observe/act del libro de 50 dólares."""

from src.agent.book import Book
from src.agent.investor import AgenteInversor
from src.agent.policy import PoliticaInversor
from src.agent.trading import AgenteTrading, consultar

__all__ = ["AgenteInversor", "AgenteTrading", "Book", "PoliticaInversor", "consultar"]
