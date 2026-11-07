"""Paquete de módulos de base de datos."""

try:
    from .connection import SQLServerConnection, get_connection
except ImportError:  # sin pyodbc (Cloud Run, MCP): lab_store y lab_reader no lo usan
    __all__ = []
else:
    __all__ = ['SQLServerConnection', 'get_connection']
