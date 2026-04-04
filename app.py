"""
Nombre: Edcarlin Angeuris Celeten Benitez
Matrícula: 24-EISN-2-017
Punto de entrada principal - Lector de Recetas Médicas
"""

from app.main import PrescriptionApp

if __name__ == "__main__":
    app = PrescriptionApp()
    interface = app.create_interface()
    interface.launch()