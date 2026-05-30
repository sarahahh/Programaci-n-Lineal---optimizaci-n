# ============================================================
# simplex_solver.py — El algoritmo Simplex
# ============================================================
# Implementa el método Simplex para MAXIMIZACIÓN con
# restricciones del tipo <=. Usa la forma estándar
# (aumentada) con variables de holgura.
#

import numpy as np


class SimplexSolver:

    """
    Resuelve un problema de PL por el método Simplex.

    """

    def __init__(self, objective, constraints):

        """
        Guarda el problema y prepara el estado inicial.

        """

        self.objective = objective
        self.constraints = constraints
        self.num_variables = len(objective)
        self.num_constraints = len(constraints)

        # Al iniciar el simplex, si las restricciones son <=,
        # las variables básicas iniciales son las holguras S1, S2, S3...
        self.basic_variables = [
            f"S{i + 1}" for i in range(self.num_constraints)
        ]

        self.tableau = None

    def create_tableau(self):

        """
        Construye el tablero simplex inicial (forma aumentada).
                  
        """

        rows = self.num_constraints + 1
        cols = self.num_variables + self.num_constraints + 1

        tableau = np.zeros((rows, cols))

        # Restricciones

        for i, constraint in enumerate(self.constraints):

            coeffs = constraint["coefficients"]
            rhs = constraint["rhs"]

            tableau[i, :self.num_variables] = coeffs

            # Variable de holgura

            tableau[i, self.num_variables + i] = 1

            # Lado derecho

            tableau[i, -1] = rhs

        # Función objetivo
        # Para maximización se colocan negativos los coeficientes

        tableau[-1, :self.num_variables] = -np.array(self.objective)

        self.tableau = tableau

    def get_variable_name(self, column_index):

        """
        Convierte un índice de columna en el nombre de variable.

        """
        # Si la columna pertenece a las variables originales
        
        if column_index < self.num_variables:
            return f"X{column_index + 1}"

        # Si la columna pertenece a variables de holgura

        slack_index = column_index - self.num_variables
        return f"S{slack_index + 1}"

    def is_optimal(self):

        """
        Verifica si el tablero actual ya es óptimo.
     
        """

        last_row = self.tableau[-1, :-1]

        # En maximización, cuando ya no hay negativos en la fila Z,
        # la solución actual es óptima

        return np.all(last_row >= 0)

    def get_pivot_column(self):

        """
        Determina la COLUMNA PIVOTE (variable que ENTRA a la base).
     
        """

        last_row = self.tableau[-1, :-1]

        # Entra la variable con el coeficiente más negativo
        return int(np.argmin(last_row))

    def get_pivot_row(self, pivot_col):

        """
        Determina la FILA PIVOTE (variable que SALE de la base).

        """

        ratios = []

        for i in range(self.num_constraints):

            element = self.tableau[i, pivot_col]

            if element > 0:
                ratio = self.tableau[i, -1] / element
            else:
                ratio = np.inf

            ratios.append(ratio)

        # Sale la variable con la menor razón positiva
        return int(np.argmin(ratios))

    def get_ratios(self, pivot_col):

        """
        Calcula y retorna TODAS las razones para mostrarlas
        en la interfaz (prueba de razón mínima visible).

        """

        ratios = []

        for i in range(self.num_constraints):

            element = self.tableau[i, pivot_col]

            if element > 0:
                ratio = self.tableau[i, -1] / element
            else:
                ratio = np.inf

            ratios.append(ratio)

        return ratios

    def pivot(self, pivot_row, pivot_col):

        """
        Realiza las operaciones de fila del PIVOTEO.

        """

        pivot_element = self.tableau[pivot_row, pivot_col]

        # Convertir el pivote en 1

        self.tableau[pivot_row] = (
            self.tableau[pivot_row] / pivot_element
        )

        # Convertir en 0 los demás elementos de la columna pivote
        
        for i in range(len(self.tableau)):

            if i != pivot_row:

                factor = self.tableau[i, pivot_col]

                self.tableau[i] = (
                    self.tableau[i]
                    - factor * self.tableau[pivot_row]
                )

    def solve(self):

        """
        Ejecuta el algoritmo Simplex completo.

        """

        self.create_tableau()

        iterations = []

        # Guardamos el tablero inicial
        iterations.append({
            "iteration": 0,
            "tableau": self.tableau.copy(),
            "basic_variables": self.basic_variables.copy(),
            "basic_variables_before": self.basic_variables.copy(),
            "entering_variable": None,
            "leaving_variable": None,
            "pivot_element": None,
            "ratios": None,
            "message": (
                "Tableau inicial. Se revisa la fila Z para buscar coeficientes negativos. "
                "Si existen coeficientes negativos, la solución aún no es óptima."
            )
        })

        # Ciclo principal del Simplex

        while not self.is_optimal():

            pivot_col = self.get_pivot_column()

            pivot_row = self.get_pivot_row(pivot_col)

            ratios = self.get_ratios(pivot_col)

            entering_variable = self.get_variable_name(pivot_col)
            leaving_variable = self.basic_variables[pivot_row]
            pivot_element = self.tableau[pivot_row, pivot_col]

            basic_variables_before = self.basic_variables.copy()

            # Actualizamos la variable básica de la fila pivote.
            # La variable que entra reemplaza a la que sale.
            self.basic_variables[pivot_row] = entering_variable

            self.pivot(pivot_row, pivot_col)

            # Guardamos cada tablero después del pivoteo

            iterations.append({
                "iteration": len(iterations),
                "tableau": self.tableau.copy(),
                "basic_variables": self.basic_variables.copy(),
                "basic_variables_before": basic_variables_before,
                "entering_variable": entering_variable,
                "leaving_variable": leaving_variable,
                "pivot_element": pivot_element,
                "ratios": ratios,
                "message": (
                    f"Entra {entering_variable}, sale {leaving_variable}. "
                    f"El elemento pivote es {pivot_element:.4f}."
                )
            })

        solution = np.zeros(self.num_variables)

        # Extraemos la solución final usando las variables básicas finales
        
        for row_index, variable_name in enumerate(self.basic_variables):

            if variable_name.startswith("X"):

                variable_index = int(variable_name[1:]) - 1

                if variable_index < self.num_variables:
                    solution[variable_index] = self.tableau[row_index, -1]

        optimal_value = self.tableau[-1, -1]

        return {
            "solution": solution,
            "optimal_value": optimal_value,
            "tableau": self.tableau,
            "iterations": iterations,
            "basic_variables": self.basic_variables
        }