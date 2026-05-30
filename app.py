# ============================================================
# app.py — Interfaz principal del programa
# ============================================================

import streamlit as st
import pandas as pd   # Para crear y mostrar tablas (DataFrames)
import numpy as np   # Para operaciones numéricas

# Importamos los módulos del proyecto que están en la carpeta /simplex/

from simplex.parser import build_problem
from simplex.simplex_solver import SimplexSolver
from simplex.graphics import solve_graphical_method

# ============================
# Configuración de la página 
# ============================

st.set_page_config(page_title="Simplex Optimizer", layout="wide")

st.title("Simplex Optimizer")
st.write("Ingrese un problema de Programación Lineal")


# =========================
# FUNCIONES AUXILIARES
# =========================

def format_objective(objective, problem_type):

    """
    Convierte la lista de coeficientes de la función objetivo
    en un string con notación matemática.

    """

    terms = []

    for i, coefficient in enumerate(objective):
        terms.append(f"{coefficient}x_{i + 1}")

    return f"{problem_type} \\ Z = " + " + ".join(terms)


def format_constraint(constraint):

    """
    Convierte un diccionario de restricción en un string matemático.

    """
    terms = []

    for i, coefficient in enumerate(constraint["coefficients"]):
        terms.append(f"{coefficient}x_{i + 1}")

    left_side = " + ".join(terms)

    return f"{left_side} {constraint['operator']} {constraint['rhs']}"


def build_tableau_dataframe(tableau, basic_variables, num_variables, num_constraints):

    """
    Convierte el tablero simplex (una matriz numpy) en un
    DataFrame de pandas con nombres de columnas y filas,
    para mostrarlo como tabla en pantalla.
    
    """
    column_names = []

    # Columnas para variables de decisión: X1, X2, ..., Xn

    for i in range(num_variables):
        column_names.append(f"X{i + 1}")

    # Columnas para variables de holgura: S1, S2, ..., Sm
    # Las variables de holgura se añaden al pasar a forma aumentada

    for i in range(num_constraints):
        column_names.append(f"S{i + 1}")

    # Última columna: el lado derecho (valores actuales)

    column_names.append("RHS")

    # Creamos el DataFrame con los nombres de columna

    tableau_df = pd.DataFrame(
        tableau,
        columns=column_names
    )

    # Los nombres de las filas son las variables básicas + Z al final

    row_names = basic_variables + ["Z"]
    tableau_df.index = row_names

    # Redondeamos a 4 decimales para que sea legible

    return tableau_df.round(4)


def show_simplex_iterations(result, problem_data, num_variables):

    """
    Esta es la función más grande del archivo. Muestra
    todo el proceso simplex paso a paso en pantalla:
      1. Resultado óptimo (valores de las variables)
      2. Variables de holgura en la solución final
      3. Cada iteración con su tablero, pivot, razones
      4. El tablero simplex final

    """
    num_constraints = len(problem_data["constraints"])

    st.subheader("Solución numérica paso a paso - Método Simplex")

    # =========================
    # RESULTADO ÓPTIMO
    # =========================

    st.write("### Resultado óptimo")

    solution_data = []

    # result["solution"] es una lista [x1, x2, ..., xn]

    for i, value in enumerate(result["solution"]):
        solution_data.append({
            "Variable": f"X{i + 1}",
            "Valor": round(value, 4)
        })

    # Convertimos la lista de diccionarios a tabla y la mostramos

    solution_df = pd.DataFrame(solution_data)
    st.dataframe(solution_df, use_container_width=True)

    # Muestra un cuadro verde con el valor óptimo Z

    st.success(f"Valor óptimo: Z = {result['optimal_value']:.4f}")

    # =========================
    # HOLGURAS
    # =========================

    """
    Las variables de holgura (S1, S2...) representan recursos
    no utilizados. Si S1 > 0, la primera restricción tiene
    "espacio sobrante" y no está activa.

    """
    st.write("### Variables de holgura en la solución final")

    slack_data = []

    for i, variable_name in enumerate(result["basic_variables"]):

        if variable_name.startswith("S"):
            slack_data.append({
                "Variable de holgura": variable_name,
                "Valor": round(result["tableau"][i, -1], 4),
                "Interpretación": "Recurso no utilizado"
            })

    if slack_data:
        slack_df = pd.DataFrame(slack_data)
        st.dataframe(slack_df, use_container_width=True)
    else:
        st.info("No hay variables de holgura positivas en la base final.")

    # =========================
    # ITERACIONES
    # =========================

    """
    Aquí está el corazón: mostramos cada paso del simplex.
    result["iterations"] es una lista de diccionarios,
    uno por cada pivoteo realizado.

    """
    st.write("### Iteraciones del método simplex")

    for step in result["iterations"]:

        st.write(f"#### Iteración {step['iteration']}")
        st.info(step["message"])

        if step["entering_variable"] is not None:

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Variable que entra", step["entering_variable"])

            with col2:
                st.metric("Variable que sale", step["leaving_variable"])

            with col3:
                st.metric("Elemento pivote", round(step["pivot_element"], 4))

            # Prueba de razón mínima

            ratio_data = []

            basic_before = step.get(
                "basic_variables_before",
                step["basic_variables"]
            )

            for i, ratio in enumerate(step["ratios"]):

                if np.isinf(ratio):
                    ratio_value = "-"
                else:
                    ratio_value = round(ratio, 4)

                ratio_data.append({
                    "Fila": basic_before[i],
                    "Razón RHS / columna pivote": ratio_value
                })

            st.write("##### Prueba de razón mínima")
            st.dataframe(
                pd.DataFrame(ratio_data),
                use_container_width=True
            )

        # Tablero de esta iteración

        tableau_df = build_tableau_dataframe(
            tableau=step["tableau"],
            basic_variables=step["basic_variables"],
            num_variables=num_variables,
            num_constraints=num_constraints
        )

        st.write("##### Tablero simplex")
        st.dataframe(tableau_df, use_container_width=True)

    # =========================
    # TABLERO SIMPLEX FINAL
    # =========================

    st.write("### Tablero simplex final")

    final_tableau_df = build_tableau_dataframe(
        tableau=result["tableau"],
        basic_variables=result["basic_variables"],
        num_variables=num_variables,
        num_constraints=num_constraints
    )

    st.dataframe(final_tableau_df, use_container_width=True)

    st.info(
        "En el tablero simplex final, las filas indican las variables básicas. "
        "La columna RHS muestra el valor final de cada variable básica. "
        "La última fila corresponde a la función objetivo."
    )


# =========================
# CONFIGURACIÓN GENERAL
# =========================

problem_type = st.selectbox(
    "Tipo de problema",
    ["Maximizar", "Minimizar"]
)

num_variables = st.number_input(
    "Número de variables",
    min_value=2,
    max_value=10,
    value=2,
    step=1
)

num_constraints = st.number_input(
    "Número de restricciones",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

solution_method = st.selectbox(
    "Seleccione el método de solución",
    [
        "Método numérico paso a paso",
        "Método gráfico",
        "Ambos"
    ]
)

st.divider()


# =========================
# FUNCIÓN OBJETIVO
# =========================

st.subheader("Función Objetivo")

objective_coeffs = []

cols = st.columns(num_variables)

for i in range(num_variables):
    coeff = cols[i].number_input(
        f"X{i + 1}",
        value=0.0,
        key=f"obj_{i}"
    )
    objective_coeffs.append(coeff)

st.latex(
    "Z = " + " + ".join(
        [f"{objective_coeffs[i]}x_{i + 1}" for i in range(num_variables)]
    )
)

st.divider()


# =========================
# RESTRICCIONES
# =========================

st.subheader("Restricciones")

constraints = []

for r in range(num_constraints):
    st.markdown(f"### Restricción {r + 1}")

    row = st.columns(num_variables + 2)

    coeffs = []

    for c in range(num_variables):
        value = row[c].number_input(
            f"X{c + 1}",
            value=0.0,
            key=f"r{r}c{c}"
        )
        coeffs.append(value)

    operator = row[num_variables].selectbox(
        "Operador",
        ["<=", ">=", "="],
        key=f"op_{r}"
    )

    rhs = row[num_variables + 1].number_input(
        "Resultado",
        value=0.0,
        key=f"rhs_{r}"
    )

    constraints.append({
        "coeffs": coeffs,
        "operator": operator,
        "rhs": rhs
    })

st.divider()


# =========================
# BOTÓN RESOLVER
# =========================

if st.button("Resolver Problema"):

    problem_data = build_problem(
        problem_type,
        objective_coeffs,
        constraints
    )

    st.success("Problema cargado correctamente")

    st.subheader("Resumen del Problema")

    st.write("### Tipo")
    st.write(problem_data["type"])

    st.write("### Función Objetivo")
    st.write(problem_data["objective"])

    st.write("### Restricciones")
    constraints_df = pd.DataFrame(problem_data["constraints"])
    st.dataframe(constraints_df, use_container_width=True)

    show_numeric = solution_method in [
        "Método numérico paso a paso",
        "Ambos"
    ]

    show_graphical = solution_method in [
        "Método gráfico",
        "Ambos"
    ]

    # =========================
    # MÉTODO GRÁFICO
    # =========================

    if show_graphical:

        if num_variables == 2:

            st.subheader("Método gráfico paso a paso")

            graphical_result = solve_graphical_method(
                objective=problem_data["objective"],
                constraints=problem_data["constraints"],
                problem_type=problem_data["type"]
            )

            st.write("## Paso 1: Modelo ingresado")

            st.write("### Función objetivo")
            st.latex(
                format_objective(
                    problem_data["objective"],
                    problem_data["type"]
                )
            )

            st.write("### Restricciones")
            for i, constraint in enumerate(problem_data["constraints"]):
                st.latex(f"R_{i + 1}: " + format_constraint(constraint))

            st.write("## Paso 2: Rectas frontera")

            st.info(
                "Para aplicar el método gráfico, cada restricción se toma como una "
                "recta frontera. Por ejemplo, una restricción del tipo <= se grafica "
                "primero como igualdad y luego se identifica el lado factible."
            )

            for i, constraint in enumerate(problem_data["constraints"]):
                frontera = constraint.copy()
                frontera["operator"] = "="
                st.latex(f"R_{i + 1}: " + format_constraint(frontera))

            st.write("## Paso 3: Vértices factibles encontrados")

            if graphical_result["vertices_table"].empty:

                st.error(
                    "No se encontraron vértices factibles. "
                    "El problema puede no tener región factible."
                )

            else:

                st.dataframe(
                    graphical_result["vertices_table"],
                    use_container_width=True
                )

                st.write("## Paso 4: Evaluación de la función objetivo")

                if problem_data["type"] == "Maximizar":
                    criterio = "mayor"
                else:
                    criterio = "menor"

                st.info(
                    f"La columna Z de la tabla anterior muestra el valor de la función objetivo "
                    f"evaluado en cada vértice factible. Como el problema es de "
                    f"{problem_data['type'].lower()}, se selecciona el vértice con el "
                    f"{criterio} valor de Z."
                )

                x1, x2 = graphical_result["optimal_point"]

                st.write("### Vértice seleccionado")
                st.latex(f"({x1:.2f}, {x2:.2f})")

                st.write("### Valor de la función objetivo")
                st.latex(f"Z = {graphical_result['optimal_value']:.2f}")

                st.write("## Paso 5: Conclusión")

                st.success(
                    f"Por lo tanto, la solución óptima es producir/asignar "
                    f"x1 = {x1:.2f} y x2 = {x2:.2f}, obteniendo "
                    f"Z = {graphical_result['optimal_value']:.2f}."
                )

                st.write("## Paso 6: Gráfica de la región factible")

                st.pyplot(graphical_result["figure"])

        else:

            st.warning(
                "El método gráfico solo aplica para problemas con dos variables."
            )

    # =========================
    # MÉTODO NUMÉRICO SIMPLEX
    # =========================

    if show_numeric:

        simplex_can_run = (
            problem_data["type"] == "Maximizar"
            and all(
                constraint["operator"] == "<="
                for constraint in problem_data["constraints"]
            )
        )

        if simplex_can_run:

            solver = SimplexSolver(
                objective=problem_data["objective"],
                constraints=problem_data["constraints"]
            )

            result = solver.solve()

            show_simplex_iterations(
                result=result,
                problem_data=problem_data,
                num_variables=num_variables
            )

        else:

            st.warning(
                "El método simplex paso a paso actual está implementado "
                "para problemas de maximización con restricciones <=. "
                "Para restricciones >=, igualdades o minimización se requiere "
                "implementar variables artificiales y método de la M grande."
            )