# ============================================================
# sensitivity.py — Análisis de sensibilidad
# ============================================================

import pandas as pd


def build_sensitivity_report(result, problem_data, num_variables):

    """
    Genera una tabla de análisis de sensibilidad básico.

    """
    
    tableau = result["tableau"]
    basic_variables = result["basic_variables"]
    constraints = problem_data["constraints"]

    num_constraints = len(constraints)

    sensitivity_rows = []

    for i in range(num_constraints):

        slack_name = f"S{i + 1}"

        slack_value = 0

        if slack_name in basic_variables:
            row_index = basic_variables.index(slack_name)
            slack_value = tableau[row_index, -1]

        if abs(slack_value) < 1e-8:
            status = "Activa"
            interpretation = "El recurso se utiliza completamente."
        else:
            status = "No activa"
            interpretation = "Existe recurso sobrante."

        sensitivity_rows.append({
            "Restricción": f"R{i + 1}",
            "Holgura": round(slack_value, 4),
            "Estado": status,
            "Interpretación": interpretation
        })

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    return sensitivity_df