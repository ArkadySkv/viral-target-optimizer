# src/screening.py
"""Implement high-throughput screening using Lipinski Projected Optimizer.

Batch process thousands of chemical compounds, evaluate their proximity to
viral target descriptors, and rank them using RDKit and PGD metrics.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from src.optimizer import LipinskiProjectedOptimizer, validate_molecule_descriptors


class VirtualScreeningEngine:
    """Screen and rank chemical libraries against viral protein targets."""

    def __init__(self, target_logp: float, target_mw: float):
        """Initialize screening engine with a target optimizer setup.

        Args:
            target_logp: Ideal partition coefficient for the viral pocket.
            target_mw: Ideal Molecular Weight for the viral pocket.
        """
        self.optimizer = LipinskiProjectedOptimizer(target_logp, target_mw)

    def screen_smiles_library(self, smiles_list: list) -> pd.DataFrame:
        """Process, filter, and rank a batch list of SMILES compounds.

        Args:
            smiles_list: List of strings containing canonical SMILES codes.

        Returns:
            Pandas DataFrame containing ranked valid chemical candidates.
        """
        print(f"[HTS] Инициализация пакетного скрининга {len(smiles_list)} соединений...")
        results = []

        for idx, smiles in enumerate(smiles_list):
            try:
                # Извлекаем реальные физические свойства молекулы через RDKit
                logp, mw = validate_molecule_descriptors(smiles)
                current_descriptors = np.array([logp, mw], dtype=float)

                # Вычисляем MSE Loss (дистанцию до идеального лекарства мишени)
                loss, _ = self.optimizer.compute_loss_and_gradient(current_descriptors)

                # Проверяем, удерживается ли молекула внутри множества Липинского
                # Считаем проекцию точки на границы Ro5 множества
                projected_descriptors = self.optimizer._project_onto_lipinski_set(current_descriptors)
                is_lipinski_valid = np.allclose(current_descriptors, projected_descriptors)

                results.append({
                    "smiles": smiles,
                    "calculated_logp": round(logp, 4),
                    "calculated_mw": round(mw, 2),
                    "target_distance_loss": round(float(loss), 4),
                    "lipinski_ro5_compliant": is_lipinski_valid
                })

            except ValueError:
                # Мягкий MLOps-пропуск поврежденных структур (SMILES Parse Error)
                print(f"[Warning] Пропущена невалидная структура на индексе {idx}: {smiles}")
                continue

        # Формируем итоговую таблицу результатов
        df_results = pd.DataFrame(results)

        # Ранжируем молекулы: чем МЕНЬШЕ loss (дистанция), тем выше аффинность к вирусу
        df_results = df_results.sort_values(by="target_distance_loss", ascending=True).reset_index(drop=True)
        return df_results


if __name__ == "__main__":
    print("=== Тестирование Высокопроизводительного Скрининга (HTS) ===")

    # Целевая мишень: Главная протеаза SARS-CoV-2 Mpro (Цель: LogP=2.5, MW=410.0)
    target_logp, target_mw = 2.5, 410.0
    engine = VirtualScreeningEngine(target_logp, target_mw)

    # Имитируем входящую химическую библиотеку скрининга (включая невалидный синтаксис)
    mock_chemical_library = [
        "CC(=O)NC1=CC=C(C=C1)O",                  # Парацетамол (Валидный, легкий)
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",            # Кофеин (Валидный)
        "CC1=C(C(C(=C(C1G)C(=O)O)C)C)C",          # Ошибка синтаксиса (SMILES Parse Error)
        "C([C@@H]1[C@@H]([C@H]([C@@H]([C@H](O1)O)O)O)O)O", # Глюкоза (Валидная)
        "CCCCCCCCCCCCCCCCCC(=O)O"                  # Стеариновая кислота (Нарушает Lipinski Ro5 по LogP)
    ]

    ranked_candidates = engine.screen_smiles_library(mock_chemical_library)

    print("\n=== Финальный хемоинформационный рейтинг молекул-кандидатов ===")
    print(ranked_candidates.to_string())