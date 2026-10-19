# src/optimizer.py
"""Implement Projected Gradient Descent and Nesterov Acceleration for AIDD.

Optimize chemical descriptor vectors towards viral target profiles while
enforcing Lipinski's Rule of Five via orthogonal mathematical projections.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


class LipinskiProjectedOptimizer:
    """Optimize drug descriptors using Projected Gradient Descent (PGD)."""

    def __init__(self, target_logp: float, target_mw: float):
        """Initialize optimizer with virus-specific chemical targets.

        Args:
            target_logp: Ideal partition coefficient for the target pocket.
            target_mw: Ideal Molecular Weight for the target pocket.
        """
        self.target = np.array([target_logp, target_mw], dtype=float)
        
        # Определяем границы выпуклого множества X (Правила Липинского Ro5)
        # Множество X: { LogP ∈ [-2.0, 5.0], MW ∈ [100.0, 500.0] }
        self.bounds = np.array([[-2.0, 5.0], [100.0, 500.0]], dtype=float)

    def _project_onto_lipinski_set(self, x: np.ndarray) -> np.ndarray:
        """Apply orthogonal projection operator (Proj_X) onto Lipinski set.

        Clip the descriptor vector to keep it inside mathematically and
        physically acceptable drug-likeness boundaries.
        """
        # Оператор проекции на гиперпараллелепипед — это поэлементный клиппинг
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def compute_loss_and_gradient(self, x: np.ndarray) -> tuple:
        """Calculate MSE loss and analytical gradient vector.

        f(x) = 0.5 * ||x - x_target||^2
        grad_f(x) = x - x_target
        """
        error = x - self.target
        loss = 0.5 * np.sum(error ** 2)
        gradient = error
        return loss, gradient

    def step_pgd(self, x_current: np.ndarray, lr: float = 0.1) -> tuple:
        """Execute one pure Projected Gradient Descent iteration.

        x_{k+1} = Proj_X ( x_k - lr * grad_f(x_k) )
        """
        loss, gradient = self.compute_loss_and_gradient(x_current)
        
        # Стандартный шаг градиентного спуска
        x_next = x_current - lr * gradient
        
        # Применение оператора проекции (каноны fmin.xyz / hse26)
        x_projected = self._project_onto_lipinski_set(x_next)
        
        return x_projected, loss

    def step_nesterov_pgd(
        self, x_k: np.ndarray, v_k: np.ndarray, epoch: int, lr: float = 0.1
    ) -> tuple:
        """Execute one Nesterov Accelerated Projected Gradient step (NAG-PGD).

        Yумный импульс Нестерова с последующей проекцией на множество ограничений.
        """
        # Вычисляем коэффициент затухания импульса ( HSE Выпуклый анализ )
        momentum = epoch / (epoch + 3.0)
        
        # Экстраполяция точки (шаг по направлению накопленной скорости)
        y_k = x_k + momentum * v_k
        
        loss, gradient = self.compute_loss_and_gradient(y_k)
        
        # Градиентный шаг из экстраполированной точки и наложение проекции
        x_next = self._project_onto_lipinski_set(y_k - lr * gradient)
        
        # Обновление вектора скорости (направления импульса)
        v_next = x_next - x_k
        
        return x_next, v_next, loss


def validate_molecule_descriptors(smiles: str) -> tuple:
    """Extract real physical chemical descriptors from a SMILES string.

    Args:
        smiles: Canonical SMILES representation of the compound.

    Returns:
        Tuple of (calculated_logp, calculated_mw).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Syntax error parsing SMILES: {smiles}")
    return Descriptors.MolLogP(mol), Descriptors.MolWt(mol)


if __name__ == "__main__":
    print("=== Тестирование хемоинформационного PGD / NAG оптимизатора ===")
    
    # Вирусологическая мишень: Главная протеаза SARS-CoV-2 Mpro (Цель: 2.5 / 410.0)
    target_logp, target_mw = 2.5, 410.0
    optimizer = LipinskiProjectedOptimizer(target_logp, target_mw)
    
    x_init = np.array([6.5, 550.0], dtype=float)
    
    x_start = optimizer._project_onto_lipinski_set(x_init)
    print(f"Стартовая точка после Proj_X (Ro5): LogP={x_start[0]:.1f}, MW={x_start[1]:.1f}")
    
    x_k = x_start
    v_k = np.zeros_like(x_k)
    
    print("\nЗапуск итерационного процесса схождения Нестерова...")
    for epoch in range(1, 6):
        x_k, v_k, loss_val = optimizer.step_nesterov_pgd(x_k, v_k, epoch, lr=0.2)
        print(f"[Эпоха {epoch}] Текущие дескрипторы: LogP={x_k[0]:.2f}, MW={x_k[1]:.2f} | Loss: {loss_val:.4f}")
        
    # Валидируем реальную молекулу (Парацетамол из PubChem) через RDKit
    real_logp, real_mw = validate_molecule_descriptors("CC(=O)NC1=CC=C(C=C1)O")
    print(f"\n[RDKit Валидация] Настоящие свойства эталонного лиганда: LogP={real_logp:.2f}, MW={real_mw:.2f}")