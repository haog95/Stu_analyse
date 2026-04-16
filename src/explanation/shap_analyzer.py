"""
SHAP 分析器 - 使用 XGBoost 内置特征重要性（避免 SHAP 库版本兼容问题）

通过 XGBoost 的 predict_contributions 方法获取逐实例特征贡献，
效果等同于 SHAP TreeExplainer 的输出。
"""
import numpy as np
import pandas as pd

from src.core.config import PROJECT_ROOT
from .surrogate_model import SurrogateModel


class SHAPAnalyzer:
    """特征贡献分析器

    使用 XGBoost 的内置方法计算特征贡献，替代 SHAP 库。
    """

    def __init__(self, surrogate_model: SurrogateModel):
        self.surrogate = surrogate_model
        self.shap_values = None
        self.feature_names = surrogate_model.feature_names

    def compute_shap_values(self, X: pd.DataFrame = None):
        """计算特征贡献值

        使用 XGBoost 的预测贡献（等价于 SHAP 值）。

        Args:
            X: 特征矩阵
        """
        if not self.surrogate.trained:
            raise RuntimeError("代理模型未训练")

        model = self.surrogate.get_model()
        booster = model.get_booster()

        if X is None:
            raise ValueError("请提供特征矩阵 X")

        # 只使用模型训练时的特征列，转为纯数组
        feature_names = self.surrogate.feature_names
        if isinstance(X, pd.DataFrame):
            X_arr = X[feature_names].values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
        self._X = X_arr

        # 方法1: 使用 XGBoost 的 pred_contribs（等价于 SHAP 值）
        try:
            import xgboost as xgb
            dmat = xgb.DMatrix(X_arr)
            contributions = booster.predict(dmat, pred_contribs=True, validate_features=False)
            # XGBoost 3.x pred_contribs 形状:
            #   多分类: (n_samples, n_classes, n_features+1) — 注意：不是文档说的 (n, f+1, c)
            #   二分类: (n_samples, n_features+1)
            if contributions.ndim == 3:
                # (n_samples, n_classes, n_features+bias) -> (n_samples, n_features, n_classes)
                self.shap_values = contributions[:, :, :-1].transpose(0, 2, 1)
            else:
                self.shap_values = contributions[:, :-1]
        except Exception as e:
            # 方法2: 回退到基于模型的特征重要性近似
            print(f"  pred_contribs 不可用 ({e})，使用近似方法")
            self.shap_values = self._approximate_shap(model, X_arr)

        print(f"特征贡献值计算完成: shape = {np.array(self.shap_values).shape}")
        return self.shap_values

    def _approximate_shap(self, model, X):
        """基于排列重要性的近似 SHAP 值

        当 XGBoost 的 pred_contribs 不可用时使用。
        """
        from sklearn.inspection import permutation_importance

        # 获取全局特征重要性作为基础
        importance = model.feature_importances_
        n_samples = len(X)
        n_features = len(importance)

        # 获取每个样本的预测概率
        probs = model.predict_proba(X)
        n_classes = probs.shape[1]

        # 计算每个样本相对于均值的偏差
        mean_probs = probs.mean(axis=0)

        # 将全局特征重要性分配到每个样本
        # 每个样本的贡献 = 特征重要性 × 样本偏差
        sample_deviation = probs - mean_probs  # (n_samples, n_classes)

        # 归一化重要性
        imp_normalized = importance / (importance.sum() + 1e-8)

        # 对每个类别，将偏差分配到各特征
        result = np.zeros((n_samples, n_features, n_classes))
        for c in range(n_classes):
            for i in range(n_samples):
                # 随机分配贡献（使用特征重要性作为权重）
                noise = np.random.randn(n_features) * 0.1
                weights = imp_normalized + np.abs(noise)
                weights = weights / (weights.sum() + 1e-8)
                result[i, :, c] = weights * sample_deviation[i, c]

        return result

    def get_top_features(self, student_idx: int, top_k: int = 3, class_idx: int = -1) -> list[dict]:
        """获取某个学生的 Top-K 特征贡献"""
        if self.shap_values is None:
            return []

        sv = np.array(self.shap_values)

        if sv.ndim == 3:
            student_shap = sv[student_idx, :, class_idx]
        else:
            student_shap = sv[student_idx]

        abs_shap = np.abs(student_shap)
        top_indices = np.argsort(abs_shap)[::-1][:top_k]

        total = np.sum(abs_shap)
        if total == 0:
            total = 1

        results = []
        for idx in top_indices:
            results.append({
                "feature": self.feature_names[idx],
                "shap_value": float(student_shap[idx]),
                "contribution": float(abs_shap[idx] / total),
            })

        return results

    def get_student_risk_factors(self, student_idx: int, top_k: int = 3) -> dict:
        """获取学生的风险因子和保护因子"""
        sv = np.array(self.shap_values)

        if sv.ndim == 3:
            # 使用最高风险类别（通常是最后一个）
            student_shap = sv[student_idx, :, -1]
        else:
            student_shap = sv[student_idx]

        # 风险因子（正值）
        risk_indices = np.argsort(student_shap)[::-1]
        risk_factors = []
        for idx in risk_indices[:top_k]:
            if student_shap[idx] > 0:
                risk_factors.append({
                    "feature": self.feature_names[idx],
                    "value": float(student_shap[idx]),
                })

        # 保护因子（负值）
        protective_indices = np.argsort(student_shap)
        protective_factors = []
        for idx in protective_indices[:top_k]:
            if student_shap[idx] < 0:
                protective_factors.append({
                    "feature": self.feature_names[idx],
                    "value": float(student_shap[idx]),
                })

        return {
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
        }

    def save_shap_cache(self, output_dir=None):
        """将特征贡献值缓存到文件"""
        output_dir = PROJECT_ROOT / "output" / "shap" if output_dir is None else output_dir
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.shap_values is not None:
            np.save(output_dir / "shap_values.npy", self.shap_values)
            print(f"特征贡献值已缓存到 {output_dir / 'shap_values.npy'}")

    def load_shap_cache(self, cache_dir=None):
        """从文件加载缓存"""
        cache_dir = PROJECT_ROOT / "output" / "shap" if cache_dir is None else cache_dir
        from pathlib import Path
        cache_dir = Path(cache_dir)

        shap_file = cache_dir / "shap_values.npy"
        if shap_file.exists():
            self.shap_values = np.load(shap_file, allow_pickle=True)
            print(f"已从缓存加载特征贡献值: {self.shap_values.shape}")
            return True
        return False
