"""
代理模型 - 基于已有画像特征训练 XGBoost 代理模型
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.core.config import get_config, PROJECT_ROOT


# 用于代理模型的特征列（排除ID、标签和衍生列）
FEATURE_COLUMNS = [
    "avg_job_rate", "total_special_time", "early_bird_rate",
    "night_owl_rate", "lib_visit_count", "procrastination_index",
    "delay_stability", "final_score", "overall_rank_pct",
    "行为偏移分值", "干预建议权重",
    "total_active_days",
    "异常得分", "沉迷指数",
    "ZYNJ_Score", "CJ_Score", "SZF_Score", "NLF_Score",
    "Schol_Count", "Schol_Total_Score",
    "Comp_Count", "Comp_Total_Score",
]

# 目标列（使用画像9的 Warning_Label 作为多分类目标）
TARGET_COLUMN = "Warning_Label"


class SurrogateModel:
    """代理 XGBoost 模型

    使用已计算的画像特征训练代理模型，用于 SHAP 分析。
    目标变量使用画像9的预警标签（4分类）。
    """

    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.trained = False
        self.metrics = {}  # 存储模型评估指标
        self._X_test = None
        self._y_test = None

    def prepare_features(self, merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """准备特征矩阵和目标变量

        注意：不会修改原始 merged_df。

        Args:
            merged_df: 合并后的完整数据集

        Returns:
            (X, y) 特征矩阵和目标变量
        """
        # 如果已经训练过，直接使用已知的特征列
        if self.trained and self.feature_names:
            available_cols = [c for c in self.feature_names if c in merged_df.columns]
        else:
            # 首次调用：确定可用的数值型特征列
            available_cols = []
            for c in FEATURE_COLUMNS:
                if c in merged_df.columns:
                    col_data = merged_df[c]
                    if pd.api.types.is_numeric_dtype(col_data):
                        available_cols.append(c)
                    else:
                        # 不修改原始数据，在副本上尝试转换
                        try:
                            converted = pd.to_numeric(col_data, errors="coerce")
                            if converted.notna().sum() > 0:
                                available_cols.append(c)
                        except Exception:
                            pass
            self.feature_names = available_cols

        X = merged_df[available_cols].copy()

        # 非数值列转换
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # 处理缺失值和无穷大值
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        # 强制转为 float64
        X = X.astype(np.float64)

        # 目标变量
        if TARGET_COLUMN in merged_df.columns:
            y = merged_df[TARGET_COLUMN].fillna("未知")
        else:
            # 如果没有 Warning_Label，使用预测概率的二值化
            if "预测概率" in merged_df.columns:
                y = (merged_df["预测概率"].fillna(0) > 0.5).map({
                    True: "高风险", False: "低风险"
                })
            else:
                raise ValueError("没有可用的目标变量列")

        return X, y

    def train(self, merged_df: pd.DataFrame, test_size=0.2, random_state=42):
        """训练代理模型

        Args:
            merged_df: 合并后的完整数据集
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            训练得分
        """
        X, y = self.prepare_features(merged_df)

        # 编码目标变量
        y_encoded = self.label_encoder.fit_transform(y)

        # 类别分布统计
        class_counts = pd.Series(y).value_counts().to_dict()

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )

        self._X_test = X_test
        self._y_test = y_test

        # 训练 XGBoost（使用纯数值数组，不带列名，避免特征名校验问题）
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="mlogloss",
        )
        X_train_arr = X_train.values
        X_test_arr = X_test.values
        self.model.fit(X_train_arr, y_train)

        # 评估
        train_score = self.model.score(X_train_arr, y_train)
        test_score = self.model.score(X_test_arr, y_test)

        self.trained = True

        # 计算多分类 AUC（One-vs-Rest）
        auc_score = self._compute_auc(X_test_arr, y_test)

        # 计算每类别的 AUC
        per_class_auc = self._compute_per_class_auc(X_test_arr, y_test)

        # 计算混淆矩阵
        confusion_mat = self._compute_confusion_matrix(X_test_arr, y_test)

        # 五折交叉验证
        cv_score = self._compute_cv_score(X, y_encoded)

        self.metrics = {
            "train_accuracy": round(float(train_score), 4),
            "test_accuracy": round(float(test_score), 4),
            "auc_ovr": round(float(auc_score), 4),
            "per_class_auc": per_class_auc,
            "confusion_matrix": confusion_mat,
            "cv_5fold_mean": round(float(cv_score), 4),
            "n_features": len(self.feature_names),
            "n_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
            "train_test_split": f"{int((1-test_size)*100)}/{int(test_size*100)}",
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        print(f"代理模型训练完成:")
        print(f"  特征数: {len(self.feature_names)}")
        print(f"  训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
        print(f"  训练集准确率: {train_score:.4f}")
        print(f"  测试集准确率: {test_score:.4f}")
        print(f"  AUC (OvR macro): {auc_score:.4f}")
        print(f"  五折交叉验证均值: {cv_score:.4f}")
        print(f"  类别分布: {class_counts}")
        print(f"  各类别AUC:")
        for cls_name, auc_val in per_class_auc.items():
            print(f"    {cls_name}: {auc_val:.4f}")

        return {"train_score": train_score, "test_score": test_score}

    def _compute_auc(self, X_test, y_test) -> float:
        """计算多分类 AUC (One-vs-Rest macro)"""
        try:
            from sklearn.metrics import roc_auc_score
            y_prob = self.model.predict_proba(X_test)
            return roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            return 0.0

    def _compute_per_class_auc(self, X_test, y_test) -> dict:
        """计算每个类别的 AUC (One-vs-Rest)"""
        try:
            from sklearn.metrics import roc_auc_score
            y_prob = self.model.predict_proba(X_test)
            y_test_onehot = np.zeros_like(y_prob)
            y_test_onehot[np.arange(len(y_test)), y_test] = 1

            per_class = {}
            for i, cls_name in enumerate(self.label_encoder.classes_):
                try:
                    auc = roc_auc_score(y_test_onehot[:, i], y_prob[:, i])
                    per_class[str(cls_name)] = round(float(auc), 4)
                except ValueError:
                    per_class[str(cls_name)] = None
            return per_class
        except Exception:
            return {}

    def _compute_confusion_matrix(self, X_test, y_test) -> dict:
        """计算混淆矩阵"""
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            y_pred = self.model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            result = {
                "matrix": cm.tolist(),
                "classes": list(self.label_encoder.classes_),
            }

            # 每类别的精确率、召回率、F1
            report = classification_report(
                y_test, y_pred,
                target_names=[str(c) for c in self.label_encoder.classes_],
                output_dict=True,
                zero_division=0,
            )
            # 移除汇总行
            per_class_metrics = {}
            for cls_name in self.label_encoder.classes_:
                cls_str = str(cls_name)
                if cls_str in report:
                    per_class_metrics[cls_str] = {
                        "precision": round(report[cls_str]["precision"], 4),
                        "recall": round(report[cls_str]["recall"], 4),
                        "f1_score": round(report[cls_str]["f1-score"], 4),
                        "support": report[cls_str]["support"],
                    }
            result["per_class_metrics"] = per_class_metrics
            result["overall"] = {
                "accuracy": round(report["accuracy"], 4),
                "macro_f1": round(report["macro avg"]["f1-score"], 4),
                "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            }
            return result
        except Exception as e:
            return {"error": str(e)}

    def _compute_cv_score(self, X, y) -> float:
        """计算五折交叉验证准确率均值"""
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                self.model, X, y, cv=5, scoring="accuracy"
            )
            return float(scores.mean())
        except Exception:
            return 0.0

    def get_metrics(self) -> dict:
        """获取模型评估指标"""
        return self.metrics

    def predict(self, X) -> np.ndarray:
        """预测"""
        if not self.trained:
            raise RuntimeError("模型未训练，请先调用 train()")
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        return self.model.predict_proba(X)

    def get_model(self):
        """获取训练好的模型对象"""
        return self.model
