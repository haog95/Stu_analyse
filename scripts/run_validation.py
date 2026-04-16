"""
系统完整性验证脚本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import load_config, PROJECT_ROOT
from src.core.data_loader import get_data_summary
from src.portraits.registry import PortraitRegistry
from src.core.data_merger import merge_all_portraits
from src.explanation.surrogate_model import SurrogateModel


def validate():
    checks = []
    print("=" * 60)
    print("学生多维画像分析系统 - 完整性验证")
    print("=" * 60)

    # 1. 数据文件
    print("\n[1/5] 检查数据文件...")
    data_dir = PROJECT_ROOT / "data"
    csv_files = list(data_dir.glob("*.csv"))
    check1 = len(csv_files) == 10
    checks.append(("数据文件", check1, f"{len(csv_files)}/10 个CSV文件"))
    print(f"  {'OK' if check1 else 'FAIL'}: {len(csv_files)} 个CSV文件")

    # 2. 配置文件
    print("\n[2/5] 检查配置文件...")
    config = load_config()
    required_configs = ["data", "id_column_mapping", "risk_thresholds", "llm"]
    check2 = all(k in config for k in required_configs)
    checks.append(("配置文件", check2, ""))
    print(f"  {'OK' if check2 else 'FAIL'}")

    # 3. 画像加载
    print("\n[3/5] 检查画像加载...")
    registry = PortraitRegistry()
    registry.initialize()
    check3 = registry.dimension_count == 10
    checks.append(("画像加载", check3, f"{registry.dimension_count}/10 个维度"))
    print(f"  {'OK' if check3 else 'FAIL'}: {registry.dimension_count} 个维度")

    # 4. 数据合并
    print("\n[4/5] 检查数据合并...")
    merged_df = merge_all_portraits()
    check4 = len(merged_df) > 2000
    checks.append(("数据合并", check4, f"{len(merged_df)} 行, {len(merged_df.columns)} 列"))
    print(f"  {'OK' if check4 else 'FAIL'}: {len(merged_df)} 行, {len(merged_df.columns)} 列")

    # 5. 模型训练
    print("\n[5/5] 检查代理模型...")
    surrogate = SurrogateModel()
    scores = surrogate.train(merged_df)
    check5 = scores["test_score"] > 0.9
    checks.append(("代理模型", check5, f"准确率: {scores['test_score']:.4f}"))
    print(f"  {'OK' if check5 else 'FAIL'}: 测试准确率 {scores['test_score']:.4f}")

    # 汇总
    print("\n" + "=" * 60)
    all_passed = all(c[1] for c in checks)
    status = "ALL PASSED" if all_passed else "SOME FAILED"
    print(f"验证结果: {status}")
    for name, passed, detail in checks:
        print(f"  {'[PASS]' if passed else '[FAIL]'} {name}: {detail}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
