import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# 加载数据
df = pd.read_csv('male.csv', encoding='utf-8')

def convert_week(week_str):
    if isinstance(week_str, str) and 'w+' in week_str:
        w, d = week_str.split('w+')
        return int(w) + int(d) / 7
    return np.nan

df['孕周_数值'] = df['检测孕周'].apply(convert_week)

# 选择基础变量
analysis_cols = ['Y染色体浓度', '孕周_数值', '孕妇BMI', '年龄']
df_analysis = df[analysis_cols].copy()
df_analysis.dropna(inplace=True)

print("=== 多项式特征工程与相关性分析 ===\n")

# 创建平方项
df_analysis['孕周_平方'] = df_analysis['孕周_数值'] ** 2
df_analysis['BMI_平方'] = df_analysis['孕妇BMI'] ** 2

# 扩展后的特征集
extended_cols = ['Y染色体浓度', '孕周_数值', '孕妇BMI', '年龄', '孕周_平方', 'BMI_平方']
df_extended = df_analysis[extended_cols].copy()

print("1. 扩展特征集描述性统计:")
print(df_extended.describe())

# 计算相关性矩阵
correlation_matrix = df_extended.corr(method='pearson')

print("\n2. 扩展相关性矩阵:")
print(correlation_matrix.round(4))

# 重点关注与Y染色体浓度的相关性
y_correlations = correlation_matrix['Y染色体浓度'].sort_values(key=abs, ascending=False)
print("\n3. 各特征与Y染色体浓度的相关性排序:")
for feature, corr in y_correlations.items():
    if feature != 'Y染色体浓度':
        strength = ""
        if abs(corr) < 0.1:
            strength = "很弱"
        elif abs(corr) < 0.3:
            strength = "弱"
        elif abs(corr) < 0.5:
            strength = "中等"
        elif abs(corr) < 0.7:
            strength = "强"
        else:
            strength = "很强"
        
        direction = "正相关" if corr > 0 else "负相关"
        print(f"   {feature}: {corr:.4f} ({strength}{direction})")

# 生成扩展相关性热力图
plt.figure(figsize=(10, 8))
# 创建英文标签
corr_data = correlation_matrix.copy()
english_labels = ['Y_Concentration', 'Gestational_Week', 'BMI', 'Age', 'Week_Squared', 'BMI_Squared']
corr_data.index = english_labels
corr_data.columns = english_labels

# 生成热力图
mask = np.triu(np.ones_like(corr_data, dtype=bool))  # 只显示下三角
sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', fmt='.3f', 
            center=0, square=True, cbar_kws={"shrink": .8})
plt.title('Extended Correlation Matrix with Polynomial Features')
plt.tight_layout()
plt.savefig('extended_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n已保存扩展相关性热力图: extended_correlation_heatmap.png")

# === 多项式回归模型对比 ===
print("\n=== 多项式回归模型对比 ===\n")

# 模型1: 原始线性模型 (孕周 + BMI)
X1 = df_extended[['孕周_数值', '孕妇BMI']]
y = df_extended['Y染色体浓度']
X1 = sm.add_constant(X1)
model1 = sm.OLS(y, X1).fit()

print("模型1 - 线性模型 (孕周 + BMI):")
print(f"   调整R² = {model1.rsquared_adj:.4f} ({model1.rsquared_adj:.2%})")
print(f"   AIC = {model1.aic:.2f}")
print(f"   F检验 p值 = {model1.f_pvalue:.4e}")

# 模型2: 添加平方项 (孕周 + BMI + 孕周² + BMI²)
X2 = df_extended[['孕周_数值', '孕妇BMI', '孕周_平方', 'BMI_平方']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(y, X2).fit()

print("\n模型2 - 多项式模型 (孕周 + BMI + 孕周² + BMI²):")
print(f"   调整R² = {model2.rsquared_adj:.4f} ({model2.rsquared_adj:.2%})")
print(f"   AIC = {model2.aic:.2f}")
print(f"   F检验 p值 = {model2.f_pvalue:.4e}")

print("\n详细回归结果:")
print(model2.summary())

# 模型比较
r2_improvement = model2.rsquared_adj - model1.rsquared_adj
aic_improvement = model1.aic - model2.aic

print(f"\n4. 模型改进效果:")
print(f"   调整R²提升: {r2_improvement:.4f} ({r2_improvement/model1.rsquared_adj:.1%})")
print(f"   AIC改善: {aic_improvement:.2f} {'(更好)' if aic_improvement > 0 else '(更差)'}")

# 系数显著性分析
print(f"\n5. 各项系数显著性:")
for param, pvalue in model2.pvalues.items():
    if param != 'const':
        significance = "显著" if pvalue < 0.05 else "不显著"
        print(f"   {param}: p = {pvalue:.4f} ({significance})")

# === 可视化分析 ===
print("\n=== 生成可视化分析图 ===")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 孕周 vs Y染色体浓度 (原始 + 平方项)
axes[0,0].scatter(df_extended['孕周_数值'], df_extended['Y染色体浓度'], alpha=0.6, label='Data')
# 拟合二次曲线
week_range = np.linspace(df_extended['孕周_数值'].min(), df_extended['孕周_数值'].max(), 100)
week_squared = week_range ** 2
bmi_mean = df_extended['孕妇BMI'].mean()
bmi_squared_mean = bmi_mean ** 2

# 预测值 (固定BMI为均值)
pred_data = np.column_stack([np.ones(len(week_range)), week_range, 
                            np.full(len(week_range), bmi_mean), week_squared, 
                            np.full(len(week_range), bmi_squared_mean)])
y_pred = model2.predict(pred_data)
axes[0,0].plot(week_range, y_pred, 'r-', linewidth=2, label='Polynomial Fit')
axes[0,0].set_xlabel('Gestational Week')
axes[0,0].set_ylabel('Y Chromosome Concentration')
axes[0,0].set_title('Polynomial: Week vs Y Concentration')
axes[0,0].legend()
axes[0,0].grid(True)

# 2. BMI vs Y染色体浓度 (原始 + 平方项)
axes[0,1].scatter(df_extended['孕妇BMI'], df_extended['Y染色体浓度'], alpha=0.6, label='Data')
bmi_range = np.linspace(df_extended['孕妇BMI'].min(), df_extended['孕妇BMI'].max(), 100)
bmi_squared_range = bmi_range ** 2
week_mean = df_extended['孕周_数值'].mean()
week_squared_mean = week_mean ** 2

# 预测值 (固定孕周为均值)
pred_data = np.column_stack([np.ones(len(bmi_range)), 
                            np.full(len(bmi_range), week_mean),
                            bmi_range, np.full(len(bmi_range), week_squared_mean), 
                            bmi_squared_range])
y_pred = model2.predict(pred_data)
axes[0,1].plot(bmi_range, y_pred, 'r-', linewidth=2, label='Polynomial Fit')
axes[0,1].set_xlabel('BMI')
axes[0,1].set_ylabel('Y Chromosome Concentration')
axes[0,1].set_title('Polynomial: BMI vs Y Concentration')
axes[0,1].legend()
axes[0,1].grid(True)

# 3. 模型比较 (R²)
models = ['Linear\n(Week+BMI)', 'Polynomial\n(+Week²+BMI²)']
r2_values = [model1.rsquared_adj, model2.rsquared_adj]
axes[0,2].bar(models, r2_values, color=['skyblue', 'lightcoral'])
axes[0,2].set_ylabel('Adjusted R²')
axes[0,2].set_title('Model Comparison: R²')
for i, v in enumerate(r2_values):
    axes[0,2].text(i, v + 0.001, f'{v:.3f}', ha='center')

# 4. 残差分析 - 多项式模型
fitted_values = model2.fittedvalues
residuals = model2.resid
axes[1,0].scatter(fitted_values, residuals, alpha=0.6)
axes[1,0].axhline(y=0, color='red', linestyle='--')
axes[1,0].set_xlabel('Fitted Values')
axes[1,0].set_ylabel('Residuals')
axes[1,0].set_title('Residuals vs Fitted (Polynomial)')
axes[1,0].grid(True)

# 5. 系数重要性
coefficients = model2.params[1:]  # 排除截距
coef_names = ['Week', 'BMI', 'Week²', 'BMI²']
colors = ['green' if abs(c) > 0.001 else 'gray' for c in coefficients]
axes[1,1].barh(coef_names, coefficients, color=colors)
axes[1,1].set_xlabel('Coefficient Value')
axes[1,1].set_title('Polynomial Model Coefficients')
axes[1,1].grid(True)

# 6. 预测 vs 实际
axes[1,2].scatter(y, fitted_values, alpha=0.6)
axes[1,2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[1,2].set_xlabel('Actual Y Concentration')
axes[1,2].set_ylabel('Predicted Y Concentration')
axes[1,2].set_title('Actual vs Predicted (Polynomial)')
axes[1,2].grid(True)

plt.tight_layout()
plt.savefig('polynomial_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存多项式分析图: polynomial_analysis.png")

# === 最终总结 ===
print(f"\n=== 多项式模型总结 ===")
print(f"原始线性模型 调整R²: {model1.rsquared_adj:.4f}")
print(f"多项式模型 调整R²: {model2.rsquared_adj:.4f}")
print(f"提升幅度: {r2_improvement:.4f} ({r2_improvement/model1.rsquared_adj:.1%})")

if r2_improvement > 0.01:
    print("✅ 多项式项显著改善了模型拟合")
elif r2_improvement > 0.005:
    print("⚠️ 多项式项有轻微改善")
else:
    print("❌ 多项式项改善有限")

# 检查多项式项的显著性
week_sq_significant = model2.pvalues['孕周_平方'] < 0.05
bmi_sq_significant = model2.pvalues['BMI_平方'] < 0.05

print(f"\n平方项显著性:")
print(f"   孕周平方项: {'显著' if week_sq_significant else '不显著'} (p = {model2.pvalues['孕周_平方']:.4f})")
print(f"   BMI平方项: {'显著' if bmi_sq_significant else '不显著'} (p = {model2.pvalues['BMI_平方']:.4f})")

if week_sq_significant or bmi_sq_significant:
    print("✅ 至少一个平方项是显著的，存在非线性关系")
else:
    print("❌ 平方项都不显著，线性关系已足够")
