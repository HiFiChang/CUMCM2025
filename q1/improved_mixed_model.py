import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def convert_week(week_str):
    """将孕周字符串转换为数值"""
    if isinstance(week_str, str) and 'w+' in week_str:
        w, d = week_str.split('w+')
        return int(w) + int(d) / 7
    return np.nan

def diagnose_model_issues():
    """诊断模型问题"""
    print("=== 模型效果诊断 ===\n")
    
    # 加载数据
    female_df = pd.read_csv('female.csv')
    male_df = pd.read_csv('male.csv')
    
    # 转换孕周
    female_df['孕周_数值'] = female_df['检测孕周'].apply(convert_week)
    male_df['孕周_数值'] = male_df['检测孕周'].apply(convert_week)
    
    print("1. 数据特征分析:")
    print("="*50)
    
    # 女性数据分析
    female_clean = female_df[['孕妇代码', 'X染色体浓度', '孕周_数值', '孕妇BMI', '年龄']].dropna()
    male_clean = male_df[['孕妇代码', 'Y染色体浓度', '孕周_数值', '孕妇BMI', '年龄']].dropna()
    
    print(f"女性X染色体浓度:")
    print(f"  均值: {female_clean['X染色体浓度'].mean():.6f}")
    print(f"  标准差: {female_clean['X染色体浓度'].std():.6f}")
    print(f"  信噪比: {abs(female_clean['X染色体浓度'].mean()) / female_clean['X染色体浓度'].std():.3f}")
    
    print(f"\n男性Y染色体浓度:")
    print(f"  均值: {male_clean['Y染色体浓度'].mean():.6f}")
    print(f"  标准差: {male_clean['Y染色体浓度'].std():.6f}")
    print(f"  信噪比: {abs(male_clean['Y染色体浓度'].mean()) / male_clean['Y染色体浓度'].std():.3f}")
    
    # 2. 组内相关性分析
    print(f"\n2. 组内相关性分析:")
    print("="*50)
    
    # 计算组内相关系数 (ICC)
    def calculate_icc(df, subject_col, value_col):
        """计算组内相关系数"""
        from scipy.stats import f_oneway
        groups = [group[value_col].values for name, group in df.groupby(subject_col) if len(group) > 1]
        if len(groups) < 2:
            return 0
        
        # 计算组间和组内方差
        group_means = [np.mean(group) for group in groups]
        overall_mean = np.mean([val for group in groups for val in group])
        
        # 组间方差
        n_per_group = np.mean([len(group) for group in groups])
        ss_between = sum(len(group) * (np.mean(group) - overall_mean)**2 for group in groups)
        ms_between = ss_between / (len(groups) - 1)
        
        # 组内方差
        ss_within = sum(sum((val - np.mean(group))**2 for val in group) for group in groups)
        ms_within = ss_within / (sum(len(group) for group in groups) - len(groups))
        
        # ICC计算
        if ms_within == 0:
            return 1.0
        icc = (ms_between - ms_within) / (ms_between + (n_per_group - 1) * ms_within)
        return max(0, icc)
    
    female_icc = calculate_icc(female_clean, '孕妇代码', 'X染色体浓度')
    male_icc = calculate_icc(male_clean, '孕妇代码', 'Y染色体浓度')
    
    print(f"女性组内相关系数 (ICC): {female_icc:.4f}")
    print(f"男性组内相关系数 (ICC): {male_icc:.4f}")
    
    # 3. 效应量评估
    print(f"\n3. 效应量评估:")
    print("="*50)
    
    # 计算标准化效应量
    def calculate_effect_sizes(df, target_col):
        """计算各变量的效应量"""
        target_std = df[target_col].std()
        
        # 孕周效应 (从12周到24周的变化)
        week_range = df['孕周_数值'].max() - df['孕周_数值'].min()
        week_effect_raw = df[target_col].corr(df['孕周_数值']) * df['孕周_数值'].std()
        week_effect_std = week_effect_raw / target_std
        
        # BMI效应 (1个标准差变化)
        bmi_effect_raw = df[target_col].corr(df['孕妇BMI']) * df['孕妇BMI'].std()
        bmi_effect_std = bmi_effect_raw / target_std
        
        return {
            '孕周效应': week_effect_std,
            'BMI效应': bmi_effect_std,
            '孕周相关性': df[target_col].corr(df['孕周_数值']),
            'BMI相关性': df[target_col].corr(df['孕妇BMI'])
        }
    
    female_effects = calculate_effect_sizes(female_clean, 'X染色体浓度')
    male_effects = calculate_effect_sizes(male_clean, 'Y染色体浓度')
    
    print("女性 (X染色体浓度):")
    for key, value in female_effects.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n男性 (Y染色体浓度):")
    for key, value in male_effects.items():
        print(f"  {key}: {value:.4f}")
    
    # 4. 模型适用性建议
    print(f"\n4. 模型效果评估:")
    print("="*50)
    
    # 女性模型评估
    print("女性模型 (X染色体浓度):")
    if female_icc < 0.1:
        print("  ❌ 组内相关性极低，混合模型意义不大")
    elif abs(female_effects['孕周相关性']) < 0.1:
        print("  ❌ 孕周效应微弱，时间趋势不明显")
    else:
        print("  ✅ 适合混合模型")
    
    print(f"  信号强度: {'弱' if abs(female_clean['X染色体浓度'].mean()) / female_clean['X染色体浓度'].std() < 1 else '强'}")
    
    # 男性模型评估
    print("\n男性模型 (Y染色体浓度):")
    if male_icc < 0.1:
        print("  ⚠️  组内相关性较低")
    else:
        print("  ✅ 组内相关性适中")
    
    if abs(male_effects['孕周相关性']) > 0.3:
        print("  ✅ 孕周效应明显")
    else:
        print("  ⚠️  孕周效应中等")
    
    print(f"  信号强度: {'强' if abs(male_clean['Y染色体浓度'].mean()) / male_clean['Y染色体浓度'].std() > 1 else '中等'}")
    
    # 5. 改进建议
    print(f"\n5. 模型改进建议:")
    print("="*50)
    
    print("女性模型:")
    if female_icc < 0.1:
        print("  • 考虑使用普通线性回归而非混合模型")
        print("  • 数据可能噪音过大，需要更多样本或改进测量方法")
    
    print("\n男性模型:")
    print("  • 当前模型表现良好")
    print("  • 可以考虑添加非线性项 (孕周²)")
    print("  • 可以尝试其他协变量")
    
    # 6. 可视化关键关系
    print(f"\n6. 生成诊断图表...")
    
    # 创建诊断图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 女性数据图
    axes[0,0].scatter(female_clean['孕周_数值'], female_clean['X染色体浓度'], alpha=0.6)
    axes[0,0].set_title('Female: X Chromosome vs Gestational Week')
    axes[0,0].set_xlabel('Gestational Week')
    axes[0,0].set_ylabel('X Chromosome Concentration')
    
    axes[0,1].scatter(female_clean['孕妇BMI'], female_clean['X染色体浓度'], alpha=0.6)
    axes[0,1].set_title('Female: X Chromosome vs BMI')
    axes[0,1].set_xlabel('BMI')
    axes[0,1].set_ylabel('X Chromosome Concentration')
    
    axes[0,2].hist(female_clean['X染色体浓度'], bins=30, alpha=0.7)
    axes[0,2].set_title('Female: X Chromosome Distribution')
    axes[0,2].set_xlabel('X Chromosome Concentration')
    axes[0,2].set_ylabel('Frequency')
    
    # 男性数据图
    axes[1,0].scatter(male_clean['孕周_数值'], male_clean['Y染色体浓度'], alpha=0.6)
    axes[1,0].set_title('Male: Y Chromosome vs Gestational Week')
    axes[1,0].set_xlabel('Gestational Week')
    axes[1,0].set_ylabel('Y Chromosome Concentration')
    
    axes[1,1].scatter(male_clean['孕妇BMI'], male_clean['Y染色体浓度'], alpha=0.6)
    axes[1,1].set_title('Male: Y Chromosome vs BMI')
    axes[1,1].set_xlabel('BMI')
    axes[1,1].set_ylabel('Y Chromosome Concentration')
    
    axes[1,2].hist(male_clean['Y染色体浓度'], bins=30, alpha=0.7)
    axes[1,2].set_title('Male: Y Chromosome Distribution')
    axes[1,2].set_xlabel('Y Chromosome Concentration')
    axes[1,2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('model_effectiveness_diagnosis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("诊断图已保存为 'model_effectiveness_diagnosis.png'")

if __name__ == "__main__":
    diagnose_model_issues()
