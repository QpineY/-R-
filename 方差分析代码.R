# ============================================================================
# 葡萄酒数据集方差分析实验报告
# UCI Wine Dataset - Analysis of Variance (ANOVA)
# ============================================================================

# 清空环境
rm(list = ls())

# 安装并加载必要的包
packages <- c("ggplot2", "dplyr", "tidyr", "gridExtra", "car", 
              "RColorBrewer", "reshape2", "agricolae", "viridis")

for(pkg in packages){
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 设置中文字体（根据系统调整）
par(family = "sans")

# 设置高饱和度配色方案
color_palette <- c("#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", 
                   "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2")

# ============================================================================
# 4.1 研究设计
# ============================================================================

cat("=" , rep("=", 70), "\n", sep = "")
cat("4.1 研究设计\n")
cat("=" , rep("=", 70), "\n", sep = "")

# 加载UCI葡萄酒数据集
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_data <- read.csv(url, header = FALSE)

# 添加列名
colnames(wine_data) <- c("Class", "Alcohol", "Malic_acid", "Ash", 
                         "Alcalinity_of_ash", "Magnesium", "Total_phenols",
                         "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
                         "Color_intensity", "Hue", "OD280_OD315", "Proline")

# 将Class转换为因子
wine_data$Class <- factor(wine_data$Class, labels = c("Class1", "Class2", "Class3"))

# 数据概览
cat("\n数据集基本信息：\n")
cat("样本数量:", nrow(wine_data), "\n")
cat("变量数量:", ncol(wine_data), "\n")
cat("葡萄酒类别分布:\n")
print(table(wine_data$Class))

cat("\n数据集前6行：\n")
print(head(wine_data))

# 描述性统计
cat("\n各类别葡萄酒的描述性统计（以酒精含量为例）：\n")
desc_stats <- wine_data %>%
  group_by(Class) %>%
  summarise(
    n = n(),
    Mean = mean(Alcohol),
    SD = sd(Alcohol),
    Min = min(Alcohol),
    Max = max(Alcohol),
    Median = median(Alcohol)
  )
print(desc_stats)

# 创建新的分类变量用于双因素方差分析
# 根据酒精含量创建高低分组
wine_data$Alcohol_Level <- cut(wine_data$Alcohol, 
                               breaks = quantile(wine_data$Alcohol, c(0, 0.5, 1)),
                               labels = c("Low", "High"),
                               include.lowest = TRUE)

# 图4.1.1: 数据集概览 - 各类别样本分布
p1 <- ggplot(wine_data, aes(x = Class, fill = Class)) +
  geom_bar(color = "black", size = 0.5) +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = color_palette[1:3]) +
  labs(title = "图4.1.1 各类别葡萄酒样本分布",
       x = "葡萄酒类别", y = "样本数量") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none",
        panel.grid.major.x = element_blank())

# 图4.1.2: 主要化学成分箱线图
wine_long <- wine_data %>%
  select(Class, Alcohol, Malic_acid, Total_phenols, Flavanoids) %>%
  pivot_longer(cols = -Class, names_to = "Variable", values_to = "Value")

p2 <- ggplot(wine_long, aes(x = Class, y = Value, fill = Class)) +
  geom_boxplot(alpha = 0.8, outlier.color = "red", outlier.size = 2) +
  facet_wrap(~Variable, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = color_palette[1:3]) +
  labs(title = "图4.1.2 不同类别葡萄酒主要化学成分分布",
       x = "葡萄酒类别", y = "含量") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        strip.background = element_rect(fill = "lightgray", color = "black"),
        strip.text = element_text(face = "bold"))

# 显示图表
print(p1)
print(p2)

# 保存图表
ggsave("图4.1.1_样本分布.png", p1, width = 8, height = 6, dpi = 300)
ggsave("图4.1.2_化学成分分布.png", p2, width = 10, height = 8, dpi = 300)


# ============================================================================
# 4.2 单因素方差分析
# ============================================================================

cat("\n\n", "=" , rep("=", 70), "\n", sep = "")
cat("4.2 单因素方差分析\n")
cat("=" , rep("=", 70), "\n", sep = "")

# 4.2.1 酒精含量的单因素方差分析
cat("\n【分析1】不同类别葡萄酒的酒精含量差异\n")
cat("-" , rep("-", 70), "\n", sep = "")

# 进行单因素方差分析
anova_alcohol <- aov(Alcohol ~ Class, data = wine_data)
cat("\n方差分析结果：\n")
print(summary(anova_alcohol))

# Tukey HSD 多重比较
tukey_alcohol <- TukeyHSD(anova_alcohol)
cat("\nTukey HSD 事后检验：\n")
print(tukey_alcohol)

# 图4.2.1: 酒精含量均值图
means_alcohol <- wine_data %>%
  group_by(Class) %>%
  summarise(Mean = mean(Alcohol),
            SE = sd(Alcohol)/sqrt(n()))

p3 <- ggplot(means_alcohol, aes(x = Class, y = Mean, fill = Class)) +
  geom_bar(stat = "identity", color = "black", size = 0.8, width = 0.6) +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), 
                width = 0.2, size = 1) +
  geom_text(aes(label = sprintf("%.2f", Mean)), 
            vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = color_palette[1:3]) +
  labs(title = "图4.2.1 不同类别葡萄酒酒精含量均值比较",
       x = "葡萄酒类别", y = "酒精含量 (%)") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none",
        panel.grid.major.x = element_blank())

# 图4.2.2: Tukey HSD 可视化
tukey_df <- as.data.frame(tukey_alcohol$Class)
tukey_df$Comparison <- rownames(tukey_df)

p4 <- ggplot(tukey_df, aes(x = Comparison, y = diff, 
                           color = ifelse(`p adj` < 0.05, "显著", "不显著"))) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), width = 0.2, size = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", size = 1) +
  scale_color_manual(values = c("显著" = "#FF6B6B", "不显著" = "#4ECDC4")) +
  labs(title = "图4.2.2 Tukey HSD 多重比较结果（酒精含量）",
       x = "组间比较", y = "均值差异", color = "显著性") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

print(p3)
print(p4)

# 4.2.2 类黄酮含量的单因素方差分析
cat("\n\n【分析2】不同类别葡萄酒的类黄酮含量差异\n")
cat("-" , rep("-", 70), "\n", sep = "")

anova_flavanoids <- aov(Flavanoids ~ Class, data = wine_data)
cat("\n方差分析结果：\n")
print(summary(anova_flavanoids))

tukey_flavanoids <- TukeyHSD(anova_flavanoids)
cat("\nTukey HSD 事后检验：\n")
print(tukey_flavanoids)

# 图4.2.3: 类黄酮含量箱线图 + 小提琴图
p5 <- ggplot(wine_data, aes(x = Class, y = Flavanoids, fill = Class)) +
  geom_violin(alpha = 0.6, trim = FALSE) +
  geom_boxplot(width = 0.2, fill = "white", outlier.color = "red", outlier.size = 2) +
  geom_jitter(width = 0.1, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = color_palette[1:3]) +
  labs(title = "图4.2.3 不同类别葡萄酒类黄酮含量分布",
       x = "葡萄酒类别", y = "类黄酮含量") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none")

print(p5)

# 图4.2.4: 方差齐性检验可视化
levene_alcohol <- leveneTest(Alcohol ~ Class, data = wine_data)
levene_flavanoids <- leveneTest(Flavanoids ~ Class, data = wine_data)

cat("\n方差齐性检验（Levene's Test）：\n")
cat("酒精含量:\n")
print(levene_alcohol)
cat("\n类黄酮含量:\n")
print(levene_flavanoids)

# 保存图表
ggsave("图4.2.1_酒精含量均值.png", p3, width = 8, height = 6, dpi = 300)
ggsave("图4.2.2_Tukey检验.png", p4, width = 8, height = 6, dpi = 300)
ggsave("图4.2.3_类黄酮分布.png", p5, width = 8, height = 6, dpi = 300)


# ============================================================================
# 4.3 双因素方差分析
# ============================================================================

cat("\n\n", "=" , rep("=", 70), "\n", sep = "")
cat("4.3 双因素方差分析\n")
cat("=" , rep("=", 70), "\n", sep = "")

# 双因素方差分析：葡萄酒类别 × 酒精含量水平 对总酚含量的影响
cat("\n【分析】葡萄酒类别和酒精含量水平对总酚含量的影响\n")
cat("-" , rep("-", 70), "\n", sep = "")

# 进行双因素方差分析
anova_two_way <- aov(Total_phenols ~ Class * Alcohol_Level, data = wine_data)
cat("\n双因素方差分析结果：\n")
print(summary(anova_two_way))

# 计算各组均值
means_two_way <- wine_data %>%
  group_by(Class, Alcohol_Level) %>%
  summarise(
    Mean = mean(Total_phenols),
    SD = sd(Total_phenols),
    SE = sd(Total_phenols)/sqrt(n()),
    .groups = "drop"
  )

cat("\n各组描述性统计：\n")
print(means_two_way)

# 图4.3.1: 交互效应图
p6 <- ggplot(means_two_way, aes(x = Class, y = Mean, color = Alcohol_Level, group = Alcohol_Level)) +
  geom_line(size = 1.5) +
  geom_point(size = 4, shape = 21, fill = "white", stroke = 2) +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), 
                width = 0.1, size = 1) +
  scale_color_manual(values = c("Low" = "#FF6B6B", "High" = "#4ECDC4")) +
  labs(title = "图4.3.1 葡萄酒类别与酒精水平对总酚含量的交互效应",
       x = "葡萄酒类别", y = "总酚含量均值",
       color = "酒精含量水平") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "top",
        legend.background = element_rect(fill = "white", color = "black"))

# 图4.3.2: 分组箱线图
p7 <- ggplot(wine_data, aes(x = Class, y = Total_phenols, fill = Alcohol_Level)) +
  geom_boxplot(position = position_dodge(0.8), alpha = 0.8, outlier.size = 2) +
  scale_fill_manual(values = c("Low" = "#FF6B6B", "High" = "#4ECDC4")) +
  labs(title = "图4.3.2 不同类别和酒精水平下总酚含量分布",
       x = "葡萄酒类别", y = "总酚含量",
       fill = "酒精含量水平") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "top")

# 图4.3.3: 热图展示均值
means_matrix <- means_two_way %>%
  select(Class, Alcohol_Level, Mean) %>%
  pivot_wider(names_from = Alcohol_Level, values_from = Mean)

means_matrix_long <- means_two_way %>%
  select(Class, Alcohol_Level, Mean)

p8 <- ggplot(means_matrix_long, aes(x = Alcohol_Level, y = Class, fill = Mean)) +
  geom_tile(color = "white", size = 1.5) +
  geom_text(aes(label = sprintf("%.2f", Mean)), 
            color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient2(low = "#4ECDC4", mid = "#FFA07A", high = "#FF6B6B",
                       midpoint = mean(means_matrix_long$Mean)) +
  labs(title = "图4.3.3 总酚含量均值热图",
       x = "酒精含量水平", y = "葡萄酒类别",
       fill = "总酚含量") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(p6)
print(p7)
print(p8)

# 简单效应分析
cat("\n简单效应分析：\n")
cat("\n在每个酒精水平下，葡萄酒类别的效应：\n")
for(level in levels(wine_data$Alcohol_Level)){
  cat("\n", level, "酒精含量组：\n", sep = "")
  subset_data <- wine_data[wine_data$Alcohol_Level == level, ]
  simple_anova <- aov(Total_phenols ~ Class, data = subset_data)
  print(summary(simple_anova))
}

# 保存图表
ggsave("图4.3.1_交互效应.png", p6, width = 10, height = 6, dpi = 300)
ggsave("图4.3.2_分组箱线图.png", p7, width = 10, height = 6, dpi = 300)
ggsave("图4.3.3_均值热图.png", p8, width = 8, height = 6, dpi = 300)


# ============================================================================
# 4.4 假设检验
# ============================================================================

cat("\n\n", "=" , rep("=", 70), "\n", sep = "")
cat("4.4 假设检验\n")
cat("=" , rep("=", 70), "\n", sep = "")

# 4.4.1 正态性检验
cat("\n【检验1】正态性检验\n")
cat("-" , rep("-", 70), "\n", sep = "")

# Shapiro-Wilk 检验
cat("\n各类别酒精含量的 Shapiro-Wilk 正态性检验：\n")
for(class in levels(wine_data$Class)){
  subset_data <- wine_data[wine_data$Class == class, "Alcohol"]
  test_result <- shapiro.test(subset_data)
  cat(class, ": W =", round(test_result$statistic, 4), 
      ", p-value =", format.pval(test_result$p.value, digits = 4), "\n")
}

# 图4.4.1: Q-Q图
p9 <- ggplot(wine_data, aes(sample = Alcohol, color = Class)) +
  stat_qq(size = 2, alpha = 0.6) +
  stat_qq_line(size = 1) +
  facet_wrap(~Class, ncol = 3) +
  scale_color_manual(values = color_palette[1:3]) +
  labs(title = "图4.4.1 酒精含量正态性Q-Q图",
       x = "理论分位数", y = "样本分位数") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none",
        strip.background = element_rect(fill = "lightgray", color = "black"),
        strip.text = element_text(face = "bold"))

# 图4.4.2: 直方图 + 正态曲线
p10 <- ggplot(wine_data, aes(x = Alcohol, fill = Class)) +
  geom_histogram(aes(y = after_stat(density)), bins = 15, 
                 alpha = 0.7, color = "black") +
  geom_density(alpha = 0.3, size = 1) +
  facet_wrap(~Class, ncol = 3) +
  scale_fill_manual(values = color_palette[1:3]) +
  labs(title = "图4.4.2 酒精含量分布直方图与密度曲线",
       x = "酒精含量 (%)", y = "密度") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none",
        strip.background = element_rect(fill = "lightgray", color = "black"),
        strip.text = element_text(face = "bold"))

print(p9)
print(p10)

# 4.4.2 方差齐性检验
cat("\n\n【检验2】方差齐性检验\n")
cat("-" , rep("-", 70), "\n", sep = "")

# Levene 检验
levene_test <- leveneTest(Alcohol ~ Class, data = wine_data)
cat("\nLevene 方差齐性检验：\n")
print(levene_test)

# Bartlett 检验
bartlett_test <- bartlett.test(Alcohol ~ Class, data = wine_data)
cat("\nBartlett 方差齐性检验：\n")
print(bartlett_test)

# 图4.4.3: 方差齐性可视化
variance_data <- wine_data %>%
  group_by(Class) %>%
  summarise(Variance = var(Alcohol),
            SD = sd(Alcohol))

p11 <- ggplot(variance_data, aes(x = Class, y = Variance, fill = Class)) +
  geom_bar(stat = "identity", color = "black", size = 0.8, width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Variance)), 
            vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = color_palette[1:3]) +
  labs(title = "图4.4.3 各类别酒精含量方差比较",
       x = "葡萄酒类别", y = "方差") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none",
        panel.grid.major.x = element_blank())

print(p11)

# 4.4.3 残差分析
cat("\n\n【检验3】残差分析\n")
cat("-" , rep("-", 70), "\n", sep = "")

# 获取残差
residuals_alcohol <- residuals(anova_alcohol)
fitted_values <- fitted(anova_alcohol)

# 图4.4.4: 残差图（四合一）
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(anova_alcohol, col = color_palette[1], pch = 16, cex = 1.2)
par(mfrow = c(1, 1))

# 使用ggplot创建更美观的残差图
residual_df <- data.frame(
  Fitted = fitted_values,
  Residuals = residuals_alcohol,
  Class = wine_data$Class,
  Standardized = rstandard(anova_alcohol)
)

p12 <- ggplot(residual_df, aes(x = Fitted, y = Residuals, color = Class)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 1) +
  geom_smooth(aes(group = 1), method = "loess", se = TRUE, 
              color = "blue", fill = "lightblue", alpha = 0.3) +
  scale_color_manual(values = color_palette[1:3]) +
  labs(title = "图4.4.4 残差与拟合值散点图",
       x = "拟合值", y = "残差") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

p13 <- ggplot(residual_df, aes(sample = Standardized)) +
  stat_qq(color = color_palette[1], size = 3, alpha = 0.7) +
  stat_qq_line(color = "red", size = 1, linetype = "dashed") +
  labs(title = "图4.4.5 标准化残差Q-Q图",
       x = "理论分位数", y = "标准化残差") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(p12)
print(p13)

# 4.4.4 效应量计算
cat("\n\n【检验4】效应量分析\n")
cat("-" , rep("-", 70), "\n", sep = "")

# 计算 Eta-squared (η²)
ss_total <- sum((wine_data$Alcohol - mean(wine_data$Alcohol))^2)
ss_between <- sum(tapply(wine_data$Alcohol, wine_data$Class, 
                         function(x) length(x) * (mean(x) - mean(wine_data$Alcohol))^2))
eta_squared <- ss_between / ss_total

cat("\nEta-squared (η²) =", round(eta_squared, 4), "\n")
cat("效应量解释: ")
if(eta_squared < 0.01){
  cat("小效应\n")
} else if(eta_squared < 0.06){
  cat("中等效应\n")
} else {
  cat("大效应\n")
}

# 计算 Omega-squared (ω²)
ms_within <- summary(anova_alcohol)[[1]]$`Mean Sq`[2]
k <- length(levels(wine_data$Class))
n <- nrow(wine_data)
omega_squared <- (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)

cat("Omega-squared (ω²) =", round(omega_squared, 4), "\n")

# 图4.4.6: 效应量可视化
effect_size_df <- data.frame(
  Measure = c("Eta-squared", "Omega-squared"),
  Value = c(eta_squared, omega_squared)
)

p14 <- ggplot(effect_size_df, aes(x = Measure, y = Value, fill = Measure)) +
  geom_bar(stat = "identity", color = "black", size = 1, width = 0.5) +
  geom_text(aes(label = sprintf("%.4f", Value)), 
            vjust = -0.5, size = 6, fontface = "bold") +
  scale_fill_manual(values = color_palette[1:2]) +
  ylim(0, max(effect_size_df$Value) * 1.2) +
  labs(title = "图4.4.6 效应量指标比较",
       x = "效应量指标", y = "数值") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none",
        panel.grid.major.x = element_blank())

print(p14)

# 保存所有假设检验相关图表
ggsave("图4.4.1_QQ图.png", p9, width = 12, height = 4, dpi = 300)
ggsave("图4.4.2_分布直方图.png", p10, width = 12, height = 4, dpi = 300)
ggsave("图4.4.3_方差比较.png", p11, width = 8, height = 6, dpi = 300)
ggsave("图4.4.4_残差图.png", p12, width = 8, height = 6, dpi = 300)
ggsave("图4.4.5_残差QQ图.png", p13, width = 8, height = 6, dpi = 300)
ggsave("图4.4.6_效应量.png", p14, width = 8, height = 6, dpi = 300)


# ============================================================================
# 综合结果汇总
# ============================================================================

cat("\n\n", "=" , rep("=", 70), "\n", sep = "")
cat("综合结果汇总\n")
cat("=" , rep("=", 70), "\n", sep = "")

# 创建结果汇总表
summary_results <- data.frame(
  分析类型 = c("单因素ANOVA(酒精)", "单因素ANOVA(类黄酮)", 
           "双因素ANOVA(主效应-类别)", "双因素ANOVA(主效应-酒精水平)",
           "双因素ANOVA(交互效应)"),
  F值 = c(
    summary(anova_alcohol)[[1]]$`F value`[1],
    summary(anova_flavanoids)[[1]]$`F value`[1],
    summary(anova_two_way)[[1]]$`F value`[1],
    summary(anova_two_way)[[1]]$`F value`[2],
    summary(anova_two_way)[[1]]$`F value`[3]
  ),
  P值 = c(
    summary(anova_alcohol)[[1]]$`Pr(>F)`[1],
    summary(anova_flavanoids)[[1]]$`Pr(>F)`[1],
    summary(anova_two_way)[[1]]$`Pr(>F)`[1],
    summary(anova_two_way)[[1]]$`Pr(>F)`[2],
    summary(anova_two_way)[[1]]$`Pr(>F)`[3]
  ),
  显著性 = c(
    ifelse(summary(anova_alcohol)[[1]]$`Pr(>F)`[1] < 0.001, "***",
           ifelse(summary(anova_alcohol)[[1]]$`Pr(>F)`[1] < 0.01, "**",
                  ifelse(summary(anova_alcohol)[[1]]$`Pr(>F)`[1] < 0.05, "*", "ns"))),
    ifelse(summary(anova_flavanoids)[[1]]$`Pr(>F)`[1] < 0.001, "***",
           ifelse(summary(anova_flavanoids)[[1]]$`Pr(>F)`[1] < 0.01, "**",
                  ifelse(summary(anova_flavanoids)[[1]]$`Pr(>F)`[1] < 0.05, "*", "ns"))),
    ifelse(summary(anova_two_way)[[1]]$`Pr(>F)`[1] < 0.001, "***",
           ifelse(summary(anova_two_way)[[1]]$`Pr(>F)`[1] < 0.01, "**",
                  ifelse(summary(anova_two_way)[[1]]$`Pr(>F)`[1] < 0.05, "*", "ns"))),
    ifelse(summary(anova_two_way)[[1]]$`Pr(>F)`[2] < 0.001, "***",
           ifelse(summary(anova_two_way)[[1]]$`Pr(>F)`[2] < 0.01, "**",
                  ifelse(summary(anova_two_way)[[1]]$`Pr(>F)`[2] < 0.05, "*", "ns"))),
    ifelse(summary(anova_two_way)[[1]]$`Pr(>F)`[3] < 0.001, "***",
           ifelse(summary(anova_two_way)[[1]]$`Pr(>F)`[3] < 0.01, "**",
                  ifelse(summary(anova_two_way)[[1]]$`Pr(>F)`[3] < 0.05, "*", "ns")))
  )
)

cat("\n方差分析结果汇总：\n")
print(summary_results)

cat("\n\n实验分析完成！所有图表已保存至工作目录。\n")
cat("工作目录:", getwd(), "\n")

cat("\n生成的图表清单：\n")
cat("- 图4.1.1_样本分布.png\n")
cat("- 图4.1.2_化学成分分布.png\n")
cat("- 图4.2.1_酒精含量均值.png\n")
cat("- 图4.2.2_Tukey检验.png\n")
cat("- 图4.2.3_类黄酮分布.png\n")
cat("- 图4.3.1_交互效应.png\n")
cat("- 图4.3.2_分组箱线图.png\n")
cat("- 图4.3.3_均值热图.png\n")
cat("- 图4.4.1_QQ图.png\n")
cat("- 图4.4.2_分布直方图.png\n")
cat("- 图4.4.3_方差比较.png\n")
cat("- 图4.4.4_残差图.png\n")
cat("- 图4.4.5_残差QQ图.png\n")
cat("- 图4.4.6_效应量.png\n")

cat("\n=" , rep("=", 70), "\n", sep = "")
