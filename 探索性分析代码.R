# ============================================================================
# 第四章：探索性数据分析
# UCI葡萄酒数据集分析
# ============================================================================

# --------------------------------------------------------------------------
# 环境准备：加载必要的包
# --------------------------------------------------------------------------
# 如果包未安装，请先运行以下代码安装：
# install.packages(c("tidyverse", "corrplot", "ggpubr", "patchwork", 
#                    "GGally", "reshape2", "RColorBrewer", "scales", 
#                    "gridExtra", "ggridges", "viridis"))

library(tidyverse)      # 数据处理和可视化
library(corrplot)       # 相关性图
library(ggpubr)         # 出版级图表
library(patchwork)      # 图表组合
library(GGally)         # 散点图矩阵
library(reshape2)       # 数据重塑
library(RColorBrewer)   # 配色
library(scales)         # 标度调整
library(gridExtra)      # 图表排列
library(ggridges)       # 山脊图
library(viridis)        # 配色方案

# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
# 加载UCI葡萄酒数据集（红葡萄酒和白葡萄酒）
# 数据来源：https://archive.ics.uci.edu/ml/datasets/wine+quality

# 读取红葡萄酒数据
red_wine <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", 
                     sep = ";", header = TRUE)
red_wine$type <- "Red"

# 读取白葡萄酒数据
white_wine <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", 
                       sep = ";", header = TRUE)
white_wine$type <- "White"

# 合并数据集
wine_data <- rbind(red_wine, white_wine)
wine_data$type <- as.factor(wine_data$type)

# 查看数据结构
str(wine_data)
head(wine_data)

# --------------------------------------------------------------------------
# 马卡龙配色方案定义
# --------------------------------------------------------------------------
macaron_colors <- c("#FFB6C1", "#B4E7CE", "#A8D8EA", "#FFD4A3", 
                    "#C9B8E4", "#FFE5CC", "#B5EAD7", "#FFDFD3",
                    "#F8B4D9", "#A8E6CF", "#FFDAC1", "#C7CEEA")

macaron_red <- "#FF9AA2"
macaron_white <- "#B5EAD7"
macaron_gradient <- c("#FFB6C1", "#FFDAC1", "#B5EAD7", "#A8D8EA", "#C9B8E4")

# 设置ggplot2主题
theme_set(theme_minimal(base_size = 12) +
            theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
                  plot.subtitle = element_text(hjust = 0.5, size = 11),
                  legend.position = "right",
                  panel.grid.minor = element_blank(),
                  panel.background = element_rect(fill = "white", color = NA),
                  plot.background = element_rect(fill = "white", color = NA)))


# ============================================================================
# 4.1 描述性统计
# ============================================================================

# --------------------------------------------------------------------------
# 4.1.1 各变量的统计量
# --------------------------------------------------------------------------

# 计算描述性统计量
descriptive_stats <- wine_data %>%
  select(-type) %>%
  summarise(across(everything(), 
                   list(Mean = ~mean(., na.rm = TRUE),
                        SD = ~sd(., na.rm = TRUE),
                        Min = ~min(., na.rm = TRUE),
                        Q25 = ~quantile(., 0.25, na.rm = TRUE),
                        Median = ~median(., na.rm = TRUE),
                        Q75 = ~quantile(., 0.75, na.rm = TRUE),
                        Max = ~max(., na.rm = TRUE)),
                   .names = "{.col}_{.fn}")) %>%
  pivot_longer(everything(), 
               names_to = c("Variable", "Statistic"),
               names_sep = "_",
               values_to = "Value") %>%
  pivot_wider(names_from = Statistic, values_from = Value)

print(descriptive_stats)

# 按红白葡萄酒分组的描述性统计
descriptive_by_type <- wine_data %>%
  group_by(type) %>%
  summarise(across(where(is.numeric), 
                   list(Mean = ~mean(., na.rm = TRUE),
                        SD = ~sd(., na.rm = TRUE)),
                   .names = "{.col}_{.fn}"))

print(descriptive_by_type)


# --------------------------------------------------------------------------
# 4.1.2 质量评分分布可视化
# --------------------------------------------------------------------------

# 质量评分分布直方图
p1 <- ggplot(wine_data, aes(x = quality)) +
  geom_histogram(binwidth = 1, fill = macaron_colors[3], 
                 color = "white", alpha = 0.8) +
  geom_density(aes(y = after_stat(count)), 
               color = macaron_colors[5], linewidth = 1.2) +
  labs(title = "Wine Quality Distribution",
       subtitle = "Histogram with Density Curve",
       x = "Quality Score", y = "Frequency") +
  scale_x_continuous(breaks = 3:9) +
  theme(panel.grid.major.x = element_blank())

print(p1)

# 按类型分组的质量评分分布
p2 <- ggplot(wine_data, aes(x = quality, fill = type)) +
  geom_histogram(binwidth = 1, position = "dodge", 
                 color = "white", alpha = 0.8) +
  scale_fill_manual(values = c(Red = macaron_red, White = macaron_white)) +
  labs(title = "Wine Quality Distribution by Type",
       subtitle = "Red vs White Wine Comparison",
       x = "Quality Score", y = "Frequency", fill = "Wine Type") +
  scale_x_continuous(breaks = 3:9)

print(p2)

# 质量评分的密度图（山脊图风格）
p3 <- ggplot(wine_data, aes(x = quality, y = type, fill = type)) +
  geom_density_ridges(alpha = 0.7, scale = 1.5) +
  scale_fill_manual(values = c(Red = macaron_red, White = macaron_white)) +
  labs(title = "Quality Distribution Density by Wine Type",
       x = "Quality Score", y = "Wine Type") +
  theme(legend.position = "none")

print(p3)


# --------------------------------------------------------------------------
# 4.1.3 红白葡萄酒对比 - 箱线图
# --------------------------------------------------------------------------

# 准备数据：将数据转换为长格式
wine_long <- wine_data %>%
  pivot_longer(cols = -c(type, quality), 
               names_to = "variable", 
               values_to = "value")

# 选择关键变量进行可视化
key_vars <- c("fixed.acidity", "volatile.acidity", "citric.acid", 
              "residual.sugar", "chlorides", "free.sulfur.dioxide",
              "total.sulfur.dioxide", "density", "pH", 
              "sulphates", "alcohol")

wine_long_filtered <- wine_long %>%
  filter(variable %in% key_vars)

# 箱线图：红白葡萄酒对比
p4 <- ggplot(wine_long_filtered, aes(x = type, y = value, fill = type)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.alpha = 0.5) +
  facet_wrap(~variable, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = c(Red = macaron_red, White = macaron_white)) +
  labs(title = "Comparison of Wine Characteristics",
       subtitle = "Red vs White Wine - Boxplot Analysis",
       x = "Wine Type", y = "Value", fill = "Wine Type") +
  theme(strip.background = element_rect(fill = macaron_colors[6], color = NA),
        strip.text = element_text(face = "bold"),
        axis.text.x = element_text(angle = 0))

print(p4)


# --------------------------------------------------------------------------
# 4.1.4 小提琴图：关键变量分布
# --------------------------------------------------------------------------

# 选择4个关键变量制作小提琴图
key_vars_violin <- c("alcohol", "volatile.acidity", "sulphates", "pH")

wine_violin <- wine_data %>%
  select(all_of(key_vars_violin), type, quality) %>%
  pivot_longer(cols = -c(type, quality), 
               names_to = "variable", 
               values_to = "value")

p5 <- ggplot(wine_violin, aes(x = type, y = value, fill = type)) +
  geom_violin(alpha = 0.7, trim = FALSE) +
  geom_boxplot(width = 0.2, alpha = 0.8, outlier.alpha = 0.3) +
  facet_wrap(~variable, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = c(Red = macaron_red, White = macaron_white)) +
  labs(title = "Distribution of Key Variables",
       subtitle = "Violin Plot with Boxplot Overlay",
       x = "Wine Type", y = "Value", fill = "Wine Type") +
  theme(strip.background = element_rect(fill = macaron_colors[4], color = NA),
        strip.text = element_text(face = "bold", size = 11))

print(p5)


# ============================================================================
# 4.2 相关性分析
# ============================================================================

# --------------------------------------------------------------------------
# 4.2.1 Pearson相关系数矩阵与显著性检验
# --------------------------------------------------------------------------

# 计算相关系数矩阵
cor_matrix <- cor(wine_data %>% select(-type), method = "pearson")
print(round(cor_matrix, 3))

# 相关性显著性检验
cor_test_results <- function(data) {
  vars <- names(data %>% select(-type))
  n <- length(vars)
  p_matrix <- matrix(NA, n, n)
  rownames(p_matrix) <- colnames(p_matrix) <- vars
  
  for(i in 1:n) {
    for(j in 1:n) {
      if(i != j) {
        test <- cor.test(data[[vars[i]]], data[[vars[j]]], method = "pearson")
        p_matrix[i, j] <- test$p.value
      } else {
        p_matrix[i, j] <- 0
      }
    }
  }
  return(p_matrix)
}

p_values <- cor_test_results(wine_data)
print("Significant correlations (p < 0.05):")
print(sum(p_values < 0.05 & p_values > 0, na.rm = TRUE) / 2)


# --------------------------------------------------------------------------
# 4.2.2 相关性热力图
# --------------------------------------------------------------------------

# 方法1：使用corrplot包
png("correlation_plot_1.png", width = 1200, height = 1000, res = 120)
corrplot(cor_matrix, 
         method = "circle",
         type = "upper",
         tl.col = "black",
         tl.srt = 45,
         tl.cex = 0.9,
         col = colorRampPalette(c(macaron_colors[1], "white", macaron_colors[3]))(200),
         addCoef.col = "black",
         number.cex = 0.6,
         title = "Pearson Correlation Matrix",
         mar = c(0, 0, 2, 0))
dev.off()

# 在RStudio中显示
corrplot(cor_matrix, 
         method = "circle",
         type = "upper",
         tl.col = "black",
         tl.srt = 45,
         tl.cex = 0.9,
         col = colorRampPalette(c(macaron_colors[1], "white", macaron_colors[3]))(200),
         addCoef.col = "black",
         number.cex = 0.6,
         title = "Pearson Correlation Matrix",
         mar = c(0, 0, 2, 0))

# 方法2：使用ggplot2制作热力图
cor_melted <- melt(cor_matrix)

p6 <- ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", value)), 
            size = 2.5, color = "black") +
  scale_fill_gradientn(colors = c(macaron_colors[1], "white", macaron_colors[3]),
                       limits = c(-1, 1),
                       name = "Correlation") +
  labs(title = "Correlation Heatmap of Wine Characteristics",
       subtitle = "Pearson Correlation Coefficients",
       x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
        axis.text.y = element_text(size = 9),
        panel.grid = element_blank(),
        legend.position = "right")

print(p6)


# --------------------------------------------------------------------------
# 4.2.3 散点图矩阵（关键变量）
# --------------------------------------------------------------------------

# 选择与quality相关性较高的变量
cor_with_quality <- cor_matrix[, "quality"]
top_vars <- names(sort(abs(cor_with_quality), decreasing = TRUE)[2:6])

wine_subset <- wine_data %>%
  select(all_of(c(top_vars, "quality", "type")))

# 使用GGally包创建散点图矩阵
p7 <- ggpairs(wine_subset,
              columns = 1:6,
              aes(color = type, alpha = 0.6),
              upper = list(continuous = wrap("cor", size = 3)),
              lower = list(continuous = wrap("points", alpha = 0.3, size = 0.5)),
              diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
              title = "Scatter Plot Matrix of Key Variables") +
  scale_color_manual(values = c(Red = macaron_red, White = macaron_white)) +
  scale_fill_manual(values = c(Red = macaron_red, White = macaron_white)) +
  theme(strip.background = element_rect(fill = macaron_colors[6], color = NA))

print(p7)


# ============================================================================
# 4.3 变量与质量关系
# ============================================================================

# --------------------------------------------------------------------------
# 4.3.1 关键变量识别
# --------------------------------------------------------------------------

# 计算与quality的相关系数并排序
quality_cor <- cor(wine_data %>% select(-type), method = "pearson")[, "quality"]
quality_cor_sorted <- sort(abs(quality_cor), decreasing = TRUE)

print("Variables ranked by correlation with quality:")
print(quality_cor_sorted)

# 可视化：与质量相关性的条形图
cor_df <- data.frame(
  variable = names(quality_cor_sorted)[-1],  # 排除quality本身
  correlation = quality_cor_sorted[-1]
) %>%
  mutate(direction = ifelse(quality_cor[variable] > 0, "Positive", "Negative"),
         abs_cor = abs(correlation))

p8 <- ggplot(cor_df, aes(x = reorder(variable, abs_cor), 
                         y = quality_cor[variable], 
                         fill = direction)) +
  geom_col(alpha = 0.8, width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c(Positive = macaron_colors[3], 
                               Negative = macaron_colors[1])) +
  labs(title = "Correlation of Variables with Wine Quality",
       subtitle = "Ranked by Absolute Correlation Coefficient",
       x = "Variables", y = "Pearson Correlation Coefficient",
       fill = "Direction") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  theme(panel.grid.major.x = element_line(color = "gray90"))

print(p8)


# --------------------------------------------------------------------------
# 4.3.2 分组箱线图：质量评分与关键变量
# --------------------------------------------------------------------------

# 将quality转换为因子以便分组
wine_data$quality_factor <- as.factor(wine_data$quality)

# 选择top 4关键变量
top4_vars <- names(quality_cor_sorted)[2:5]

wine_quality_long <- wine_data %>%
  select(all_of(top4_vars), quality_factor, type) %>%
  pivot_longer(cols = -c(quality_factor, type),
               names_to = "variable",
               values_to = "value")

p9 <- ggplot(wine_quality_long, aes(x = quality_factor, y = value, 
                                    fill = quality_factor)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  facet_wrap(~variable, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = colorRampPalette(macaron_gradient)(7)) +
  labs(title = "Key Variables Distribution by Quality Score",
       subtitle = "Boxplot Analysis of Top Correlated Variables",
       x = "Quality Score", y = "Value", fill = "Quality") +
  theme(strip.background = element_rect(fill = macaron_colors[6], color = NA),
        strip.text = element_text(face = "bold", size = 11),
        legend.position = "none")

print(p9)


# --------------------------------------------------------------------------
# 4.3.3 散点图（含拟合线）：关键变量与质量
# --------------------------------------------------------------------------

# 创建4个散点图，展示最相关的变量
create_scatter_with_fit <- function(data, var_name, color_pal) {
  ggplot(data, aes_string(x = var_name, y = "quality")) +
    geom_point(aes(color = type), alpha = 0.4, size = 1.5) +
    geom_smooth(method = "lm", se = TRUE, color = macaron_colors[5], 
                fill = macaron_colors[5], alpha = 0.2, linewidth = 1.2) +
    geom_smooth(aes(color = type), method = "lm", se = FALSE, 
                linewidth = 0.8, linetype = "dashed") +
    scale_color_manual(values = c(Red = macaron_red, White = macaron_white)) +
    labs(title = paste("Quality vs", gsub("\\.", " ", var_name)),
         x = gsub("\\.", " ", var_name),
         y = "Quality Score",
         color = "Wine Type") +
    theme(legend.position = "bottom")
}

# 生成4个散点图
scatter_plots <- lapply(top4_vars, function(var) {
  create_scatter_with_fit(wine_data, var, macaron_colors)
})

# 组合图表
p10 <- wrap_plots(scatter_plots, ncol = 2) +
  plot_annotation(
    title = "Relationship between Key Variables and Wine Quality",
    subtitle = "Scatter plots with linear regression lines",
    theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
                  plot.subtitle = element_text(hjust = 0.5, size = 11))
  )

print(p10)


# --------------------------------------------------------------------------
# 4.3.4 质量分组的密度图
# --------------------------------------------------------------------------

# 创建质量分组（低、中、高）
wine_data <- wine_data %>%
  mutate(quality_group = case_when(
    quality <= 5 ~ "Low (3-5)",
    quality == 6 ~ "Medium (6)",
    quality >= 7 ~ "High (7-9)"
  ),
  quality_group = factor(quality_group, 
                         levels = c("Low (3-5)", "Medium (6)", "High (7-9)")))

# 关键变量的密度图
wine_density <- wine_data %>%
  select(all_of(top4_vars[1:2]), quality_group) %>%
  pivot_longer(cols = -quality_group,
               names_to = "variable",
               values_to = "value")

p11 <- ggplot(wine_density, aes(x = value, fill = quality_group)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~variable, scales = "free", ncol = 1) +
  scale_fill_manual(values = c("Low (3-5)" = macaron_colors[1],
                               "Medium (6)" = macaron_colors[4],
                               "High (7-9)" = macaron_colors[3])) +
  labs(title = "Distribution of Key Variables by Quality Group",
       subtitle = "Density Plot Comparison",
       x = "Value", y = "Density", fill = "Quality Group") +
  theme(strip.background = element_rect(fill = macaron_colors[6], color = NA),
        strip.text = element_text(face = "bold"))

print(p11)


# ============================================================================
# 4.4 综合可视化面板
# ============================================================================

# 创建一个综合面板，展示主要发现
summary_plot1 <- ggplot(wine_data, aes(x = alcohol, y = quality, color = type)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2) +
  scale_color_manual(values = c(Red = macaron_red, White = macaron_white)) +
  labs(title = "Alcohol vs Quality", x = "Alcohol (%)", y = "Quality") +
  theme(legend.position = "none")

summary_plot2 <- ggplot(wine_data, aes(x = volatile.acidity, y = quality, color = type)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2) +
  scale_color_manual(values = c(Red = macaron_red, White = macaron_white)) +
  labs(title = "Volatile Acidity vs Quality", 
       x = "Volatile Acidity", y = "Quality") +
  theme(legend.position = "none")

summary_plot3 <- ggplot(wine_data, aes(x = quality_factor, fill = type)) +
  geom_bar(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c(Red = macaron_red, White = macaron_white)) +
  labs(title = "Quality Distribution by Type", 
       x = "Quality Score", y = "Count", fill = "Type")

summary_plot4 <- ggplot(wine_data, aes(x = type, y = alcohol, fill = type)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.2, alpha = 0.8) +
  scale_fill_manual(values = c(Red = macaron_red, White = macaron_white)) +
  labs(title = "Alcohol Content by Type", 
       x = "Wine Type", y = "Alcohol (%)") +
  theme(legend.position = "none")

# 组合面板
p12 <- (summary_plot1 | summary_plot2) / (summary_plot3 | summary_plot4) +
  plot_annotation(
    title = "Exploratory Data Analysis Summary",
    subtitle = "Key Findings from Wine Quality Dataset",
    theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
                  plot.subtitle = element_text(hjust = 0.5, size = 12))
  )

print(p12)


# ============================================================================
# 4.5 导出统计结果
# ============================================================================

# 导出描述性统计表
write.csv(descriptive_stats, "descriptive_statistics.csv", row.names = FALSE)
write.csv(descriptive_by_type, "descriptive_by_type.csv", row.names = FALSE)

# 导出相关系数矩阵
write.csv(cor_matrix, "correlation_matrix.csv")

# 导出与质量相关性排序
write.csv(cor_df, "quality_correlation_ranking.csv", row.names = FALSE)

# 打印总结信息
cat("\n========================================\n")
cat("第四章 探索性数据分析 - 完成\n")
cat("========================================\n")
cat("数据集信息:\n")
cat("- 总样本数:", nrow(wine_data), "\n")
cat("- 红葡萄酒:", sum(wine_data$type == "Red"), "\n")
cat("- 白葡萄酒:", sum(wine_data$type == "White"), "\n")
cat("- 变量数:", ncol(wine_data) - 3, "\n")
cat("\n关键发现:\n")
cat("- 与质量相关性最高的变量:", names(quality_cor_sorted)[2], 
    "(r =", round(quality_cor_sorted[2], 3), ")\n")
cat("- 质量评分范围:", min(wine_data$quality), "-", max(wine_data$quality), "\n")
cat("- 平均质量评分:", round(mean(wine_data$quality), 2), "\n")
cat("========================================\n")

