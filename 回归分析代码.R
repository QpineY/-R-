# ============================================================================
# 葡萄酒数据集回归分析完整代码
# ============================================================================

# 清空环境
rm(list = ls())

# 安装并加载必要的包
packages <- c("ggplot2", "corrplot", "car", "MASS", "leaps", "glmnet", 
              "gridExtra", "reshape2", "RColorBrewer", "dplyr", "caret")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 设置高饱和度配色方案
vibrant_colors <- c("#FF1744", "#00E676", "#2979FF", "#FFD600", 
                    "#F50057", "#00BFA5", "#651FFF", "#FF6D00")

# ============================================================================
# 1. 数据加载与预处理
# ============================================================================

# 从UCI加载葡萄酒数据集
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_data <- read.csv(url, header = FALSE)

# 添加列名
colnames(wine_data) <- c("Class", "Alcohol", "Malic_acid", "Ash", 
                         "Alcalinity_of_ash", "Magnesium", "Total_phenols",
                         "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
                         "Color_intensity", "Hue", "OD280_OD315", "Proline")

# 查看数据结构
print("数据集基本信息：")
str(wine_data)
summary(wine_data)

# 设置随机种子
set.seed(123)

# 划分训练集和测试集（80%训练，20%测试）
train_index <- createDataPartition(wine_data$Alcohol, p = 0.8, list = FALSE)
train_data <- wine_data[train_index, ]
test_data <- wine_data[-train_index, ]

cat("\n训练集样本数：", nrow(train_data), "\n")
cat("测试集样本数：", nrow(test_data), "\n\n")

# ============================================================================
# 6.1 多元线性回归建模
# ============================================================================

cat("=" , rep("=", 70), "\n", sep = "")
cat("6.1 多元线性回归建模\n")
cat("=" , rep("=", 70), "\n", sep = "")

# 以Alcohol（酒精含量）作为因变量，其他化学成分作为自变量
# 排除Class变量
formula_full <- Alcohol ~ Malic_acid + Ash + Alcalinity_of_ash + Magnesium + 
  Total_phenols + Flavanoids + Nonflavanoid_phenols + 
  Proanthocyanins + Color_intensity + Hue + OD280_OD315 + Proline

# 建立完整模型
model_full <- lm(formula_full, data = train_data)

# 输出模型摘要
print(summary(model_full))

# 图表1：相关性热图
cat("\n生成图表：相关性热图\n")
cor_matrix <- cor(train_data[, -1])  # 排除Class列

png("图1_相关性热图.png", width = 1200, height = 1000, res = 120)
corrplot(cor_matrix, method = "color", type = "upper", 
         col = colorRampPalette(c("#2979FF", "white", "#FF1744"))(200),
         tl.col = "black", tl.srt = 45, tl.cex = 0.9,
         addCoef.col = "black", number.cex = 0.7,
         title = "葡萄酒数据集变量相关性热图", 
         mar = c(0, 0, 2, 0))
dev.off()

# 图表2：实际值 vs 拟合值散点图
cat("生成图表：实际值vs拟合值散点图\n")
fitted_df <- data.frame(
  Actual = train_data$Alcohol,
  Fitted = fitted(model_full),
  Residuals = residuals(model_full)
)

p1 <- ggplot(fitted_df, aes(x = Actual, y = Fitted)) +
  geom_point(color = vibrant_colors[1], size = 3, alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, color = vibrant_colors[3], 
              linewidth = 1.5, linetype = "dashed") +
  labs(title = "实际值 vs 拟合值", 
       x = "实际酒精含量", 
       y = "拟合酒精含量") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        panel.grid.major = element_line(color = "grey90"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1))

ggsave("图2_实际值vs拟合值.png", p1, width = 10, height = 8, dpi = 300)

# 图表3：回归系数可视化
cat("生成图表：回归系数可视化\n")
coef_df <- data.frame(
  Variable = names(coef(model_full))[-1],
  Coefficient = coef(model_full)[-1],
  SE = summary(model_full)$coefficients[-1, 2]
)
coef_df$CI_lower <- coef_df$Coefficient - 1.96 * coef_df$SE
coef_df$CI_upper <- coef_df$Coefficient + 1.96 * coef_df$SE

p2 <- ggplot(coef_df, aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 1) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), 
                width = 0.3, color = vibrant_colors[2], linewidth = 1) +
  geom_point(size = 4, color = vibrant_colors[1]) +
  coord_flip() +
  labs(title = "回归系数及95%置信区间", 
       x = "变量", 
       y = "回归系数") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        panel.grid.major = element_line(color = "grey90"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1))

ggsave("图3_回归系数可视化.png", p2, width = 10, height = 8, dpi = 300)

# ============================================================================
# 6.2 模型诊断
# ============================================================================

cat("\n", rep("=", 72), "\n", sep = "")
cat("6.2 模型诊断\n")
cat(rep("=", 72), "\n", sep = "")

# 图表4：残差诊断四合一图
cat("\n生成图表：残差诊断四合一图\n")
png("图4_残差诊断四合一.png", width = 1400, height = 1400, res = 120)
par(mfrow = c(2, 2), col.main = "black", cex.main = 1.5, 
    col.lab = "black", cex.lab = 1.2)

# 1. 残差vs拟合值图
plot(model_full, which = 1, col = vibrant_colors[1], pch = 16, cex = 1.5,
     col.smooth = vibrant_colors[3], lwd = 2)

# 2. QQ图
plot(model_full, which = 2, col = vibrant_colors[2], pch = 16, cex = 1.5,
     col.lines = vibrant_colors[3], lwd = 2)

# 3. Scale-Location图
plot(model_full, which = 3, col = vibrant_colors[4], pch = 16, cex = 1.5,
     col.smooth = vibrant_colors[3], lwd = 2)

# 4. 残差vs杠杆值图
plot(model_full, which = 5, col = vibrant_colors[5], pch = 16, cex = 1.5,
     col.lines = vibrant_colors[3], lwd = 2)

dev.off()
par(mfrow = c(1, 1))

# 正态性检验
cat("\n正态性检验（Shapiro-Wilk检验）：\n")
shapiro_test <- shapiro.test(residuals(model_full))
print(shapiro_test)

# 方差齐性检验
cat("\nBreusch-Pagan检验（方差齐性）：\n")
bp_test <- ncvTest(model_full)
print(bp_test)

# 多重共线性检验
cat("\n方差膨胀因子（VIF）：\n")
vif_values <- vif(model_full)
print(vif_values)

# 图表5：VIF可视化
cat("\n生成图表：VIF可视化\n")
vif_df <- data.frame(
  Variable = names(vif_values),
  VIF = vif_values
)

p3 <- ggplot(vif_df, aes(x = reorder(Variable, VIF), y = VIF)) +
  geom_bar(stat = "identity", fill = vibrant_colors[3], alpha = 0.8) +
  geom_hline(yintercept = 5, linetype = "dashed", color = vibrant_colors[1], 
             linewidth = 1.2) +
  geom_hline(yintercept = 10, linetype = "dashed", color = vibrant_colors[5], 
             linewidth = 1.2) +
  annotate("text", x = 1, y = 5.5, label = "VIF = 5", color = vibrant_colors[1], 
           size = 4, fontface = "bold") +
  annotate("text", x = 1, y = 10.5, label = "VIF = 10", color = vibrant_colors[5], 
           size = 4, fontface = "bold") +
  coord_flip() +
  labs(title = "方差膨胀因子（VIF）", 
       x = "变量", 
       y = "VIF值") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        panel.grid.major = element_line(color = "grey90"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1))

ggsave("图5_VIF可视化.png", p3, width = 10, height = 8, dpi = 300)

# 异常值检测
cat("\nCook距离检测异常值：\n")
cooks_d <- cooks.distance(model_full)
influential <- which(cooks_d > 4/nrow(train_data))
cat("影响点索引：", influential, "\n")

# 图表6：Cook距离图
cat("\n生成图表：Cook距离图\n")
cooks_df <- data.frame(
  Index = 1:length(cooks_d),
  CooksD = cooks_d,
  Influential = cooks_d > 4/nrow(train_data)
)

p4 <- ggplot(cooks_df, aes(x = Index, y = CooksD, color = Influential)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_hline(yintercept = 4/nrow(train_data), linetype = "dashed", 
             color = vibrant_colors[1], linewidth = 1.2) +
  scale_color_manual(values = c("FALSE" = vibrant_colors[3], 
                                "TRUE" = vibrant_colors[1]),
                     labels = c("正常点", "影响点")) +
  labs(title = "Cook距离 - 异常值检测", 
       x = "观测值索引", 
       y = "Cook距离") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        legend.position = "top",
        legend.title = element_blank(),
        panel.grid.major = element_line(color = "grey90"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1))

ggsave("图6_Cook距离图.png", p4, width = 10, height = 8, dpi = 300)

# ============================================================================
# 6.3 变量选择与模型优化
# ============================================================================

cat("\n", rep("=", 72), "\n", sep = "")
cat("6.3 变量选择与模型优化\n")
cat(rep("=", 72), "\n", sep = "")

# 方法1：逐步回归
cat("\n【方法1】逐步回归（双向）：\n")
model_step <- stepAIC(model_full, direction = "both", trace = FALSE)
print(summary(model_step))

# 方法2：最优子集选择
cat("\n【方法2】最优子集选择：\n")
X_train <- train_data[, c("Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
                          "Total_phenols", "Flavanoids", "Nonflavanoid_phenols",
                          "Proanthocyanins", "Color_intensity", "Hue", 
                          "OD280_OD315", "Proline")]
y_train <- train_data$Alcohol

regfit_full <- regsubsets(formula_full, data = train_data, nvmax = 12)
reg_summary <- summary(regfit_full)

# 图表7：模型选择指标对比
cat("\n生成图表：模型选择指标对比\n")
metrics_df <- data.frame(
  NumVars = 1:12,
  RSS = reg_summary$rss,
  AdjR2 = reg_summary$adjr2,
  Cp = reg_summary$cp,
  BIC = reg_summary$bic
)

p5 <- ggplot(metrics_df, aes(x = NumVars)) +
  geom_line(aes(y = scale(RSS), color = "RSS"), linewidth = 1.5) +
  geom_point(aes(y = scale(RSS), color = "RSS"), size = 3) +
  geom_line(aes(y = scale(AdjR2), color = "Adj R²"), linewidth = 1.5) +
  geom_point(aes(y = scale(AdjR2), color = "Adj R²"), size = 3) +
  geom_line(aes(y = scale(Cp), color = "Cp"), linewidth = 1.5) +
  geom_point(aes(y = scale(Cp), color = "Cp"), size = 3) +
  geom_line(aes(y = scale(BIC), color = "BIC"), linewidth = 1.5) +
  geom_point(aes(y = scale(BIC), color = "BIC"), size = 3) +
  scale_color_manual(values = c("RSS" = vibrant_colors[1], 
                                "Adj R²" = vibrant_colors[2],
                                "Cp" = vibrant_colors[3], 
                                "BIC" = vibrant_colors[4])) +
  labs(title = "模型选择指标对比（标准化）", 
       x = "变量数量", 
       y = "标准化指标值",
       color = "指标") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        legend.position = "top",
        panel.grid.major = element_line(color = "grey90"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1))

ggsave("图7_模型选择指标对比.png", p5, width = 10, height = 8, dpi = 300)

# 选择最佳模型
best_adjr2 <- which.max(reg_summary$adjr2)
best_cp <- which.min(reg_summary$cp)
best_bic <- which.min(reg_summary$bic)

cat("\n最佳模型（基于Adjusted R²）：", best_adjr2, "个变量\n")
cat("最佳模型（基于Cp）：", best_cp, "个变量\n")
cat("最佳模型（基于BIC）：", best_bic, "个变量\n")

# 选择BIC最优模型
best_vars <- names(coef(regfit_full, best_bic))[-1]
cat("\n最优变量集合（BIC）：", paste(best_vars, collapse = ", "), "\n")

# 建立优化后的模型
formula_best <- as.formula(paste("Alcohol ~", paste(best_vars, collapse = " + ")))
model_best <- lm(formula_best, data = train_data)

cat("\n优化后模型摘要：\n")
print(summary(model_best))

# 方法3：LASSO回归
cat("\n【方法3】LASSO回归：\n")
X_matrix <- model.matrix(formula_full, train_data)[, -1]
y_vector <- train_data$Alcohol

cv_lasso <- cv.glmnet(X_matrix, y_vector, alpha = 1, nfolds = 10)
best_lambda <- cv_lasso$lambda.min
cat("最优lambda：", best_lambda, "\n")

model_lasso <- glmnet(X_matrix, y_vector, alpha = 1, lambda = best_lambda)
lasso_coef <- coef(model_lasso)
cat("\nLASSO回归系数：\n")
print(lasso_coef)

# 图表8：LASSO路径图
cat("\n生成图表：LASSO路径图\n")
png("图8_LASSO路径图.png", width = 1200, height = 900, res = 120)
lasso_full <- glmnet(X_matrix, y_vector, alpha = 1)
plot(lasso_full, xvar = "lambda", label = TRUE, lwd = 2,
     col = vibrant_colors[1:12])
abline(v = log(best_lambda), lty = 2, lwd = 2, col = vibrant_colors[7])
title("LASSO回归路径图", cex.main = 1.5)
legend("topright", legend = colnames(X_matrix), 
       col = vibrant_colors[1:12], lwd = 2, cex = 0.8)
dev.off()

# 图表9：交叉验证误差图
cat("生成图表：LASSO交叉验证误差图\n")
png("图9_LASSO交叉验证.png", width = 1200, height = 900, res = 120)
plot(cv_lasso, col = vibrant_colors[1], lwd = 2)
title("LASSO交叉验证误差", cex.main = 1.5)
dev.off()

# 模型对比
cat("\n", rep("-", 72), "\n", sep = "")
cat("模型对比总结：\n")
cat(rep("-", 72), "\n", sep = "")

models_comparison <- data.frame(
  Model = c("完整模型", "逐步回归", "最优子集", "LASSO"),
  R_squared = c(
    summary(model_full)$r.squared,
    summary(model_step)$r.squared,
    summary(model_best)$r.squared,
    cor(y_vector, predict(model_lasso, X_matrix))^2
  ),
  Adj_R_squared = c(
    summary(model_full)$adj.r.squared,
    summary(model_step)$adj.r.squared,
    summary(model_best)$adj.r.squared,
    NA
  ),
  AIC = c(
    AIC(model_full),
    AIC(model_step),
    AIC(model_best),
    NA
  ),
  BIC = c(
    BIC(model_full),
    BIC(model_step),
    BIC(model_best),
    NA
  ),
  Num_Vars = c(
    length(coef(model_full)) - 1,
    length(coef(model_step)) - 1,
    length(coef(model_best)) - 1,
    sum(lasso_coef != 0) - 1
  )
)

print(models_comparison)

# ============================================================================
# 6.4 模型验证
# ============================================================================

cat("\n", rep("=", 72), "\n", sep = "")
cat("6.4 模型验证\n")
cat(rep("=", 72), "\n", sep = "")

# 在测试集上进行预测
pred_full <- predict(model_full, newdata = test_data)
pred_step <- predict(model_step, newdata = test_data)
pred_best <- predict(model_best, newdata = test_data)

X_test <- model.matrix(formula_full, test_data)[, -1]
pred_lasso <- predict(model_lasso, newx = X_test)

# 计算评估指标
calculate_metrics <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predicted))
  r2 <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(c(MSE = mse, RMSE = rmse, MAE = mae, R2 = r2))
}

metrics_full <- calculate_metrics(test_data$Alcohol, pred_full)
metrics_step <- calculate_metrics(test_data$Alcohol, pred_step)
metrics_best <- calculate_metrics(test_data$Alcohol, pred_best)
metrics_lasso <- calculate_metrics(test_data$Alcohol, as.vector(pred_lasso))

cat("\n测试集性能评估：\n")
validation_results <- data.frame(
  Model = c("完整模型", "逐步回归", "最优子集", "LASSO"),
  rbind(metrics_full, metrics_step, metrics_best, metrics_lasso)
)
print(validation_results)

# 图表10：预测性能对比
cat("\n生成图表：预测性能对比\n")
validation_long <- melt(validation_results, id.vars = "Model")

p6 <- ggplot(validation_long, aes(x = Model, y = value, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  facet_wrap(~variable, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = vibrant_colors[1:4]) +
  labs(title = "测试集预测性能对比", 
       x = "模型", 
       y = "指标值") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none",
        panel.grid.major = element_line(color = "grey90"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        strip.text = element_text(face = "bold", size = 12))

ggsave("图10_预测性能对比.png", p6, width = 12, height = 10, dpi = 300)

# 图表11：实际值vs预测值（四模型对比）
cat("\n生成图表：实际值vs预测值四模型对比\n")
pred_df <- data.frame(
  Actual = rep(test_data$Alcohol, 4),
  Predicted = c(pred_full, pred_step, pred_best, as.vector(pred_lasso)),
  Model = rep(c("完整模型", "逐步回归", "最优子集", "LASSO"), 
              each = length(pred_full))
)

p7 <- ggplot(pred_df, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "black", linewidth = 1) +
  facet_wrap(~Model, ncol = 2) +
  scale_color_manual(values = vibrant_colors[1:4]) +
  labs(title = "实际值 vs 预测值（测试集）", 
       x = "实际酒精含量", 
       y = "预测酒精含量") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        legend.position = "none",
        panel.grid.major = element_line(color = "grey90"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        strip.text = element_text(face = "bold", size = 12))

ggsave("图11_实际vs预测四模型对比.png", p7, width = 12, height = 10, dpi = 300)

# 图表12：残差分布对比
cat("\n生成图表：残差分布对比\n")
residuals_df <- data.frame(
  Residuals = c(test_data$Alcohol - pred_full,
                test_data$Alcohol - pred_step,
                test_data$Alcohol - pred_best,
                test_data$Alcohol - as.vector(pred_lasso)),
  Model = rep(c("完整模型", "逐步回归", "最优子集", "LASSO"), 
              each = length(pred_full))
)

p8 <- ggplot(residuals_df, aes(x = Residuals, fill = Model)) +
  geom_histogram(bins = 15, alpha = 0.7, color = "black") +
  facet_wrap(~Model, ncol = 2, scales = "free_y") +
  scale_fill_manual(values = vibrant_colors[1:4]) +
  geom_vline(xintercept = 0, linetype = "dashed", 
             color = "black", linewidth = 1) +
  labs(title = "残差分布对比（测试集）", 
       x = "残差", 
       y = "频数") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        legend.position = "none",
        panel.grid.major = element_line(color = "grey90"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        strip.text = element_text(face = "bold", size = 12))

ggsave("图12_残差分布对比.png", p8, width = 12, height = 10, dpi = 300)

# K折交叉验证
cat("\n【K折交叉验证】（10折）：\n")
train_control <- trainControl(method = "cv", number = 10)

cv_full <- train(formula_full, data = train_data, method = "lm", 
                 trControl = train_control)
cv_best <- train(formula_best, data = train_data, method = "lm", 
                 trControl = train_control)

cat("\n完整模型交叉验证结果：\n")
print(cv_full$results)

cat("\n最优模型交叉验证结果：\n")
print(cv_best$results)

# 图表13：交叉验证结果可视化
cat("\n生成图表：交叉验证结果\n")
cv_results <- data.frame(
  Model = c("完整模型", "最优子集"),
  RMSE = c(cv_full$results$RMSE, cv_best$results$RMSE),
  Rsquared = c(cv_full$results$Rsquared, cv_best$results$Rsquared),
  MAE = c(cv_full$results$MAE, cv_best$results$MAE)
)

cv_long <- melt(cv_results, id.vars = "Model")

p9 <- ggplot(cv_long, aes(x = Model, y = value, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  facet_wrap(~variable, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = vibrant_colors[c(1, 3)]) +
  labs(title = "10折交叉验证结果对比", 
       x = "模型", 
       y = "指标值") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        legend.position = "none",
        panel.grid.major = element_line(color = "grey90"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        strip.text = element_text(face = "bold", size = 12))

ggsave("图13_交叉验证结果.png", p9, width = 14, height = 6, dpi = 300)

# ============================================================================
# 总结报告
# ============================================================================

cat("\n", rep("=", 72), "\n", sep = "")
cat("分析总结\n")
cat(rep("=", 72), "\n", sep = "")

cat("\n1. 数据集信息：\n")
cat("   - 样本总数：", nrow(wine_data), "\n")
cat("   - 训练集：", nrow(train_data), "样本\n")
cat("   - 测试集：", nrow(test_data), "样本\n")
cat("   - 特征数：", ncol(wine_data) - 1, "\n")

cat("\n2. 最佳模型推荐：\n")
best_model_idx <- which.min(validation_results$RMSE)
cat("   根据测试集RMSE，推荐使用：", 
    as.character(validation_results$Model[best_model_idx]), "\n")
cat("   - RMSE：", round(validation_results$RMSE[best_model_idx], 4), "\n")
cat("   - R²：", round(validation_results$R2[best_model_idx], 4), "\n")
cat("   - MAE：", round(validation_results$MAE[best_model_idx], 4), "\n")

cat("\n3. 生成的图表清单：\n")
cat("   图1：相关性热图\n")
cat("   图2：实际值vs拟合值散点图\n")
cat("   图3：回归系数可视化\n")
cat("   图4：残差诊断四合一图\n")
cat("   图5：VIF可视化\n")
cat("   图6：Cook距离图\n")
cat("   图7：模型选择指标对比\n")
cat("   图8：LASSO路径图\n")
cat("   图9：LASSO交叉验证误差图\n")
cat("   图10：预测性能对比\n")
cat("   图11：实际vs预测四模型对比\n")
cat("   图12：残差分布对比\n")
cat("   图13：交叉验证结果\n")

cat("\n", rep("=", 72), "\n", sep = "")
cat("分析完成！所有图表已保存至工作目录。\n")
cat(rep("=", 72), "\n", sep = "")

# 保存工作空间
save.image("wine_regression_analysis.RData")
cat("\n工作空间已保存为：wine_regression_analysis.RData\n")
