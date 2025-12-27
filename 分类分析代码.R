# ============================================================================
# UCI葡萄酒数据集 - 分类分析完整代码
# 章节7: 分类分析
# ============================================================================

# 清空环境
rm(list = ls())

# 安装并加载必要的包
packages <- c("ggplot2", "randomForest", "caret", "pROC", "gridExtra", 
              "reshape2", "RColorBrewer", "scales", "dplyr", "corrplot", "nnet")

for(pkg in packages){
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 设置高饱和度配色方案
colors_high_sat <- c("#FF1744", "#00E676", "#2979FF", "#FFD600", 
                     "#E040FB", "#00E5FF", "#FF6E40", "#76FF03")

# ============================================================================
# 7.1 数据准备
# ============================================================================

# 加载UCI葡萄酒数据集
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_data <- read.csv(url, header = FALSE)

# 设置列名
colnames(wine_data) <- c("Class", "Alcohol", "Malic_acid", "Ash", 
                         "Alcalinity_of_ash", "Magnesium", "Total_phenols",
                         "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
                         "Color_intensity", "Hue", "OD280_OD315", "Proline")

# 将类别转换为因子
wine_data$Class <- as.factor(wine_data$Class)

# 数据集概览
cat("数据集维度:", dim(wine_data), "\n")
cat("类别分布:\n")
print(table(wine_data$Class))

# 数据标准化（保留类别列）
wine_scaled <- wine_data
wine_scaled[, -1] <- scale(wine_data[, -1])

# 划分训练集和测试集（70%训练，30%测试）
set.seed(123)
train_index <- createDataPartition(wine_scaled$Class, p = 0.7, list = FALSE)
train_data <- wine_scaled[train_index, ]
test_data <- wine_scaled[-train_index, ]

cat("\n训练集样本数:", nrow(train_data), "\n")
cat("测试集样本数:", nrow(test_data), "\n")

# 图表1: 类别分布图
class_dist <- as.data.frame(table(wine_data$Class))
colnames(class_dist) <- c("Class", "Count")

p1 <- ggplot(class_dist, aes(x = Class, y = Count, fill = Class)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = Count), vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = colors_high_sat[1:3]) +
  labs(title = "葡萄酒类别分布", x = "类别", y = "样本数量") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        legend.position = "none",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14, face = "bold"))

print(p1)

# 图表2: 特征相关性热图
cor_matrix <- cor(wine_data[, -1])
png("correlation_heatmap.png", width = 1200, height = 1000, res = 120)
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, tl.cex = 0.8,
         col = colorRampPalette(c("#2979FF", "white", "#FF1744"))(200),
         addCoef.col = "black", number.cex = 0.6,
         title = "特征相关性热图", mar = c(0,0,2,0))
dev.off()
cat("\n相关性热图已保存为 correlation_heatmap.png\n")

# ============================================================================
# 7.2 逻辑回归
# ============================================================================

cat("\n" , rep("=", 60), "\n", sep = "")
cat("7.2 逻辑回归分析\n")
cat(rep("=", 60), "\n", sep = "")

# 训练多分类逻辑回归模型
logit_model <- nnet::multinom(Class ~ ., data = train_data, trace = FALSE, maxit = 500)

# 预测
logit_pred_train <- predict(logit_model, train_data)
logit_pred_test <- predict(logit_model, test_data)

# 预测概率
logit_prob_test <- predict(logit_model, test_data, type = "probs")

# 混淆矩阵
logit_cm_train <- confusionMatrix(logit_pred_train, train_data$Class)
logit_cm_test <- confusionMatrix(logit_pred_test, test_data$Class)

cat("\n逻辑回归 - 训练集准确率:", round(logit_cm_train$overall['Accuracy'], 4), "\n")
cat("逻辑回归 - 测试集准确率:", round(logit_cm_test$overall['Accuracy'], 4), "\n")

# 图表3: 逻辑回归混淆矩阵热图
cm_logit <- as.data.frame(logit_cm_test$table)
colnames(cm_logit) <- c("Prediction", "Reference", "Freq")

p2 <- ggplot(cm_logit, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", size = 1.5) +
  geom_text(aes(label = Freq), size = 8, fontface = "bold", color = "white") +
  scale_fill_gradient(low = "#00E5FF", high = "#E040FB", 
                      name = "样本数", limits = c(0, max(cm_logit$Freq))) +
  labs(title = "逻辑回归 - 混淆矩阵", x = "真实类别", y = "预测类别") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 12, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"))

print(p2)

# 图表4: 逻辑回归各类别性能指标
logit_metrics <- data.frame(
  Class = rownames(logit_cm_test$byClass),
  Sensitivity = logit_cm_test$byClass[, "Sensitivity"],
  Specificity = logit_cm_test$byClass[, "Specificity"],
  Precision = logit_cm_test$byClass[, "Pos Pred Value"]
)
logit_metrics$Class <- gsub("Class: ", "", logit_metrics$Class)

logit_metrics_long <- melt(logit_metrics, id.vars = "Class")

p3 <- ggplot(logit_metrics_long, aes(x = Class, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = round(value, 3)), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = colors_high_sat[c(1, 2, 3)],
                    labels = c("灵敏度", "特异度", "精确度")) +
  labs(title = "逻辑回归 - 各类别性能指标", 
       x = "类别", y = "指标值", fill = "指标") +
  ylim(0, 1.1) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11))

print(p3)

# ============================================================================
# 7.3 随机森林
# ============================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("7.3 随机森林分析\n")
cat(rep("=", 60), "\n", sep = "")

# 训练随机森林模型
set.seed(123)
rf_model <- randomForest(Class ~ ., data = train_data, 
                         ntree = 500, mtry = 3, importance = TRUE)

# 预测
rf_pred_train <- predict(rf_model, train_data)
rf_pred_test <- predict(rf_model, test_data)

# 预测概率
rf_prob_test <- predict(rf_model, test_data, type = "prob")

# 混淆矩阵
rf_cm_train <- confusionMatrix(rf_pred_train, train_data$Class)
rf_cm_test <- confusionMatrix(rf_pred_test, test_data$Class)

cat("\n随机森林 - 训练集准确率:", round(rf_cm_train$overall['Accuracy'], 4), "\n")
cat("随机森林 - 测试集准确率:", round(rf_cm_test$overall['Accuracy'], 4), "\n")
cat("随机森林 - OOB误差率:", round(mean(rf_model$err.rate[, "OOB"]), 4), "\n")

# 图表5: 随机森林混淆矩阵热图
cm_rf <- as.data.frame(rf_cm_test$table)
colnames(cm_rf) <- c("Prediction", "Reference", "Freq")

p4 <- ggplot(cm_rf, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", size = 1.5) +
  geom_text(aes(label = Freq), size = 8, fontface = "bold", color = "white") +
  scale_fill_gradient(low = "#FFD600", high = "#FF1744", 
                      name = "样本数", limits = c(0, max(cm_rf$Freq))) +
  labs(title = "随机森林 - 混淆矩阵", x = "真实类别", y = "预测类别") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 12, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"))

print(p4)

# 图表6: 随机森林各类别性能指标
rf_metrics <- data.frame(
  Class = rownames(rf_cm_test$byClass),
  Sensitivity = rf_cm_test$byClass[, "Sensitivity"],
  Specificity = rf_cm_test$byClass[, "Specificity"],
  Precision = rf_cm_test$byClass[, "Pos Pred Value"]
)
rf_metrics$Class <- gsub("Class: ", "", rf_metrics$Class)

rf_metrics_long <- melt(rf_metrics, id.vars = "Class")

p5 <- ggplot(rf_metrics_long, aes(x = Class, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = round(value, 3)), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = colors_high_sat[c(5, 6, 7)],
                    labels = c("灵敏度", "特异度", "精确度")) +
  labs(title = "随机森林 - 各类别性能指标", 
       x = "类别", y = "指标值", fill = "指标") +
  ylim(0, 1.1) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11))

print(p5)

# 图表7: 特征重要性图
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  MeanDecreaseAccuracy = importance(rf_model)[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini = importance(rf_model)[, "MeanDecreaseGini"]
)

importance_df <- importance_df[order(-importance_df$MeanDecreaseAccuracy), ]
importance_df$Feature <- factor(importance_df$Feature, 
                                levels = importance_df$Feature)

p6 <- ggplot(importance_df, aes(x = Feature, y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = colors_high_sat[1], width = 0.7) +
  geom_text(aes(label = round(MeanDecreaseAccuracy, 2)), 
            hjust = -0.2, size = 3.5, fontface = "bold") +
  coord_flip() +
  labs(title = "随机森林 - 特征重要性排序", 
       x = "特征", y = "平均准确率下降") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 14, face = "bold"))

print(p6)

# 图表8: OOB误差率变化曲线
oob_error <- data.frame(
  Trees = 1:rf_model$ntree,
  OOB = rf_model$err.rate[, "OOB"],
  Class1 = rf_model$err.rate[, "1"],
  Class2 = rf_model$err.rate[, "2"],
  Class3 = rf_model$err.rate[, "3"]
)

oob_error_long <- melt(oob_error, id.vars = "Trees")

p7 <- ggplot(oob_error_long, aes(x = Trees, y = value, color = variable)) +
  geom_line(size = 1.2) +
  scale_color_manual(values = colors_high_sat[1:4],
                     labels = c("总体OOB", "类别1", "类别2", "类别3")) +
  labs(title = "随机森林 - OOB误差率变化", 
       x = "决策树数量", y = "误差率", color = "类别") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11),
        legend.position = "right")

print(p7)

# ============================================================================
# 7.4 模型对比
# ============================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("7.4 模型对比分析\n")
cat(rep("=", 60), "\n", sep = "")

# 图表9: 模型准确率对比（训练集 vs 测试集）
accuracy_comparison <- data.frame(
  Model = rep(c("逻辑回归", "随机森林"), each = 2),
  Dataset = rep(c("训练集", "测试集"), 2),
  Accuracy = c(
    logit_cm_train$overall['Accuracy'],
    logit_cm_test$overall['Accuracy'],
    rf_cm_train$overall['Accuracy'],
    rf_cm_test$overall['Accuracy']
  )
)

p8 <- ggplot(accuracy_comparison, aes(x = Model, y = Accuracy, fill = Dataset)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = paste0(round(Accuracy * 100, 2), "%")), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = colors_high_sat[c(2, 5)]) +
  labs(title = "模型准确率对比", x = "模型", y = "准确率", fill = "数据集") +
  ylim(0, 1.1) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11))

print(p8)

# 图表10: 综合性能指标对比
# 计算各模型的综合指标
logit_overall <- data.frame(
  Model = "逻辑回归",
  Accuracy = logit_cm_test$overall['Accuracy'],
  Kappa = logit_cm_test$overall['Kappa'],
  Sensitivity = mean(logit_cm_test$byClass[, "Sensitivity"], na.rm = TRUE),
  Specificity = mean(logit_cm_test$byClass[, "Specificity"], na.rm = TRUE),
  Precision = mean(logit_cm_test$byClass[, "Pos Pred Value"], na.rm = TRUE)
)

rf_overall <- data.frame(
  Model = "随机森林",
  Accuracy = rf_cm_test$overall['Accuracy'],
  Kappa = rf_cm_test$overall['Kappa'],
  Sensitivity = mean(rf_cm_test$byClass[, "Sensitivity"], na.rm = TRUE),
  Specificity = mean(rf_cm_test$byClass[, "Specificity"], na.rm = TRUE),
  Precision = mean(rf_cm_test$byClass[, "Pos Pred Value"], na.rm = TRUE)
)

overall_comparison <- rbind(logit_overall, rf_overall)
overall_comparison_long <- melt(overall_comparison, id.vars = "Model")

p9 <- ggplot(overall_comparison_long, aes(x = variable, y = value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = round(value, 3)), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = colors_high_sat[c(3, 1)]) +
  scale_x_discrete(labels = c("准确率", "Kappa系数", "灵敏度", "特异度", "精确度")) +
  labs(title = "模型综合性能指标对比", x = "性能指标", y = "指标值", fill = "模型") +
  ylim(0, 1.1) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text.x = element_text(size = 11, angle = 15, hjust = 1),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11))

print(p9)

# 图表11: ROC曲线对比（以类别1为例）
# 逻辑回归ROC
logit_roc <- roc(as.numeric(test_data$Class == "1"), logit_prob_test[, 1], 
                 quiet = TRUE)

# 随机森林ROC
rf_roc <- roc(as.numeric(test_data$Class == "1"), rf_prob_test[, 1], 
              quiet = TRUE)

# 绘制ROC曲线
roc_data <- data.frame(
  FPR = c(1 - logit_roc$specificities, 1 - rf_roc$specificities),
  TPR = c(logit_roc$sensitivities, rf_roc$sensitivities),
  Model = c(rep("逻辑回归", length(logit_roc$specificities)),
            rep("随机森林", length(rf_roc$specificities)))
)

p10 <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "gray50", size = 1) +
  scale_color_manual(values = colors_high_sat[c(3, 1)]) +
  annotate("text", x = 0.6, y = 0.3, 
           label = paste0("逻辑回归 AUC = ", round(auc(logit_roc), 3)),
           size = 5, fontface = "bold", color = colors_high_sat[3]) +
  annotate("text", x = 0.6, y = 0.2, 
           label = paste0("随机森林 AUC = ", round(auc(rf_roc), 3)),
           size = 5, fontface = "bold", color = colors_high_sat[1]) +
  labs(title = "ROC曲线对比 (类别1)", 
       x = "假阳性率 (FPR)", y = "真阳性率 (TPR)", color = "模型") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11),
        legend.position = c(0.8, 0.2))

print(p10)

# 图表12: 各类别AUC对比
auc_comparison <- data.frame(
  Class = rep(c("类别1", "类别2", "类别3"), 2),
  Model = rep(c("逻辑回归", "随机森林"), each = 3),
  AUC = c(
    auc(roc(as.numeric(test_data$Class == "1"), logit_prob_test[, 1], quiet = TRUE)),
    auc(roc(as.numeric(test_data$Class == "2"), logit_prob_test[, 2], quiet = TRUE)),
    auc(roc(as.numeric(test_data$Class == "3"), logit_prob_test[, 3], quiet = TRUE)),
    auc(roc(as.numeric(test_data$Class == "1"), rf_prob_test[, 1], quiet = TRUE)),
    auc(roc(as.numeric(test_data$Class == "2"), rf_prob_test[, 2], quiet = TRUE)),
    auc(roc(as.numeric(test_data$Class == "3"), rf_prob_test[, 3], quiet = TRUE))
  )
)

p11 <- ggplot(auc_comparison, aes(x = Class, y = AUC, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = round(AUC, 3)), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = colors_high_sat[c(6, 7)]) +
  labs(title = "各类别AUC值对比", x = "类别", y = "AUC值", fill = "模型") +
  ylim(0, 1.1) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11))

print(p11)

# ============================================================================
# 结果汇总表格
# ============================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("模型性能汇总表\n")
cat(rep("=", 60), "\n", sep = "")

summary_table <- data.frame(
  模型 = c("逻辑回归", "随机森林"),
  训练集准确率 = c(
    round(logit_cm_train$overall['Accuracy'], 4),
    round(rf_cm_train$overall['Accuracy'], 4)
  ),
  测试集准确率 = c(
    round(logit_cm_test$overall['Accuracy'], 4),
    round(rf_cm_test$overall['Accuracy'], 4)
  ),
  Kappa系数 = c(
    round(logit_cm_test$overall['Kappa'], 4),
    round(rf_cm_test$overall['Kappa'], 4)
  ),
  平均灵敏度 = c(
    round(mean(logit_cm_test$byClass[, "Sensitivity"], na.rm = TRUE), 4),
    round(mean(rf_cm_test$byClass[, "Sensitivity"], na.rm = TRUE), 4)
  ),
  平均特异度 = c(
    round(mean(logit_cm_test$byClass[, "Specificity"], na.rm = TRUE), 4),
    round(mean(rf_cm_test$byClass[, "Specificity"], na.rm = TRUE), 4)
  ),
  平均精确度 = c(
    round(mean(logit_cm_test$byClass[, "Pos Pred Value"], na.rm = TRUE), 4),
    round(mean(rf_cm_test$byClass[, "Pos Pred Value"], na.rm = TRUE), 4)
  )
)

print(summary_table)

cat("\n分析完成！所有图表已生成。\n")
cat("建议将图表依次插入报告的对应章节中。\n")

# ============================================================================
# 保存结果
# ============================================================================

# 保存模型
save(logit_model, rf_model, file = "classification_models.RData")
cat("\n模型已保存至 classification_models.RData\n")

# 保存汇总表格
write.csv(summary_table, "model_summary.csv", row.names = FALSE, fileEncoding = "UTF-8")
cat("汇总表格已保存至 model_summary.csv\n")

cat("\n", rep("=", 60), "\n", sep = "")
cat("所有分析完成！\n")
cat(rep("=", 60), "\n", sep = "")
