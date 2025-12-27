# ============================================================================
# 葡萄酒数据集聚类分析（完全兼容版）
# ============================================================================

# 安装并加载必要的包（只使用基础包）
packages <- c("ggplot2", "factoextra", "cluster", "NbClust", 
              "gridExtra", "reshape2", "fmsb")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# ============================================================================
# 自定义函数：手动实现缺失的函数
# ============================================================================

# 1. 调整兰德指数（ARI）
adjusted_rand_index <- function(cluster1, cluster2) {
  # 构建列联表
  tab <- table(cluster1, cluster2)
  
  # 计算各种组合数
  n <- sum(tab)
  ni_dot <- rowSums(tab)
  n_dot_j <- colSums(tab)
  
  # 计算组合数
  sum_comb_tab <- sum(choose(tab, 2))
  sum_comb_ni <- sum(choose(ni_dot, 2))
  sum_comb_nj <- sum(choose(n_dot_j, 2))
  
  # 计算期望值
  expected_index <- sum_comb_ni * sum_comb_nj / choose(n, 2)
  max_index <- (sum_comb_ni + sum_comb_nj) / 2
  
  # 计算ARI
  if (max_index == expected_index) {
    return(0)
  } else {
    ari <- (sum_comb_tab - expected_index) / (max_index - expected_index)
    return(ari)
  }
}

# 2. 标准化互信息（NMI）
normalized_mutual_info <- function(cluster1, cluster2) {
  # 构建列联表
  tab <- table(cluster1, cluster2)
  n <- sum(tab)
  
  # 计算熵
  entropy <- function(labels) {
    probs <- table(labels) / length(labels)
    -sum(probs * log(probs + 1e-10))
  }
  
  # 计算互信息
  mi <- 0
  for (i in 1:nrow(tab)) {
    for (j in 1:ncol(tab)) {
      if (tab[i, j] > 0) {
        mi <- mi + (tab[i, j] / n) * log((tab[i, j] * n) / 
                                           (sum(tab[i, ]) * sum(tab[, j])) + 1e-10)
      }
    }
  }
  
  # 计算NMI
  h1 <- entropy(cluster1)
  h2 <- entropy(cluster2)
  
  if (h1 == 0 || h2 == 0) {
    return(0)
  } else {
    nmi <- mi / sqrt(h1 * h2)
    return(nmi)
  }
}

# 3. Dunn指数
calculate_dunn_index <- function(data, clusters) {
  dist_matrix <- as.matrix(dist(data))
  
  # 计算类内最大距离
  max_intra <- 0
  for (k in unique(clusters)) {
    cluster_points <- which(clusters == k)
    if (length(cluster_points) > 1) {
      cluster_dist <- dist_matrix[cluster_points, cluster_points]
      max_intra <- max(max_intra, max(cluster_dist))
    }
  }
  
  # 计算类间最小距离
  min_inter <- Inf
  cluster_ids <- unique(clusters)
  for (i in 1:(length(cluster_ids)-1)) {
    for (j in (i+1):length(cluster_ids)) {
      points_i <- which(clusters == cluster_ids[i])
      points_j <- which(clusters == cluster_ids[j])
      inter_dist <- dist_matrix[points_i, points_j]
      min_inter <- min(min_inter, min(inter_dist))
    }
  }
  
  # Dunn指数
  return(min_inter / max_intra)
}

# 4. Davies-Bouldin指数
calculate_db_index <- function(data, clusters, centers) {
  n_clusters <- length(unique(clusters))
  
  # 计算每个聚类的平均距离（散度）
  S <- numeric(n_clusters)
  for (k in 1:n_clusters) {
    cluster_points <- data[clusters == k, , drop = FALSE]
    if (nrow(cluster_points) > 0) {
      S[k] <- mean(sqrt(rowSums((cluster_points - 
                                   matrix(centers[k,], 
                                          nrow = nrow(cluster_points), 
                                          ncol = ncol(centers), 
                                          byrow = TRUE))^2)))
    }
  }
  
  # 计算Davies-Bouldin指数
  DB <- 0
  for (i in 1:n_clusters) {
    max_ratio <- 0
    for (j in 1:n_clusters) {
      if (i != j) {
        M_ij <- sqrt(sum((centers[i,] - centers[j,])^2))
        ratio <- (S[i] + S[j]) / M_ij
        max_ratio <- max(max_ratio, ratio)
      }
    }
    DB <- DB + max_ratio
  }
  return(DB / n_clusters)
}

# 设置高饱和度配色方案
my_colors <- c("#FF1744", "#00E676", "#2979FF", "#FFD600", 
               "#E040FB", "#00E5FF", "#FF6E40", "#76FF03")

# ============================================================================
# 1. 数据加载与预处理
# ============================================================================

cat("========== 数据加载与预处理 ==========\n")

# 从UCI加载葡萄酒数据集
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_data <- read.csv(url, header = FALSE)

# 添加列名
colnames(wine_data) <- c("Class", "Alcohol", "Malic_acid", "Ash", 
                         "Alcalinity_of_ash", "Magnesium", "Total_phenols",
                         "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
                         "Color_intensity", "Hue", "OD280_OD315", "Proline")

# 保存真实类别标签
true_labels <- wine_data$Class

# 提取特征数据（去除类别列）
wine_features <- wine_data[, -1]

# 数据标准化（Z-score标准化）
wine_scaled <- scale(wine_features)

cat("数据维度:", dim(wine_scaled), "\n")
cat("真实类别分布:\n")
print(table(true_labels))

# ============================================================================
# 6.1 最优聚类数确定
# ============================================================================

cat("\n========== 6.1 最优聚类数确定 ==========\n")

# 方法1: 肘部法则（Elbow Method）
set.seed(123)
wss <- numeric(10)
for (i in 1:10) {
  kmeans_model <- kmeans(wine_scaled, centers = i, nstart = 25)
  wss[i] <- kmeans_model$tot.withinss
}

# 绘制肘部法则图
elbow_plot <- ggplot(data.frame(k = 1:10, wss = wss), aes(x = k, y = wss)) +
  geom_line(color = "#FF1744", linewidth = 1.2) +
  geom_point(color = "#FF1744", size = 4, shape = 19) +
  geom_point(data = data.frame(k = 3, wss = wss[3]), 
             aes(x = k, y = wss), color = "#FFD600", size = 6, shape = 18) +
  labs(title = "肘部法则确定最优聚类数",
       subtitle = "寻找WSS下降速率变缓的拐点",
       x = "聚类数 K", 
       y = "组内平方和 (WSS)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(fill = NA, color = "grey70")) +
  scale_x_continuous(breaks = 1:10) +
  annotate("text", x = 3, y = wss[3] + 20, label = "最优K=3", 
           color = "#FFD600", fontface = "bold", size = 5)

print(elbow_plot)

# 方法2: 轮廓系数法（Silhouette Method）
cat("\n计算轮廓系数...\n")
silhouette_scores <- numeric(9)
for (i in 2:10) {
  kmeans_model <- kmeans(wine_scaled, centers = i, nstart = 25)
  sil <- silhouette(kmeans_model$cluster, dist(wine_scaled))
  silhouette_scores[i-1] <- mean(sil[, 3])
  cat("K =", i, ", 平均轮廓系数 =", round(silhouette_scores[i-1], 4), "\n")
}

silhouette_plot <- ggplot(data.frame(k = 2:10, score = silhouette_scores), 
                          aes(x = k, y = score)) +
  geom_line(color = "#00E676", linewidth = 1.2) +
  geom_point(color = "#00E676", size = 4, shape = 19) +
  geom_point(data = data.frame(k = 3, score = silhouette_scores[2]), 
             aes(x = k, y = score), color = "#E040FB", size = 6, shape = 18) +
  labs(title = "轮廓系数法确定最优聚类数",
       subtitle = "轮廓系数越高，聚类质量越好",
       x = "聚类数 K", 
       y = "平均轮廓系数") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(fill = NA, color = "grey70")) +
  scale_x_continuous(breaks = 2:10) +
  annotate("text", x = 3, y = silhouette_scores[2] + 0.02, 
           label = "最优K=3", color = "#E040FB", fontface = "bold", size = 5)

print(silhouette_plot)

# 方法3: Gap统计量法
cat("\n计算Gap统计量...\n")
set.seed(123)
gap_stat <- clusGap(wine_scaled, FUN = kmeans, nstart = 25, 
                    K.max = 10, B = 50)

gap_plot <- fviz_gap_stat(gap_stat) +
  labs(title = "Gap统计量法确定最优聚类数",
       subtitle = "Gap统计量最大值对应的K为最优聚类数") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        panel.grid.minor = element_blank())

print(gap_plot)

# 方法4: NbClust综合评估
cat("\n使用NbClust进行综合评估...\n")
set.seed(123)
nb_result <- tryCatch({
  NbClust(wine_scaled, distance = "euclidean", 
          min.nc = 2, max.nc = 10, 
          method = "kmeans", index = "hartigan")
}, error = function(e) {
  cat("NbClust计算出错，跳过\n")
  NULL
})

if (!is.null(nb_result)) {
  cat("NbClust推荐的最优聚类数:", nb_result$Best.nc[1], "\n")
}

# ============================================================================
# 6.2 K-means聚类
# ============================================================================

cat("\n========== 6.2 K-means聚类 ==========\n")

# 使用最优聚类数K=3进行聚类
set.seed(123)
kmeans_final <- kmeans(wine_scaled, centers = 3, nstart = 50, iter.max = 100)

cat("聚类结果:\n")
cat("各聚类样本数:", kmeans_final$size, "\n")
cat("组间平方和:", round(kmeans_final$betweenss, 2), "\n")
cat("组内平方和:", round(kmeans_final$tot.withinss, 2), "\n")
cat("总平方和:", round(kmeans_final$totss, 2), "\n")
cat("R²（解释方差比）:", round(kmeans_final$betweenss / kmeans_final$totss, 4), "\n")

# 图1: PCA降维可视化聚类结果
pca_result <- prcomp(wine_scaled)
pca_data <- data.frame(PC1 = pca_result$x[, 1],
                       PC2 = pca_result$x[, 2],
                       Cluster = as.factor(kmeans_final$cluster),
                       True_Class = as.factor(true_labels))

variance_explained <- summary(pca_result)$importance[2, 1:2] * 100

pca_cluster_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 4, alpha = 0.8) +
  stat_ellipse(aes(fill = Cluster), geom = "polygon", alpha = 0.15, 
               level = 0.95, show.legend = FALSE) +
  scale_color_manual(values = my_colors[1:3], 
                     labels = c("聚类1", "聚类2", "聚类3")) +
  scale_fill_manual(values = my_colors[1:3]) +
  labs(title = "K-means聚类结果（PCA降维可视化）",
       subtitle = "95%置信椭圆显示聚类边界",
       x = paste0("主成分1 (", round(variance_explained[1], 2), "%)"),
       y = paste0("主成分2 (", round(variance_explained[2], 2), "%)"),
       color = "聚类") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        legend.position = "right",
        panel.grid.minor = element_blank(),
        panel.border = element_rect(fill = NA, color = "grey70"))

print(pca_cluster_plot)

# 图2: 真实类别vs聚类结果对比
pca_true_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = True_Class)) +
  geom_point(size = 4, alpha = 0.8) +
  stat_ellipse(aes(fill = True_Class), geom = "polygon", alpha = 0.15, 
               level = 0.95, show.legend = FALSE) +
  scale_color_manual(values = my_colors[4:6], 
                     labels = c("品种1", "品种2", "品种3")) +
  scale_fill_manual(values = my_colors[4:6]) +
  labs(title = "真实葡萄酒品种分布（PCA降维可视化）",
       subtitle = "用于与聚类结果对比",
       x = paste0("主成分1 (", round(variance_explained[1], 2), "%)"),
       y = paste0("主成分2 (", round(variance_explained[2], 2), "%)"),
       color = "真实品种") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        legend.position = "right",
        panel.grid.minor = element_blank(),
        panel.border = element_rect(fill = NA, color = "grey70"))

print(pca_true_plot)

# 图3: 聚类中心热图
centers_df <- as.data.frame(kmeans_final$centers)
centers_df$Cluster <- paste0("聚类", 1:3)
centers_long <- reshape2::melt(centers_df, id.vars = "Cluster")

heatmap_plot <- ggplot(centers_long, aes(x = variable, y = Cluster, fill = value)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = round(value, 2)), color = "white", 
            fontface = "bold", size = 3.5) +
  scale_fill_gradient2(low = "#2979FF", mid = "white", high = "#FF1744",
                       midpoint = 0, name = "标准化值") +
  labs(title = "K-means聚类中心特征热图",
       subtitle = "展示各聚类在不同特征上的标准化均值",
       x = "特征", y = "聚类") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.text.y = element_text(size = 11),
        panel.grid = element_blank())

print(heatmap_plot)

# 图4: 各聚类特征雷达图
par(mfrow = c(1, 3), mar = c(1, 1, 3, 1))
for (i in 1:3) {
  # 准备雷达图数据
  radar_data <- rbind(
    max = rep(3, ncol(centers_df) - 1),
    min = rep(-3, ncol(centers_df) - 1),
    centers_df[i, -ncol(centers_df)]
  )
  
  radarchart(radar_data,
             axistype = 1,
             pcol = my_colors[i],
             pfcol = adjustcolor(my_colors[i], alpha.f = 0.3),
             plwd = 3,
             cglcol = "grey",
             cglty = 1,
             axislabcol = "grey30",
             caxislabels = seq(-3, 3, 1.5),
             cglwd = 0.8,
             vlcex = 0.7,
             title = paste0("聚类", i, "特征轮廓"))
}
par(mfrow = c(1, 1))

# ============================================================================
# 6.3 聚类验证
# ============================================================================

cat("\n========== 6.3 聚类验证 ==========\n")

# 1. 内部验证指标

# 轮廓系数
cat("\n计算轮廓系数...\n")
sil <- silhouette(kmeans_final$cluster, dist(wine_scaled))
avg_sil <- mean(sil[, 3])
cat("平均轮廓系数:", round(avg_sil, 4), "\n")

# 轮廓系数可视化
sil_plot <- fviz_silhouette(sil, palette = my_colors[1:3],
                            ggtheme = theme_minimal()) +
  labs(title = "轮廓系数图",
       subtitle = paste0("平均轮廓系数 = ", round(avg_sil, 4)),
       x = "样本", y = "轮廓系数") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        axis.text.x = element_blank())

print(sil_plot)

# Dunn指数
cat("计算Dunn指数...\n")
dunn_index <- calculate_dunn_index(wine_scaled, kmeans_final$cluster)
cat("Dunn指数:", round(dunn_index, 4), "\n")

# Davies-Bouldin指数
cat("计算Davies-Bouldin指数...\n")
db_index <- calculate_db_index(wine_scaled, kmeans_final$cluster, 
                               kmeans_final$centers)
cat("Davies-Bouldin指数:", round(db_index, 4), "(越小越好)\n")

# 2. 外部验证指标（与真实标签比较）

# 调整兰德指数（ARI）
cat("\n计算调整兰德指数...\n")
ari <- adjusted_rand_index(kmeans_final$cluster, true_labels)
cat("调整兰德指数(ARI):", round(ari, 4), "\n")

# 标准化互信息（NMI）
cat("计算标准化互信息...\n")
nmi <- normalized_mutual_info(kmeans_final$cluster, true_labels)
cat("标准化互信息(NMI):", round(nmi, 4), "\n")

# 混淆矩阵
confusion_matrix <- table(Predicted = kmeans_final$cluster, 
                          Actual = true_labels)
cat("\n混淆矩阵:\n")
print(confusion_matrix)

# 计算准确率（通过最佳匹配）
accuracy <- max(
  sum(diag(confusion_matrix)),
  sum(confusion_matrix[c(1,2,3), c(1,3,2)]),
  sum(confusion_matrix[c(1,2,3), c(2,1,3)]),
  sum(confusion_matrix[c(1,2,3), c(2,3,1)]),
  sum(confusion_matrix[c(1,2,3), c(3,1,2)]),
  sum(confusion_matrix[c(1,2,3), c(3,2,1)])
) / sum(confusion_matrix)
cat("聚类准确率（最佳匹配）:", round(accuracy, 4), "\n")

# 混淆矩阵热图
confusion_df <- as.data.frame(confusion_matrix)
confusion_plot <- ggplot(confusion_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Freq), color = "white", 
            fontface = "bold", size = 8) +
  scale_fill_gradient(low = "#E3F2FD", high = "#FF1744", 
                      name = "样本数") +
  labs(title = "聚类结果混淆矩阵",
       subtitle = paste0("ARI = ", round(ari, 4), " | NMI = ", round(nmi, 4), 
                         " | 准确率 = ", round(accuracy, 4)),
       x = "真实品种", y = "预测聚类") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        panel.grid = element_blank(),
        axis.text = element_text(size = 11))

print(confusion_plot)

# 3. 聚类稳定性验证（Bootstrap）
cat("\n进行Bootstrap稳定性验证...\n")
set.seed(123)
n_bootstrap <- 100
ari_bootstrap <- numeric(n_bootstrap)

pb <- txtProgressBar(min = 0, max = n_bootstrap, style = 3)
for (i in 1:n_bootstrap) {
  # 有放回抽样
  sample_idx <- sample(1:nrow(wine_scaled), replace = TRUE)
  wine_boot <- wine_scaled[sample_idx, ]
  labels_boot <- true_labels[sample_idx]
  
  # 聚类
  kmeans_boot <- kmeans(wine_boot, centers = 3, nstart = 25)
  
  # 计算ARI
  ari_bootstrap[i] <- adjusted_rand_index(kmeans_boot$cluster, labels_boot)
  
  setTxtProgressBar(pb, i)
}
close(pb)

cat("\nBootstrap稳定性验证 (n=100):\n")
cat("ARI均值:", round(mean(ari_bootstrap), 4), "\n")
cat("ARI标准差:", round(sd(ari_bootstrap), 4), "\n")
cat("ARI 95%置信区间: [", round(quantile(ari_bootstrap, 0.025), 4), 
    ",", round(quantile(ari_bootstrap, 0.975), 4), "]\n")

# Bootstrap ARI分布图
bootstrap_plot <- ggplot(data.frame(ARI = ari_bootstrap), aes(x = ARI)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, 
                 fill = "#00E676", color = "white", alpha = 0.8) +
  geom_density(color = "#FF1744", linewidth = 1.2) +
  geom_vline(xintercept = mean(ari_bootstrap), 
             color = "#2979FF", linetype = "dashed", linewidth = 1.2) +
  labs(title = "Bootstrap聚类稳定性验证",
       subtitle = paste0("基于100次有放回抽样的ARI分布 (均值 = ", 
                         round(mean(ari_bootstrap), 4), ")"),
       x = "调整兰德指数 (ARI)", y = "密度") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        panel.grid.minor = element_blank()) +
  annotate("text", x = mean(ari_bootstrap), y = 0, 
           label = paste0("均值 = ", round(mean(ari_bootstrap), 4)),
           vjust = -1, color = "#2979FF", fontface = "bold", size = 4)

print(bootstrap_plot)

# 4. 综合评估指标汇总表
evaluation_metrics <- data.frame(
  指标类别 = c("内部验证", "内部验证", "内部验证", 
           "外部验证", "外部验证", "外部验证", "稳定性验证"),
  指标名称 = c("平均轮廓系数", "Dunn指数", "Davies-Bouldin指数",
           "调整兰德指数(ARI)", "标准化互信息(NMI)", "聚类准确率",
           "Bootstrap ARI均值"),
  指标值 = c(round(avg_sil, 4), round(dunn_index, 4), round(db_index, 4),
          round(ari, 4), round(nmi, 4), round(accuracy, 4),
          round(mean(ari_bootstrap), 4)),
  评价 = c(
    ifelse(avg_sil > 0.5, "优秀", ifelse(avg_sil > 0.3, "良好", "一般")),
    ifelse(dunn_index > 0.1, "良好", "一般"),
    ifelse(db_index < 1, "优秀", ifelse(db_index < 2, "良好", "一般")),
    ifelse(ari > 0.7, "优秀", ifelse(ari > 0.5, "良好", "一般")),
    ifelse(nmi > 0.7, "优秀", ifelse(nmi > 0.5, "良好", "一般")),
    ifelse(accuracy > 0.8, "优秀", ifelse(accuracy > 0.6, "良好", "一般")),
    ifelse(mean(ari_bootstrap) > 0.7, "稳定", "较稳定")
  )
)

cat("\n========== 聚类评估指标汇总 ==========\n")
print(evaluation_metrics)

# 评估指标可视化
metrics_for_plot <- evaluation_metrics[evaluation_metrics$指标类别 != "稳定性验证", ]

metrics_plot <- ggplot(metrics_for_plot, 
                       aes(x = reorder(指标名称, 指标值), 
                           y = 指标值, fill = 指标类别)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = 指标值), hjust = -0.2, 
            fontface = "bold", size = 4) +
  coord_flip() +
  scale_fill_manual(values = c("内部验证" = "#2979FF", "外部验证" = "#FF1744")) +
  labs(title = "聚类评估指标汇总",
       subtitle = "内部验证与外部验证指标对比",
       x = "", y = "指标值", fill = "指标类别") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        legend.position = "top",
        panel.grid.minor = element_blank()) +
  ylim(0, max(metrics_for_plot$指标值) * 1.15)

print(metrics_plot)

# ============================================================================
# 结果保存
# ============================================================================

# 保存聚类结果
wine_data$Cluster <- kmeans_final$cluster
write.csv(wine_data, "wine_clustering_results.csv", row.names = FALSE)

# 保存评估指标
write.csv(evaluation_metrics, "clustering_evaluation_metrics.csv", row.names = FALSE)

cat("\n========== 分析完成 ==========\n")
cat("聚类结果已保存至: wine_clustering_results.csv\n")
cat("评估指标已保存至: clustering_evaluation_metrics.csv\n")
cat("\n最终评估结果:\n")
cat("- 最优聚类数: 3\n")
cat("- 平均轮廓系数:", round(avg_sil, 4), "\n")
cat("- 调整兰德指数(ARI):", round(ari, 4), "\n")
cat("- 标准化互信息(NMI):", round(nmi, 4), "\n")
cat("- 聚类准确率:", round(accuracy, 4), "\n")
cat("- 聚类质量评价:", evaluation_metrics$评价[4], "\n")
