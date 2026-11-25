# install.packages(c("dplyr", "ggplot2", "corrplot", "VIM", "car", "caret", "rpart", "randomForest", "gbm"))

library(dplyr)
library(ggplot2)
library(corrplot)
library(VIM)          # for KNN imputation
library(car)           # for VIF
library(caret)         # for train/test split + confusion matrix
library(rpart)         # decision trees
library(randomForest)  # random forest
library(gbm)           # gradient boosting

set.seed(1)

# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================
phones <- read.csv("used_device_data.csv", stringsAsFactors = FALSE)

dim(phones)
str(phones)
head(phones)

# "4g" and "5g" in the CSV become "X4g" and "X5g" once loaded.
names(phones)

# ============================================================
# STEP 2: FIX DATA TYPES
# these columns are categories, not numbers, so they need to be factors
# ============================================================
phones$device_brand <- as.factor(phones$device_brand)
phones$os           <- as.factor(phones$os)
phones$X4g          <- as.factor(phones$X4g)
phones$X5g          <- as.factor(phones$X5g)

# ============================================================
# STEP 3: FIRST LOOK AT MISSING VALUES
# just a look for now -- we don't fix anything yet. Fixing comes
# AFTER the split (Step 5), so that how we fill gaps in training
# never uses any information from the test rows, and vice versa.
# ============================================================
colSums(is.na(phones))

# ============================================================
# STEP 4: TRAIN/TEST SPLIT -- done BEFORE any preprocessing
# Every step from here on (imputation, outlier checks, EDA,
# brand grouping, encoding, the price_class cutoff) is done
# separately for train vs. test. If we imputed, encoded, or
# picked a median cutoff using the full dataset first, the
# test set would quietly leak information into decisions made
# about the training set, and the test performance numbers at
# the end would be too optimistic.
# ============================================================
train_index  <- createDataPartition(phones$normalized_used_price, p = 0.8, list = FALSE)
phones_train <- phones[train_index, ]
phones_test  <- phones[-train_index, ]

dim(phones_train)
dim(phones_test)

# ============================================================
# STEP 5: FIX MISSING VALUES (KNN IMPUTATION)
# only a handful of columns have gaps, and they're all under 5%
# of the data, so we fill them in with KNN imputation instead of
# just dropping rows or using a flat average.
#
# kNN() is run separately on phones_train and phones_test, so a
# training row is never "healed" using a test row as a donor
# (or vice versa) -- each set only borrows from its own rows.
# ============================================================
cols_to_fix <- c("rear_camera_mp", "front_camera_mp", "internal_memory",
                  "ram", "battery", "weight")

phones_train <- kNN(phones_train, variable = cols_to_fix, k = 5, imp_var = FALSE)
phones_test  <- kNN(phones_test,  variable = cols_to_fix, k = 5, imp_var = FALSE)

# double check nothing is missing anymore
colSums(is.na(phones_train))
colSums(is.na(phones_test))

# ============================================================
# STEP 6: CHECK ZERO VALUES (training set only)
# some phones really do have 0 MP front cameras (older/budget
# models) so we leave these as they are instead of treating
# them as errors
# ============================================================
sum(phones_train$front_camera_mp == 0)

# ============================================================
# STEP 7: LOOK AT OUTLIERS (IQR rule, training set only)
# We only look at training data here -- outlier-hunting is part
# of getting to know what we're learning from, not a reason to
# look at the test set before it's needed.
# ============================================================
find_outliers <- function(x) {
  q1  <- quantile(x, 0.25)
  q3  <- quantile(x, 0.75)
  iqr <- q3 - q1
  sum(x < (q1 - 1.5 * iqr) | x > (q3 + 1.5 * iqr))
}

find_outliers(phones_train$ram)
find_outliers(phones_train$screen_size)
find_outliers(phones_train$weight)
find_outliers(phones_train$battery)
find_outliers(phones_train$normalized_used_price)

# ============================================================
# STEP 8: EXPLORE THE DATA (EDA, training set only)
# ============================================================
table(phones_train$os)
ggplot(phones_train, aes(x = os)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Number of Phones by Operating System (Training Set)")

# battery capacity distribution
ggplot(phones_train, aes(x = battery)) +
  geom_histogram(bins = 30, fill = "steelblue") +
  labs(title = "Battery Capacity Distribution (Training Set)", x = "Battery (mAh)")

# battery vs release year
ggplot(phones_train, aes(x = factor(release_year), y = battery)) +
  geom_boxplot() +
  labs(title = "Battery Capacity by Release Year (Training Set)", x = "Release Year")

# weight by OS
ggplot(phones_train, aes(x = os, y = weight)) +
  geom_boxplot() +
  labs(title = "Phone Weight by Operating System (Training Set)")

# internal memory vs RAM
ggplot(phones_train, aes(x = internal_memory, y = ram)) +
  geom_point(alpha = 0.4) +
  labs(title = "Internal Memory vs RAM (Training Set)")

# used price by brand (top 10 brands only, so the chart is readable)
# NOTE: "top 10" is decided from the TRAINING set only. Step 10
# reuses this exact same list on the test set, rather than letting
# the test set pick its own "top 10".
top_brands <- phones_train %>%
  count(device_brand, sort = TRUE) %>%
  head(10) %>%
  pull(device_brand)

phones_train %>%
  filter(device_brand %in% top_brands) %>%
  ggplot(aes(x = reorder(device_brand, normalized_used_price, median),
             y = normalized_used_price)) +
  geom_boxplot() +
  coord_flip() +
  labs(title = "Used Price by Brand - Top 10 (Training Set)", x = "Brand", y = "Used Price")

# ============================================================
# STEP 9: CORRELATION WITH PRICE (training set only)
# ============================================================
num_data <- phones_train %>%
  select(screen_size, rear_camera_mp, front_camera_mp, internal_memory,
         ram, battery, weight, release_year, days_used,
         normalized_new_price, normalized_used_price)

cor_matrix <- cor(num_data)
corrplot(cor_matrix, method = "color", addCoef.col = "black", number.cex = 0.6)

# which features are most correlated with used price?
sort(cor_matrix[, "normalized_used_price"], decreasing = TRUE)

# ============================================================
# STEP 10: GROUP LOW-VOLUME BRANDS
# with 34 brands, most barely show up in the data, so we keep
# the top 10 (learned from the TRAINING set above, Step 8) and
# lump everything else into "Other_Brands". The test set reuses
# that same top_brands list instead of recomputing its own --
# otherwise train and test could end up with different brand
# categories, which would break the encoding in Step 11.
# ============================================================
phones_train$brand_grouped <- as.character(phones_train$device_brand)
phones_train$brand_grouped[!(phones_train$brand_grouped %in% top_brands)] <- "Other_Brands"
phones_train$brand_grouped <- as.factor(phones_train$brand_grouped)

phones_test$brand_grouped <- as.character(phones_test$device_brand)
phones_test$brand_grouped[!(phones_test$brand_grouped %in% top_brands)] <- "Other_Brands"
phones_test$brand_grouped <- factor(phones_test$brand_grouped, levels = levels(phones_train$brand_grouped))

table(phones_train$brand_grouped)
table(phones_test$brand_grouped)

# ============================================================
# STEP 11: ONE-HOT ENCODE THE CATEGORICAL COLUMNS
# The encoder is FIT on the training set only (dummyVars(...)),
# then just APPLIED (predict(), not re-fit) to the test set.
# That guarantees train and test end up with identical dummy
# columns, even if a category happens to be rare in one set.
# ============================================================
dummies <- dummyVars(~ brand_grouped + os + X4g + X5g, data = phones_train, fullRank = TRUE)

encoded_train <- data.frame(predict(dummies, newdata = phones_train))
encoded_test  <- data.frame(predict(dummies, newdata = phones_test))

model_data_train <- phones_train %>%
  select(screen_size, rear_camera_mp, front_camera_mp, internal_memory,
         ram, battery, weight, release_year, days_used,
         normalized_new_price, normalized_used_price) %>%
  bind_cols(encoded_train)

model_data_test <- phones_test %>%
  select(screen_size, rear_camera_mp, front_camera_mp, internal_memory,
         ram, battery, weight, release_year, days_used,
         normalized_new_price, normalized_used_price) %>%
  bind_cols(encoded_test)

# ============================================================
# STEP 12: CHECK FOR MULTICOLLINEARITY (VIF)
# Checked on the training set, and this time including the
# brand/os/4g/5g dummy columns too (not just the raw numeric
# predictors) -- those dummies go into every model below, so
# they need to be checked as well.
# ============================================================
vif_check <- lm(normalized_used_price ~ ., data = model_data_train)
sort(vif(vif_check), decreasing = TRUE)

# nothing is above 10, so we keep all the predictors

# ============================================================
# STEP 13: CREATE THE HIGH/LOW PRICE LABEL (for classification)
# The cutoff is the median of the TRAINING set's price only.
# We reuse that exact number on the test set instead of letting
# the test set compute its own median -- otherwise "High" and
# "Low" would mean different things in each set.
# ============================================================
price_cutoff <- median(model_data_train$normalized_used_price)
price_cutoff

model_data_train$price_class <- as.factor(
  ifelse(model_data_train$normalized_used_price > price_cutoff, "High", "Low")
)
model_data_test$price_class <- as.factor(
  ifelse(model_data_test$normalized_used_price > price_cutoff, "High", "Low")
)

table(model_data_train$price_class)
table(model_data_test$price_class)

# ============================================================
# STEP 14: BUILD THE FINAL MODELING TABLES
# One split, shared by both tasks:
#  - reg_train / reg_test drop price_class (that's the
#    classification label, not a predictor)
#  - class_train / class_test drop normalized_used_price
#    (that's literally what price_class was built from, so
#    leaving it in would hand the classifier the answer)
# ============================================================
reg_train <- model_data_train %>% select(-price_class)
reg_test  <- model_data_test  %>% select(-price_class)

class_train <- model_data_train %>% select(-normalized_used_price)
class_test  <- model_data_test  %>% select(-normalized_used_price)

table(class_train$price_class)
table(class_test$price_class)

# ============================================================
# STEP 15: REGRESSION MODELS (predicting the exact price)
# ============================================================
# --- Model 1: Linear Regression ---
lm_model <- lm(normalized_used_price ~ ., data = reg_train)
summary(lm_model)
lm_pred <- predict(lm_model, reg_test)

# --- Model 2: Decision Tree ---
tree_model <- rpart(normalized_used_price ~ ., data = reg_train)
tree_pred <- predict(tree_model, reg_test)

# --- Model 3: Random Forest ---
rf_model <- randomForest(normalized_used_price ~ ., data = reg_train, ntree = 300)
rf_pred <- predict(rf_model, reg_test)

# --- Model 4: Gradient Boosting ---
gbm_model <- gbm(normalized_used_price ~ ., data = reg_train,
                  distribution = "gaussian", n.trees = 300,
                  interaction.depth = 3, shrinkage = 0.05)
gbm_pred <- predict(gbm_model, reg_test, n.trees = 300)

# --- Compare the four models ---
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
r_squared <- function(actual, predicted) 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)

results_regression <- data.frame(
  Model = c("Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"),
  RMSE = c(rmse(reg_test$normalized_used_price, lm_pred),
           rmse(reg_test$normalized_used_price, tree_pred),
           rmse(reg_test$normalized_used_price, rf_pred),
           rmse(reg_test$normalized_used_price, gbm_pred)),
  R_squared = c(r_squared(reg_test$normalized_used_price, lm_pred),
                r_squared(reg_test$normalized_used_price, tree_pred),
                r_squared(reg_test$normalized_used_price, rf_pred),
                r_squared(reg_test$normalized_used_price, gbm_pred))
)

print(results_regression)

# ============================================================
# STEP 16: CLASSIFICATION MODELS (predicting High or Low)
# ============================================================
# --- Model 1: Logistic Regression ---
logit_model <- glm(price_class ~ ., data = class_train, family = "binomial")
logit_prob <- predict(logit_model, class_test, type = "response")
logit_pred <- as.factor(ifelse(logit_prob > 0.5, "High", "Low"))

# --- Model 2: Decision Tree ---
tree_class_model <- rpart(price_class ~ ., data = class_train, method = "class")
tree_class_pred <- predict(tree_class_model, class_test, type = "class")

# --- Model 3: Random Forest ---
rf_class_model <- randomForest(price_class ~ ., data = class_train, ntree = 300)
rf_class_pred <- predict(rf_class_model, class_test)

# --- Model 4: Gradient Boosting ---
# gbm needs a 0/1 numeric target instead of a factor
class_train_num <- class_train %>%
  mutate(price_class_num = ifelse(price_class == "High", 1, 0)) %>%
  select(-price_class)

gbm_class_model <- gbm(price_class_num ~ ., data = class_train_num,
                        distribution = "bernoulli", n.trees = 300,
                        interaction.depth = 3, shrinkage = 0.05)
gbm_class_prob <- predict(gbm_class_model, class_test, n.trees = 300, type = "response")
gbm_class_pred <- as.factor(ifelse(gbm_class_prob > 0.5, "High", "Low"))

# --- Compare the four models using accuracy ---
acc <- function(predicted, actual) mean(predicted == actual)

results_classification <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"),
  Accuracy = c(acc(logit_pred, class_test$price_class),
               acc(tree_class_pred, class_test$price_class),
               acc(rf_class_pred, class_test$price_class),
               acc(gbm_class_pred, class_test$price_class))
)

print(results_classification)

# a closer look at the best-performing model
confusionMatrix(rf_class_pred, class_test$price_class, positive = "High")
