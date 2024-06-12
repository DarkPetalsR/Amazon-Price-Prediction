#Importing the dataset and storing it in an object myproj

AmazonSales <- read.csv("G:\\Data Analytics\\Amazon_project.csv")

#View the dataset

View(AmazonSales)

install.packages("webshot")
webshot::install_phantomjs()


#Installing necessary packages 

install.packages("tidyverse")
install.packages("ggplot2")
install.packages("heatmaply")
install.packages("caret")
install.packages("webshot")
install.packages("webshot2")
install.packages("kableExtra")
install.packages("knitr")
install.packages("rmarkdown")
install.packages("tinytex")  
install.packages("knitr")
install.packages("pander")
install.packages("officer")
install.packages("flextable")
install.packages("wordcloud")
install.packages("glmnet")

library(glmnet)

library(wordcloud)
library(rmarkdown)
library(tinytex)
library(knitr)
library(kableExtra)
library(tidyverse) 
library(caret)
library(nnet)
library(webshot)
library(webshot2)
library(markdown)
library(dplyr)
library(ggplot2)
library(pander)
library(officer)
library(flextable)
library(heatmaply)

# Display the first few rows of the AmazonSales dataset
head(AmazonSales)

# Generate summary statistics for each variable in the AmazonSales dataset 
str(AmazonSales)

# Provide a concise summary of the structure of the AmazonSales dataset
summary(AmazonSales)


#Creating a new dataframe without irrelavant columns for our analysis 

# Selecting all columns except "img link" and "product link"
AmazonSalesDF <- AmazonSales[, !(names(AmazonSales) %in% c("img_link", "product_link"))]

# Print the new data frame
print(AmazonSalesDF)

#Viewing the new dataframe

View(AmazonSalesDF)


#Removing , separators from the column 

AmazonSalesDF$rating_count <- gsub(",", "", AmazonSalesDF$rating_count) 
AmazonSalesDF$discounted_price <- gsub("₹", "", AmazonSalesDF$discounted_price)
AmazonSalesDF$discounted_price <- gsub(",", "", AmazonSalesDF$discounted_price)
AmazonSalesDF$actual_price <- gsub("₹", "", AmazonSalesDF$actual_price) 
AmazonSalesDF$actual_price <- gsub(",", "", AmazonSalesDF$actual_price)
AmazonSalesDF$discount_percentage <- gsub("%", "", AmazonSalesDF$discount_percentage) 

view(AmazonSalesDF)



#Checking data types for all the columns 
# Use sapply to get data types for each column
data_types <- sapply(AmazonSalesDF, class)
print(data_types)

#Changing data types 
AmazonSalesDF$discounted_price <- as.numeric(AmazonSalesDF$discounted_price) 
AmazonSalesDF$actual_price <- as.numeric(AmazonSalesDF$actual_price) 
AmazonSalesDF$discount_percentage <- as.numeric(AmazonSalesDF$discount_percentage) 
AmazonSalesDF$rating <- as.numeric(AmazonSalesDF$rating)
AmazonSalesDF$rating_count <- as.numeric(AmazonSalesDF$rating_count)

#Viewing Datatypes

str(AmazonSalesDF)

View(AmazonSalesDF)


#Identifying Duplicate values 

duplicate_rows <- duplicated(AmazonSalesDF)

View(duplicate_rows)

print(duplicate_rows)

#Removing Duplicate Data accordingly 

unique_AmazonSalesDF <- unique(AmazonSalesDF)

View (unique_AmazonSalesDF)

#Checking for Missing values
missing_counts <- colSums(is.na(unique_AmazonSalesDF))

# Print the missing value counts
print(missing_counts)

#Identifying the rows with missing and NA values 

# Check for rows with "NA" values
rows_with_na <- unique_AmazonSalesDF[!complete.cases(unique_AmazonSalesDF), ]

# Display the rows with "NA" values
print("Rows with NA values:")
print(rows_with_na)

#Result: print(rows_with_na)
#product_id
#721  B08L12N5H1
#1367 B0B94JPY2N
#1464 B0BQRJ3C47

# Calculate the mean of the 'rating' column (excluding NAs)
mean_rating <- mean(unique_AmazonSalesDF$rating, na.rm = TRUE)

# Print the calculated mean
print(paste("Mean Rating:", mean_rating))

# Impute missing values in 'rating' with the calculated mean
unique_AmazonSalesDF$rating[is.na(unique_AmazonSalesDF$rating)] <- mean_rating

# Verify that missing values have been imputed
summary(unique_AmazonSalesDF$rating)


# Calculate the mean of the 'rating' column (excluding NAs)
mean_rating_count <- mean(unique_AmazonSalesDF$rating_count, na.rm = TRUE)

# Print the calculated mean
print(paste("Mean of rating count:", mean_rating_count))

# Impute missing values in 'rating' with the calculated mean
unique_AmazonSalesDF$rating_count[is.na(unique_AmazonSalesDF$rating_count)] <- mean_rating_count

# Verify if that missing values have been imputed
summary(unique_AmazonSalesDF$rating_count)

View(unique_AmazonSalesDF)


# Checking the frequency count of unique values in the 'category' column
category_counts <- table(unique_AmazonSalesDF$category)

print(category_counts)

unique_AmazonSalesDF <- unique_AmazonSalesDF  %>%
  mutate(main_category = sub("\\|.*", "", category))

# Display the structure of the modified dataset
str(unique_AmazonSalesDF )

View(unique_AmazonSalesDF)

############################################################################################

#DATA VISUALIZATION 

# Create a scatter plot for Rating Vs Discount Percentage 
plot(unique_AmazonSalesDF$rating, unique_AmazonSalesDF$discount_percentage, col = "darkblue", pch = 16, cex = 1.5, xlab = "rating", ylab = "discount_percentage")


# Histogram for discounted_price
ggplot(unique_AmazonSalesDF, aes(x = discounted_price)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black") +
  labs(title = "Histogram of Discounted Price", x = "Discounted Price", y = "Frequency") 

# Histogram for actual_price
ggplot(unique_AmazonSalesDF, aes(x = actual_price)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black") +
  labs(title = "Histogram of Actual Price", x = "Actual Price", y = "Frequency") 


# Histogram for the rating column
hist(unique_AmazonSalesDF$rating, col = "blue", main = "Distribution of Ratings", xlab = "Rating", ylab = "Frequency")


# Histogram for Rating count
ggplot(unique_AmazonSalesDF, aes(x = rating_count)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black") +
  labs(title = "Histogram of Rating Count", x = "Rating Count", y = "Frequency") 


# Scatter plot for rating_count vs. rating
ggplot(unique_AmazonSalesDF, aes(x = rating, y = rating_count)) +
  geom_point(color = "blue") +
  labs(title = "Scatter Plot of Rating vs. Rating Count", x = "Rating", y = "Rating Count")

#Scatter Plot for Actual price Vs Discounted Price 
ggplot(unique_AmazonSalesDF, aes(x = discounted_price, y = actual_price)) +
  geom_point(color = "blue") +
  labs(title = "Scatter Plot of Discounted Price Vs Actual Price", x = "Discounted Price", y = "Actual Price")



#Exploring the correlation between Actual price and Discount price 
  
  plot(
    x = unique_AmazonSalesDF$actual_price,
    y = unique_AmazonSalesDF$discounted_price,
    main = "Actual Price vs. Discounted Price",
    xlab = "Actual Price",
    ylab = "Discounted Price",
    pch = 16,
    col = "blue",
    cex = 1.2,  # Increase point size
    xlim = c(0, max(unique_AmazonSalesDF$actual_price) * 1.1),  # Adjust x-axis limits
    ylim = c(0, max(unique_AmazonSalesDF$discounted_price) * 1.1)  # Adjust y-axis limits
  )

# Add grid lines
grid()

# Add a legend
legend(
  "topright",
  legend = c("Data Points"),
  col = "blue",
  pch = 16,
  cex = 1.2,
  bg = "white"
)

#Scatter plot for Rating and Rating count

plot(unique_AmazonSalesDF$rating, unique_AmazonSalesDF$rating_count, 
     main="Rating vs. Rating Count",
     xlab="Rating",
     ylab="Rating Count",
     pch=16, col="orange")


#Histogram for Discount Percentage 
hist(unique_AmazonSalesDF$discount_percentage,
     main="Distribution of Discount Percentages",
     xlab="Discount Percentage",
     col="lightblue", border="black")




# Bar plot showing the distribution of products across main categories,
unique_AmazonSalesDF %>%
  group_by(main_category) %>%
  summarize(count_by_mainCategory = n()) %>%
  arrange(desc(count_by_mainCategory)) %>%
  mutate(main_category = factor(main_category, levels = main_category)) %>%
  ggplot(aes(main_category, count_by_mainCategory)) +
  geom_bar(stat = "identity", aes(fill = main_category)) +
  geom_text(aes(label = count_by_mainCategory, vjust = -1)) +
  labs(title = "Number of products by main_category",
       x = "Main category",
       y = "Number of products") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#CORRELATION MATRIX#


# We want to keep only numeric columns
numeric_data <- unique_AmazonSalesDF %>%
  select_if(is.numeric)

# Create the correlation matrix for numeric columns
correlation_matrix <- cor(numeric_data)

heatmaply(
  correlation_matrix,
  main = "Correlation Matrix Heatmap",
  width = 700,
  height = 700,
  margins = c(50, 50),
  fontsize_col = 12,
  fontsize_row = 12,
  cellnote = round(correlation_matrix, 2),
  notecol = "black",
  symm = TRUE,
  col = colorRampPalette(c("blue", "white", "red"))(100),
  na_col = "grey",
  cexRow = 0.8,
  cexCol = 0.8,
  plot_method = "plotly",
  show_dendrogram = c(FALSE, FALSE),
  hoverinfo = "all",
  legend_title = "Correlation",
  legend_title_font_size = 15,
  xlab = "Variables",
  ylab = "Variables"
)

library(heatmaply)


# Average rating by main category
average_rating_by_category <- unique_AmazonSalesDF %>%
  group_by(main_category) %>%
  summarize(average_rating = mean(rating)) %>%
  arrange(desc(average_rating))

# Display the results
print(average_rating_by_category)

# Box plot of the complete range of ratings grouped by main category
library(ggplot2)

unique_AmazonSalesDF %>%
  ggplot(aes(x = rating, y = main_category, fill = main_category)) +
  geom_boxplot() +
  labs(title = "Box Plot of Rating Grouped by Main Category",
       x = "Rating",
       y = "Main Category") +
  theme_minimal() +
  theme(legend.position = "none")



# Calculating average rating_count for the entire data set
average_rating_count_total <- unique_AmazonSalesDF %>%
  summarize(average_rating_count = mean(rating_count))

# Display the result
print(average_rating_count_total)

# Average rating_count by main category
average_rating_count_by_category <- unique_AmazonSalesDF %>%
  group_by(main_category) %>%
  summarize(average_rating_count = mean(rating_count)) %>%
  arrange(desc(average_rating_count))

# Display the results
print(average_rating_count_by_category)

# Box plot of the complete range of rating counts grouped by main category
library(ggplot2)

unique_AmazonSalesDF %>%
  ggplot(aes(x = rating_count, y = main_category, fill = main_category)) +
  geom_boxplot() +
  labs(title = "Box Plot of Rating Count Grouped by Main Category",
       x = "Rating Count",
       y = "Main Category") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_x_continuous(limits = c(0, 500000), breaks = seq(0, 500000, by = 100000),
                     labels = scales::comma_format())


#Creating Pie chart with Main category and the corresponding percentages 

# Install and load necessary packages
if (!requireNamespace("plotly", quietly = TRUE)) {
  install.packages("plotly")
}
library(plotly)


# Create the pie chart
pie_chart <- plot_ly(unique_AmazonSalesDF, labels = ~main_category, values = ~discount_percentage, type = "pie") %>%
  layout(
    title = 'Percentage of Discount for each Category',
    autosize = FALSE,
    width = 800,
    height = 800
  ) %>%
  add_trace(textposition = 'inside', textinfo = 'percent+label')

# Display the chart
pie_chart



# Install and load necessary packages
if (!requireNamespace("plotly", quietly = TRUE)) {
  install.packages("plotly")
}
library(plotly)


#Average Discount Percentage for each category 


# Create the bar chart
bar_chart <- unique_AmazonSalesDF %>%
  group_by(main_category) %>%
  summarize(discount_percentage = mean(discount_percentage)) %>%
  arrange(discount_percentage) %>%
  ggplot(aes(x = discount_percentage, y = main_category)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Average Discount Percentage for each Main Category",
       x = "Discount Percentage",
       y = "Main Category") +
  theme_minimal()

# Display the chart
print(bar_chart)



#################################################################################################


#TEXT ANALYSIS 
  

# Load required libraries
library(tidyverse)
library(tm)
library(ggplot2)
library(tidytext)
library(textstem)

install.packages("textstem")

unique_AmazonSalesDF$processed_text <- unique_AmazonSalesDF$review_content %>%
  tolower() %>%                          # Convert to lowercase
  str_remove_all("[^a-zA-Z\\s]") %>%     # Remove non-alphabetic characters
  str_replace_all("\\b\\d+\\b", "") %>%  # Remove numbers
  str_replace_all("\\s+", " ") %>%       # Remove extra whitespaces
  str_trim()                             # Trim leading and trailing whitespaces

# Remove stopwords
stop_words <- stopwords("en")
unique_AmazonSalesDF$processed_text <- unique_AmazonSalesDF$processed_text %>%
  str_split(" ") %>%
  map_chr(~ paste(setdiff(., stop_words), collapse = " "))

# View the processed dataset
head(unique_AmazonSalesDF$processed_text)

# Tokenize the processed_text column
tokenized_data <- unique_AmazonSalesDF %>%
  unnest_tokens(word, processed_text)

# View the tokenized data
head(tokenized_data$word)

View(tokenized_data)

#Word frequency analysis
token_frequency <- tokenized_data %>%
  count(word, sort = TRUE)


head(token_frequency, 20)



View(tokenized_data)


# Loading sentiment lexicon 
sentiment_data <- tokenized_data %>%
  inner_join(get_sentiments("bing"))

# View sentiment distribution
ggplot(sentiment_data, aes(x = sentiment, fill = sentiment)) +
  geom_bar() +
  labs(title = "Sentiment Distribution", x = "Sentiment", y = "Count") +
  theme_minimal()

#Sentiment Distribution by Category
ggplot(sentiment_data, aes(x = category, fill = sentiment)) +
  geom_bar() +
  labs(title = "Sentiment Distribution by Category", x = "Category", y = "Sentiment") +
  theme_minimal()


# Create a bar plot
ggplot(sentiment_data, aes(x = sentiment, fill = as.factor(rating))) +
  geom_bar(position = "stack") +
  labs(title = "Sentiment and Rating Distribution",
       x = "Sentiment",
       y = "Count",
       fill = "Rating") +
  theme_minimal()


ggplot(sentiment_data, aes(x = sentiment, y = discount_percentage, fill = sentiment)) +
  geom_bar(stat = "identity") +
  labs(title = "Bar Graph of Discount Percentage by Sentiment",
       x = "Sentiment",
       y = "Discount Percentage",
       fill = "Sentiment") +
  theme_minimal()


ggplot(unique_AmazonSalesDF, aes(x = rating, y = actual_price)) +
  geom_point(color = "#4285F4", size = 3, alpha = 0.7) +  # Customized point aesthetics
  labs(title = "Scatter Plot of Rating vs. Actual Price",
       x = "Rating",
       y = "Actual Price") +
  theme_minimal() +  # Choose a theme (optional)
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        axis.text.x = element_text(angle = 45, hjust = 1))  # Adjust text size and rotation

  
library(tidytext)

# Example: Word frequency analysis
word_frequency <- tokenized_data %>%
  count(word, sort = TRUE)

head(word_frequency, 10)


# Install and load necessary library for word cloud
install.packages("wordcloud")
library(wordcloud)

# Extract word frequency for the first 400 words
top_words <- head(word_frequency, 400)

# Create a word cloud
wordcloud(words = top_words$word, freq = top_words$n, scale = c(3, 0.5), colors = brewer.pal(8, "Dark2"))



  
###############################################################################################

#PREDICTION MODELS 
  
#Research Question: "To what extent do actual price, discount percentage, 
#product rating, and rating count contribute to predicting the discount price 
#in a linear regression model for consumer goods, and 
#how do these variables interact with each other in influencing the final discounted pricing?



#Linear Regression for Price prediction
  

# Selecting columns for X and y
  X <- unique_AmazonSalesDF[, c("rating", "rating_count", "actual_price", "discount_percentage")]
  y <- unique_AmazonSalesDF$discounted_price
  
# Scaling the features
  X_scaled <- scale(X)
  
# Combining scaled features and target into a data frame
  train_data <- data.frame(cbind(X_scaled, y = y))
  
# Splitting the scaled data into training and testing sets
  set.seed(123)  # for reproducibility
  split_index <- sample(seq_len(nrow(unique_AmazonSalesDF)), size = 0.7 * nrow(unique_AmazonSalesDF))
  train_data <- train_data[split_index, ]
  test_data <- train_data[-split_index, ]
  
# Creating and fitting the linear regression model
  linreg_model <- lm(y ~ ., data = train_data)
  
# Predicting on the test set
  pred <- predict(linreg_model, newdata = test_data)
  
# Calculating evaluation metrics
  mse <- mean((test_data$y - pred)^2)  # Mean Squared Error
  emse <- mean((test_data$y - mean(train_data$y))^2)  # Explained Mean Squared Error
  mae <- mean(abs(test_data$y - pred))  # Mean Absolute Error
  r_squared <- cor(test_data$y, pred)^2  # R-squared
  adj_r_squared <- 1 - ((1 - r_squared) * (length(test_data$y) - 1) / (length(test_data$y) - length(coef(linreg_model)) - 1))  # Adjusted R-squared
  
# Printing results
  cat("Mean Squared Error (MSE): ", mse, "\n")
  cat("Explained Mean Squared Error (EMSE): ", emse, "\n")
  cat("Mean Absolute Error (MAE): ", mae, "\n")
  cat("R-squared Score: ", round(r_squared, 3), "\n")
  cat("Adjusted R-squared Score: ", round(adj_r_squared, 3), "\n")
  
  
# Obtain summary of the linear regression model
  linreg_summary <- summary(linreg_model) 


######Ridge Regression######
  

# Creating and fitting the Ridge regression model
  alpha <- 0  # Ridge regression (alpha = 0)
  ridge_model <- cv.glmnet(as.matrix(train_data[, -ncol(train_data)]), train_data$y, alpha = alpha)
  
# Predicting on the test set
  ridge_pred <- predict(ridge_model, s = ridge_model$lambda.min, newx = as.matrix(test_data[, -ncol(test_data)]))

# Calculating evaluation metrics for Ridge regression 
  ridge_mse <- mean((test_data$y - ridge_pred)^2)  # Mean Squared Error
  ridge_emse <- mean((test_data$y - mean(train_data$y))^2)  # Explained Mean Squared Error
  ridge_mae <- mean(abs(test_data$y - ridge_pred))  # Mean Absolute Error
  
# Calculate R-squared
  ridge_r_squared <- 1 - ridge_mse / ridge_emse
  
  # Calculate Adjusted R-squared
  n <- nrow(test_data)
  p <- ncol(test_data) - 1  # Number of predictors (excluding the intercept)
  ridge_adj_r_squared <- 1 - (1 - ridge_r_squared) * (n - 1) / (n - p - 1)
  
  # Printing results for Ridge regression
  cat("Ridge Regression Results:\n")
  cat("Mean Squared Error (MSE): ", ridge_mse, "\n")
  cat("Explained Mean Squared Error (EMSE): ", ridge_emse, "\n")
  cat("Mean Absolute Error (MAE): ", ridge_mae, "\n")
  cat("R-squared Score: ", round(ridge_r_squared, 3), "\n")
  cat("Adjusted R-squared Score: ", round(ridge_adj_r_squared, 3), "\n")
  

  
######Lasso Regression######
  
  # Lasso Regression
  # Creating and fitting the Lasso regression model
  lasso_model <- cv.glmnet(as.matrix(train_data[, -ncol(train_data)]), train_data$y, alpha = 1)  # alpha = 1 for Lasso
  
  # Predicting on the test set
  lasso_pred <- predict(lasso_model, s = lasso_model$lambda.min, newx = as.matrix(test_data[, -ncol(test_data)]))
  
  # Calculating evaluation metrics for Lasso regression
  lasso_mse <- mean((test_data$y - lasso_pred)^2)  # Mean Squared Error
  lasso_emse <- mean((test_data$y - mean(train_data$y))^2)  # Explained Mean Squared Error
  lasso_mae <- mean(abs(test_data$y - lasso_pred))  # Mean Absolute Error
  
  # Calculate R-squared
  lasso_r_squared <- 1 - lasso_mse / lasso_emse
  
  # Calculate Adjusted R-squared
  lasso_adj_r_squared <- 1 - (1 - lasso_r_squared) * (n - 1) / (n - p - 1)
  
  # Printing results for Lasso regression
  cat("Lasso Regression Results:\n")
  cat("Mean Squared Error (MSE): ", lasso_mse, "\n")
  cat("Explained Mean Squared Error (EMSE): ", lasso_emse, "\n")
  cat("Mean Absolute Error (MAE): ", lasso_mae, "\n")
  cat("R-squared Score: ", round(lasso_r_squared, 3), "\n")
  cat("Adjusted R-squared Score: ", round(lasso_adj_r_squared, 3), "\n")
  
  
  
#####ELASTIC NET REGRESSION#####
  
  
  library(glmnet)
  
  # Elastic Net Regression
  # Creating and fitting the Elastic Net regression model
  enet_model <- cv.glmnet(as.matrix(train_data[, -ncol(train_data)]), train_data$y, alpha = 0.5)  # alpha = 0.5 for Elastic Net
  
  # Predicting on the test set
  enet_pred <- predict(enet_model, s = enet_model$lambda.min, newx = as.matrix(test_data[, -ncol(test_data)]))
  
  # Calculating evaluation metrics for Elastic Net regression
  enet_mse <- mean((test_data$y - enet_pred)^2)  # Mean Squared Error
  enet_emse <- mean((test_data$y - mean(train_data$y))^2)  # Explained Mean Squared Error
  enet_mae <- mean(abs(test_data$y - enet_pred))  # Mean Absolute Error
  
  # Calculate R-squared
  enet_r_squared <- 1 - enet_mse / enet_emse
  
  # Check the values of n and p
  n <- nrow(test_data)  
  p <- ncol(test_data) - 1  
  

  # Calculate Adjusted R-squared
  enet_adj_r_squared <- 1 - (1 - enet_r_squared) * (n - 1) / (n - p - 1)
  
  
  # Check the values of n and p
  cat("Number of observations (n): ", n, "\n")
  cat("Number of predictors (p): ", p, "\n")
  
  # Calculate Adjusted R-squared
  enet_adj_r_squared <- 1 - (1 - enet_r_squared) * (n - 1) / (n - p - 1)
  
  # Printing results for Elastic Net regression
  cat("Elastic Net Regression Results:\n")
  cat("Mean Squared Error (MSE): ", enet_mse, "\n")
  cat("Explained Mean Squared Error (EMSE): ", enet_emse, "\n")
  cat("Mean Absolute Error (MAE): ", enet_mae, "\n")
  cat("R-squared Score: ", round(enet_r_squared, 3), "\n")
  cat("Adjusted R-squared Score: ", round(enet_adj_r_squared, 3), "\n")


###############################################################################################
  

  # Comparison between LR, RR, Lasso, and Elastic Net models
  compare2 <- data.frame(Method=c('Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression'), 
                         MSE=NA, RMSE = NA, MAE=NA, R_Square = NA, Adjusted_R_Square=NA)
  
  # Fill in the comparison dataframe
  compare2$MSE <- c(round(mse, 2), round(ridge_mse, 2), round(lasso_mse, 2), round(enet_mse, 2))
  compare2$RMSE <- c(round(sqrt(mse), 2), round(sqrt(ridge_mse), 2), round(sqrt(lasso_mse), 2), round(sqrt(enet_mse), 2))
  compare2$MAE <- c(round(mae, 2), round(ridge_mae, 2), round(lasso_mae, 2), round(enet_mae, 2))
  compare2$R_Square <- c(round(r_squared, 2), round(ridge_r_squared, 2), round(lasso_r_squared, 2), round(enet_r_squared, 2))
  compare2$Adjusted_R_Square <- c(round(adj_r_squared, 2), round(ridge_adj_r_squared, 2), round(lasso_adj_r_squared, 2), round(enet_adj_r_squared, 2))
  
  # Display comparison results with kableExtra formatting
  kable(compare2, format = "html") %>%
    kable_styling(bootstrap_options = "striped", full_width = F, position = "center")
  

  
  
  
  
  