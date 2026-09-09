# Setting up
# Load libraries
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(forcats)
library(tibble)
library(scales)

# Load text files
file_a <- "/Users/chenyiran/Desktop/IDS570/texts/A07594__Circle_of_Commerce.txt"
file_b <- "/Users/chenyiran/Desktop/IDS570/texts/B14801__Free_Trade.txt"

# Read the raw text files into R
text_a <- read_file(file_a)
text_b <- read_file(file_b)

# Combine into a tibble for tidytext workflows
texts <- tibble(
  doc_title = c("Text A", "Text B"),
  text = c(text_a, text_b)
)

# 1.Diagnostic table
# Create a tibble called words that contains one word token per row BEFORE STOPWORDS REMOVAL
words <- texts %>%
  unnest_tokens(word, text) %>%
  mutate(word = str_to_lower(word))

# Create a stopword list
data("stop_words")

# Add project-specific stopwords
custom_stopwords <- tibble(
  word = c(
    "vnto", "haue", "doo", "hath", "bee", "ye", "thee"
  )
)

all_stopwords <- bind_rows(stop_words, custom_stopwords) %>%
  distinct(word)

# Create another tibble for AFTER REMOVAL
words_stop_removed <- words %>%
  anti_join(all_stopwords, by = "word")

words_stop_removed

# Counting numbers of tokens for two tibbles
tokens_before <- words %>%
  count(doc_title, name = "total_tokens", sort = TRUE)

tokens_after <- words_stop_removed %>%
  count(doc_title, name = "tokens_after_stopwords", sort = TRUE)

# Combining counts into one table
corpus_diagnostics <- tokens_before %>%
  left_join(tokens_after, by = "doc_title") %>%
  mutate(
    tokens_removed = total_tokens - tokens_after_stopwords,
    percent_removed = tokens_removed / total_tokens * 100
  )

corpus_diagnostics

#2. Compare the most frequent words before and after stopword removal
# Count word frequencies in both texts before removal
word_counts_before <- words %>%
  count(doc_title, word, sort = TRUE)

#Identify the 15 most frequent words
plot_n_words <- 15

word_comparison_before <- word_counts_before %>%
  pivot_wider(
    names_from = doc_title,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(max_n = pmax(`Text A`, `Text B`)) %>%
  arrange(desc(max_n))

#Create visualization
word_plot_before <- word_comparison_before %>%
  slice_head(n = plot_n_words) %>%
  pivot_longer(
    cols = c(`Text A`, `Text B`),
    names_to = "doc_title",
    values_to = "n"
  ) %>%
  mutate(word = fct_reorder(word, n, .fun = max))

ggplot(word_plot_before, aes(x = n, y = word)) +
  geom_col() +
  facet_wrap(~ doc_title, scales = "free_x") +
  labs(
    title = "Most frequent words before stopword removal",
    subtitle = "Top 15 words by maximum frequency across both texts",
    x = "Word frequency",
    y = NULL
  ) +
  theme_minimal()

#Repeat for texts after removal
word_counts_after <- words_stop_removed %>%
  count(doc_title, word, sort = TRUE)

word_comparison_after <- word_counts_after %>%
  pivot_wider(
    names_from = doc_title,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(max_n = pmax(`Text A`, `Text B`)) %>%
  arrange(desc(max_n))

word_plot_after <- word_comparison_after %>%
  slice_head(n = plot_n_words) %>%
  pivot_longer(
    cols = c(`Text A`, `Text B`),
    names_to = "doc_title",
    values_to = "n"
  ) %>%
  mutate(word = fct_reorder(word, n, .fun = max))

ggplot(word_plot_after, aes(x = n, y = word)) +
  geom_col() +
  facet_wrap(~ doc_title, scales = "free_x") +
  labs(
    title = "Most frequent words after stopword removal",
    subtitle = "Top 15 words by maximum frequency across both texts",
    x = "Word frequency",
    y = NULL
  ) +
  theme_minimal()


# 3.Remove ONLY the standard stopwords
# Remove standard stopwords from original texts
words_standard_stop <- words %>%
  anti_join(stop_words, by = "word")
  
top15_standard <- words_standard_stop %>%
  count(word, sort = TRUE) %>%
  slice_head(n = 15)

# Compare with also removing custom stopwords
top15_all_stopwords <- words_stop_removed %>%
  count(word, sort = TRUE) %>%
  slice_head(n = 15)

# Make a joined table to compare choice in stopwords
comparison_table <- full_join(
  top15_standard,
  top15_all_stopwords,
  by = "word",
  suffix = c("_standard_only", "_standard_custom")
)

comparison_table

