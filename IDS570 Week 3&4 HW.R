library(tidyverse)
library(tidytext)
library(quanteda)

circle_raw <- read_file("/Users/chenyiran/Desktop/IDS570/texts/A07594__Circle_of_Commerce.txt")
free_raw   <- read_file("/Users/chenyiran/Desktop/IDS570/texts/B14801__Free_Trade.txt")
third_raw  <- read_file("/Users/chenyiran/Desktop/IDS570/texts/A06785.txt")

# ---- Part I Raw-Count Sentiment Analysis
# Step 1: Tokenize and clean the text

texts <- tibble(
  doc_title = c(
    "Circle_of_Commerce",
    "Free_Trade",
    "Third_Text_A06785"
  ),
  text = c(
    circle_raw,
    free_raw,
    third_raw
  )
)

custom_stop <- c(
  "vnto", "haue", "doo", "hath", "bee", "ye", "thee",
  "hee", "shall", "hast", "doe", "beene", "thereof", "thus"
)

tokens_clean <- texts %>%
  mutate(
    text = str_replace_all(text, "ſ", "s")
  ) %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  filter(!word %in% custom_stop)

head(tokens_clean)

# Step 2: Join tokens to the Bing sentiment dictionary

bing <- get_sentiments("bing")
raw_sentiment <- tokens_clean %>%
  inner_join(bing, by = "word")

raw_sentiment %>%
  count(doc_title, sentiment)

# Step 3: Compute raw sentiment totals
raw_summary <- raw_sentiment %>%
  count(doc_title, sentiment) %>%
  pivot_wider(
    names_from = sentiment,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(
    raw_net_sentiment = positive - negative
  )

raw_summary <- raw_summary %>%
  rename(
    raw_positive = positive,
    raw_negative = negative
  )

raw_summary

# ----- Part II TF-IDF–Weighted Sentiment Analysis
clean_docs <- tokens_clean %>%
  group_by(doc_title) %>%
  summarise(
    text_clean = str_c(word, collapse = " "),
    .groups = "drop"
  )

texts_quanteda <- clean_docs$text_clean
names(texts_quanteda) <- clean_docs$doc_title

corp <- corpus(texts_quanteda)
toks <- tokens(corp)

# Step 4: Compute TF-IDF for words in each document
dfm_mat <- dfm(toks)

dfm_mat
topfeatures(dfm_mat, 25)

tfidf_dfm <- dfm_tfidf(dfm_mat)

tfidf_dfm

# Step 5: Convert the TF-IDF results to a tidy table

tfidf_wide <- convert(
  tfidf_dfm,
  to = "data.frame"
)

head(tfidf_wide)

tfidf_tidy <- tfidf_wide %>%
  pivot_longer(
    cols = -doc_id,
    names_to = "word",
    values_to = "tfidf"
  ) %>%
  filter(tfidf > 0)

names(tfidf_wide)

tfidf_tidy <- tfidf_tidy %>%
  filter(
    doc_id %in% c(
      "Circle_of_Commerce",
      "Free_Trade"
    )
  )

# Step 6: Keep only sentiment-bearing words

tfidf_sentiment <- tfidf_tidy %>%
  inner_join(bing, by = "word")

tfidf_sentiment %>%
  arrange(doc_id, desc(tfidf))

# Step 7: Compute TF-IDF–weighted sentiment totals
tfidf_summary <- tfidf_sentiment %>%
  group_by(doc_id, sentiment) %>%
  summarise(
    tfidf_total = sum(tfidf),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = sentiment,
    values_from = tfidf_total,
    values_fill = 0
  ) %>%
  mutate(
    tfidf_net_sentiment = positive - negative
  )

tfidf_summary <- tfidf_summary %>%
  rename(
    doc_title = doc_id,
    tfidf_positive = positive,
    tfidf_negative = negative
  )

tfidf_summary

# Part III. Compare Raw vs. TF-IDF Sentiment

raw_summary_final <- raw_summary %>%
  filter(
    doc_title %in% c(
      "Circle_of_Commerce",
      "Free_Trade"
    )
  )

sentiment_comparison <- raw_summary_final %>%
  left_join(
    tfidf_summary,
    by = "doc_title"
  )

sentiment_comparison

write_csv(
  sentiment_comparison,
  "~/Downloads/sentiment_comparison.csv"
)

tfidf_sentiment %>%
  group_by(doc_id, sentiment) %>%
  slice_max(tfidf, n = 10, with_ties = FALSE) %>%
  arrange(doc_id, sentiment, desc(tfidf))
