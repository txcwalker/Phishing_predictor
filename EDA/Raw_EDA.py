import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scripts.data_utility import final_dataset

# Importing total dataset from data_utility
df = pd.concat([
    final_dataset["train"].to_pandas(),
    final_dataset["validation"].to_pandas(),
    final_dataset["test"].to_pandas()
], ignore_index=True)

# Defining text length (length of individual observation/message) in data set
df["text_length"] = df["text"].str.len()

# Filtering so only messages of 2500 characters or fewer show
df_filtered = df[df['text_length']<2500]

# Getting words per message
df_filtered["word_count"] = df_filtered["text"].apply(lambda x: len(str(x).split()))

# Splitting into two dfs one for phising/spam and the other for legit messages
ham_df = df_filtered[df_filtered["label"] == 0]
spam_df = df_filtered[df_filtered["label"] == 1]

# Building list of words to skip since they are common articles, punctuations, etc etc
skip_word = ['the','to','and','of','.',',','a','for','in','-','on','is','that','you','i','this','with','be','at','by',
             '_','your',':','/','!','=', 'as','or','are','our','*','we',"'",'|','?','"',')','(','it','not','s',
             'enron','if','was','has','>','=20','he','she','but','have','an','am','can','pm','hou','ect','etc',
             'vince','kaminski','[image]','1400','houston', 'houston,']

# Function to get the top X words
def get_top_words(text_series, min_length=1, n=20, label="output", output_dir="../outputs"):
    # Clean and tokenize all text into a flat list of words
    cleaned_texts = [t.lower() for t in text_series if isinstance(t, str)]
    all_words = ' '.join(cleaned_texts).split()
    filtered_words = [
        word for word in all_words
        if word not in skip_word and len(word) >= min_length
    ]
    total_words = len(filtered_words)
    total_messages = len(cleaned_texts)

    # Count top words
    top_counts = Counter(filtered_words).most_common(n)
    word_df = pd.DataFrame(top_counts, columns=["word", "count"])
    word_df["percent"] = (word_df["count"] / total_words * 100).round(2)

    # Count how many messages contain each word
    word_df["message_count"] = word_df["word"].apply(
        lambda w: sum(w in t for t in cleaned_texts)
    )
    word_df["message_percent"] = (word_df["message_count"] / total_messages * 100).round(2)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=word_df, x="count", y="word", palette="viridis")

    for i, row in word_df.iterrows():
        ax.text(row["count"], i, f"{row['percent']}% | {row['message_percent']}% msgs", va='center', ha='left', fontsize=9)

    plt.title(f"Top {n} Words in {label}")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"top_words_{label}_messages.png")
    plt.savefig(plot_path)
    plt.show()

    return word_df



# Running function to get top words in ham_df/spam_df
ham_filtered_words_count = get_top_words(ham_df['text'],4,25, label = 'Ham_Messages')

# Running function to get top words in ham_df/spam_df
spam_filtered_words_count = get_top_words(spam_df['text'],1,25,label = 'Spam_Messages')

# Function to find top phrases used in ham_df/spam_df
def get_top_phrases(text_series, ngram_range=(1, 1), min_word_length=1, top_n=20, label="output", output_dir="../outputs"):
    cleaned_texts = [t.lower() for t in text_series if isinstance(t, str)]
    all_tokens = ' '.join(cleaned_texts).split()
    filtered_tokens = [word for word in all_tokens if word not in skip_word and len(word) >= min_word_length]

    # Build n-gram counts
    ngram_counts = Counter()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams = zip(*[filtered_tokens[i:] for i in range(n)])
        phrases = [' '.join(ngram) for ngram in ngrams]
        ngram_counts.update(phrases)

    top_counts = ngram_counts.most_common(top_n)

    if not top_counts:
        print(f"No phrases found for {label}.")
        return []

    total_ngrams = sum(ngram_counts.values())
    phrase_df = pd.DataFrame(top_counts, columns=["phrase", "count"])
    phrase_df["percent"] = (phrase_df["count"] / total_ngrams * 100).round(2)

    # Count how many messages contain each phrase
    total_messages = len(cleaned_texts)
    phrase_df["message_count"] = phrase_df["phrase"].apply(
        lambda p: sum(p in t for t in cleaned_texts)
    )
    phrase_df["message_percent"] = (phrase_df["message_count"] / total_messages * 100).round(2)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=phrase_df, x="count", y="phrase", palette="magma")

    """for i, row in phrase_df.iterrows():
        ax.text(row["count"], i, f"{row['percent']}% | {row['message_percent']}% msgs", va='center', ha='left', fontsize=9)"""

    plt.title(f"Top {top_n} Phrases in {label}")
    plt.xlabel("Frequency")
    plt.ylabel("Phrase")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"top_phrases_{label}.png")
    plt.savefig(plot_path)
    plt.show()

    return phrase_df

# Running Top Phrases Function
top_ham_phrases = get_top_phrases(ham_df["text"], ngram_range=(2, 5), min_word_length = 3,top_n = 25, label="ham")
top_spam_phrases = get_top_phrases(spam_df["text"], ngram_range=(2, 5), min_word_length=3, top_n=25, label="spam")


# Words to Check Function
check_words = ['sex','horny','milf','teen','naked']
def count_word_occurrences(text_series, words):
    """
    Counts how many times each word (or phrase) appears in a pandas Series of text.
    Returns a dictionary of word: total_count.
    """
    if isinstance(words, str):
        words = [words]  # Wrap single word in list

    counts = {}
    for word in words:
        word_lower = word.lower()
        count = text_series.str.lower().str.count(rf'\\b{word_lower}\\b').sum()
        counts[word] = int(count)  # ensure int not float from NaNs

    return counts

# Count how many times specific words appear in spam messages
spam_counts = count_word_occurrences(spam_df["text"], check_words)
print("Spam word counts:", spam_counts)

# Count a single word in all messages
# all_counts = count_word_occurrences(df["text"], "unsubscribe")
# print("Unsubscribe count:", all_counts)



# Ensure output directory exists
output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)

df_filtered = df_filtered[df_filtered['text_length'] < 1000]


# --- Ham Plots ---
# Character Count
plt.figure(figsize=(8, 5))
sns.histplot(df_filtered[df_filtered["label"] == 0]["text_length"], bins=50, kde=True, color="skyblue")
plt.title("Ham Message Character Count")
plt.xlabel("Character Count")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ham_character_length_histogram.png"))
plt.show()

# Word Count
plt.figure(figsize=(8, 5))
sns.histplot(df_filtered[df_filtered["label"] == 0]["word_count"], bins=50, kde=True, color="forestgreen")
plt.title("Ham Message Word Count")
plt.xlabel("Word Count")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ham_word_count_histogram.png"))
plt.show()

# --- Spam Plot ---
# Character Count
plt.figure(figsize=(8, 5))
sns.histplot(df_filtered[df_filtered["label"] == 1]["text_length"], bins=50, kde=True, color="salmon")
plt.title("Spam Character Count")
plt.xlabel("Character Count")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spam_character_lengths_histogram.png"))
plt.show()

# Word Count
plt.figure(figsize=(8, 5))
sns.histplot(df_filtered[df_filtered["label"] == 1]["word_count"], bins=50, kde=True, color="cyan")
plt.title("Spam Word Count")
plt.xlabel("Word Count")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spam_word_count_histogram.png"))
plt.show()

# Character Count Comparison Histograms
# Layered
plt.figure(figsize=(8, 5))
sns.histplot(df_filtered[df_filtered["label"] == 0]["text_length"], bins=50, kde=True, color="skyblue", label="Ham")
sns.histplot(df_filtered[df_filtered["label"] == 1]["text_length"], bins=50, kde=True, color="salmon", label="Spam")
plt.title("Ham vs Spam Character Count")
plt.xlabel("Character Count")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "combined_character_lengths.png"))
plt.show()

#Side by Side
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

sns.histplot(df_filtered[df_filtered["label"] == 0]["text_length"], bins=50, kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Ham Character Count")
axes[0].set_xlabel("Character Count")
axes[0].set_ylabel("Count")

sns.histplot(df_filtered[df_filtered["label"] == 1]["text_length"], bins=50, kde=True, ax=axes[1], color="salmon")
axes[1].set_title("Spam Character Count")
axes[1].set_xlabel("Character Count")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "side_by_side_character_lengths.png"))
plt.show()

# Word Count Comparison Histograms
# Layered
plt.figure(figsize=(8, 5))
sns.histplot(df_filtered[df_filtered["label"] == 0]["word_count"], bins=50, kde=True, color="forestgreen", label="Ham")
sns.histplot(df_filtered[df_filtered["label"] == 1]["word_count"], bins=50, kde=True, color="cyan", label="Spam")
plt.title("Ham vs Spam Word Counts")
plt.xlabel("Length")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "combined_word_count.png"))
plt.show()

# Side by Side
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

sns.histplot(df_filtered[df_filtered["label"] == 0]["word_count"], bins=50, kde=True, ax=axes[0], color="forestgreen")
axes[0].set_title("Ham Word Count")
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Count")

sns.histplot(df_filtered[df_filtered["label"] == 1]["word_count"], bins=50, kde=True, ax=axes[1], color="cyan")
axes[1].set_title("Spam Word Count")
axes[1].set_xlabel("Word Count")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "side_by_side_word_count.png"))
plt.show()

