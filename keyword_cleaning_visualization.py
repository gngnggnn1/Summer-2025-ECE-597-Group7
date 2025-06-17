import pandas as pd
import string

data = pd.read_csv('CaptstoneProjectData_2025.csv')
data = data.iloc[:,:2]


# count nan content and subject
nan_count_col1 = data.iloc[:, 0].isna().sum()
nan_count_col2 = data.iloc[:, 1].isna().sum()

print(f"num of nan in Subject: {nan_count_col1}") # 5
print(f"num of nan in Body: {nan_count_col2}") # 109

# find how many emails from the outside of UVic email system.

notice = "Notice: This message was sent from outside the University of Victoria email system. Please be cautious with links and sensitive information."
col = data.iloc[:, 1].fillna('')
mask = col.str.startswith(notice)
count = mask.sum()
print(f"email from outside system: {count}") # 993
data.loc[mask, data.columns[1]] = data.loc[mask, data.columns[1]].str[len(notice):].str.lstrip()
data.iloc[:, 0] = data.iloc[:, 0].fillna('')

# remove the notice above, count the keywords
subject = data.iloc[:,0]
body = data.iloc[:,1]
# 转小写 + 去标点 + 拆词
subject_words = subject.str.lower().str.split().explode()
subject_words = subject_words.str.strip(string.punctuation)

# 统计词频
subject_counts = subject_words.value_counts()
print("Subject Top 20 Words:")
print(subject_counts.head(20))

body_words = body.str.lower().str.split().explode()
body_words = body_words.str.strip(string.punctuation)

body_counts = body_words.value_counts()
print("Body Top 20 Words:")
print(body_counts.head(20))

body_words = body.str.lower().str.split().explode()
body_words = body_words.str.strip(string.punctuation)

body_counts = body_words.value_counts()
print("Body Top 20 Words:")
print(body_counts.head(20))

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

subject_counts_filtered = subject_words[~subject_words.isin(stop)].value_counts()
subject_counts_filtered = subject_counts_filtered[subject_counts_filtered.index != '']
body_counts_filtered = body_words[~body_words.isin(stop)].value_counts()
body_counts_filtered = body_counts_filtered[body_counts_filtered.index != '']
body_counts_filtered = body_counts_filtered.drop(body_counts_filtered.index[9])

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 生成 subject 词云
wordcloud_subject = WordCloud(width=800, height=400, background_color='white')\
    .generate_from_frequencies(subject_counts_filtered.to_dict())

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_subject, interpolation='bilinear')
plt.axis('off')
plt.title("Subject Keywords Word Cloud")
# plt.show()
# 保存 subject 词云为图片
wordcloud_subject.to_file("subject_wordcloud.png")

# 生成并保存 body 词云为图片
wordcloud_body = WordCloud(width=800, height=400, background_color='white')\
    .generate_from_frequencies(body_counts_filtered.to_dict())
wordcloud_body.to_file("body_wordcloud.png")

import matplotlib
matplotlib.use('Agg')  # 强制使用无GUI后端
# 画柱状图：Subject Top 20
top_subject = subject_counts_filtered.head(20)
plt.figure(figsize=(12, 6))
top_subject.plot(kind='bar', color='skyblue')
plt.title("Top 20 Keywords in Subject")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("subject_top20_keywords.png")
plt.close()

# 画柱状图：Body Top 20
top_body = body_counts_filtered.head(20)
plt.figure(figsize=(12, 6))
top_body.plot(kind='bar', color='lightcoral')
plt.title("Top 20 Keywords in Body")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("body_top20_keywords.png")
plt.close()

# how

# 计算 subject 和 body 的平均词数
subject_word_counts = subject.str.split().apply(len)
body_word_counts = body.str.split().apply(len)

avg_subject_words = subject_word_counts.mean()
avg_body_words = body_word_counts.mean()

print(f"Average number of words in subject: {avg_subject_words:.2f}")
print(f"Average number of words in body: {avg_body_words:.2f}")
