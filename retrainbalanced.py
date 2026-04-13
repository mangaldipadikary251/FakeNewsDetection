
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# import pickle


# from sklearn.model_selection import train_test_split
# # Now your command will work:
# #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# # 1. Load the updated dataset
# df = pd.read_csv(r"C:\dscode\project_6th_sem\master_dataset.csv")

# # 2. Balance the classes
# df_fake = df[df['label'] == 0]
# df_real = df[df['label'] == 1].sample(n=len(df_fake), random_state=7) # Match Fake count
# df_balanced = pd.concat([df_real, df_fake]).sample(frac=1).reset_index(drop=True)

# print(f"New Balanced Counts:\n{df_balanced['label'].value_counts()}")

# # 3. Train the model
# vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# x = vectorizer.fit_transform(df_balanced['text'].astype('U'))
# y = df_balanced['label']
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
# model = PassiveAggressiveClassifier(max_iter=50)
# model.fit(x_train, y_train)

# # 4. Save the "Balanced" Brain
# pickle.dump(model, open(r"C:\dscode\project_6th_sem\models\fakenews_model.pkl", 'wb'))
# pickle.dump(vectorizer, open(r"C:\dscode\project_6th_sem\models\tfidf_vectorizer.pkl", 'wb'))

# print("🚀 Balanced Model Saved! It is now ready for your Web App.")


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# import pickle

# # 1. Load data
# df = pd.read_csv(r"C:\dscode\project_6th_sem\master_dataset.csv")

# # --- DEBUGGING CHECK: SEE WHAT IS ACTUALLY IN YOUR FILE ---
# print("Column Names found:", df.columns.tolist())
# print("Value Counts for 'label':")
# print(df['label'].value_counts()) 

# # 2. Check if we actually have data
# if len(df[df['label'] == 0]) == 0 or len(df[df['label'] == 1]) == 0:
#     print("❌ ERROR: One of your classes (0 or 1) is empty!")
#     print("Check if your CSV labels are actually 0/1 or something else.")
# else:
#     # 3. Balancing logic
#     df_fake = df[df['label'] == 0]
#     df_real = df[df['label'] == 1].sample(n=len(df_fake), random_state=7)
#     df_balanced = pd.concat([df_real, df_fake]).sample(frac=1).reset_index(drop=True)

#     print(f"✅ Success! Balanced Dataset Size: {len(df_balanced)}")

#     # 4. Train the model
#     # We use stop_words=None first to ensure we don't get the 'Empty Vocabulary' error
#     vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
#     x = vectorizer.fit_transform(df_balanced['text'].values.astype('U'))
#     y = df_balanced['label']

#     model = PassiveAggressiveClassifier(max_iter=50)
#     model.fit(x, y)

#     # 5. Save
#     pickle.dump(model, open(r"C:\dscode\project_6th_sem\models\fakenews_model.pkl", 'wb'))
#     pickle.dump(vectorizer, open(r"C:\dscode\project_6th_sem\models\tfidf_vectorizer.pkl", 'wb'))
#     print("🚀 Model successfully balanced and saved!")


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# import pickle

# # 1. Load data
# df = pd.read_csv(r"C:\dscode\project_6th_sem\master_dataset.csv")

# # 2. CLEANUP: Convert words to numbers
# # This turns 'fake' into 0 and 'real' into 1
# df['label'] = df['label'].replace({'fake': 0, 'real': 1})

# # Convert the column to numeric just in case some are strings
# df['label'] = pd.to_numeric(df['label'], errors='coerce')

# # Drop any rows that are missing text or label after conversion
# df = df.dropna(subset=['label', 'text'])

# print("--- After Cleanup ---")
# print(df['label'].value_counts())

# # 3. Balancing logic
# # Now that labels are unified, we can separate them properly
# df_fake = df[df['label'] == 0]
# df_real = df[df['label'] == 1].sample(n=len(df_fake), random_state=7)
# df_balanced = pd.concat([df_real, df_fake]).sample(frac=1).reset_index(drop=True)

# print(f"✅ Balanced Dataset Size: {len(df_balanced)}")

# # 4. Train the model
# vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# x = vectorizer.fit_transform(df_balanced['text'].values.astype('U'))
# y = df_balanced['label']

# model = PassiveAggressiveClassifier(max_iter=50)
# model.fit(x, y)

# # 5. Save the upgraded models
# pickle.dump(model, open(r"C:\dscode\project_6th_sem\models\fakenews_model.pkl", 'wb'))
# pickle.dump(vectorizer, open(r"C:\dscode\project_6th_sem\models\tfidf_vectorizer.pkl", 'wb'))

# print("🚀 SUCCESS! Your model is now trained and perfectly balanced.")


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# 1. Load data
df = pd.read_csv(r"C:\dscode\project_6th_sem\master_dataset.csv")

# 2. CLEANUP: Convert words to numbers
df['label'] = df['label'].replace({'fake': 0, 'real': 1})
df['label'] = pd.to_numeric(df['label'], errors='coerce')

# Ensure 'text' is string and drop missing values
df['text'] = df['text'].fillna('').astype(str)
df = df.dropna(subset=['label'])

# 3. Balancing logic
df_fake = df[df['label'] == 0]
df_real = df[df['label'] == 1].sample(n=len(df_fake), random_state=7)
df_balanced = pd.concat([df_real, df_fake]).sample(frac=1).reset_index(drop=True)

print(f"✅ Balanced Dataset Size: {len(df_balanced)}")

# 4. Train the model (Memory Efficient Way)
# We use a list of strings instead of a numpy Unicode array to save RAM
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x = vectorizer.fit_transform(df_balanced['text'].tolist()) 
y = df_balanced['label']

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x, y)

# 5. Save the upgraded models
pickle.dump(model, open(r"C:\dscode\project_6th_sem\models\fakenews_model.pkl", 'wb'))
pickle.dump(vectorizer, open(r"C:\dscode\project_6th_sem\models\tfidf_vectorizer.pkl", 'wb'))

print("🚀 SUCCESS! Your model is now trained and saved.")