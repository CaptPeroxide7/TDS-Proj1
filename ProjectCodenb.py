import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Q1. Who are the top 5 users in Melbourne with the highest number of followers? List their login in order, comma-separated
users = pd.read_csv('users.csv')
users.head()
top5 = users.sort_values(by='followers', ascending=False).head()
print(','.join(top5['login'].tolist()))

#Q2. Who are the 5 earliest registered GitHub users in Melbourne? List their login in ascending order of created_at, comma-separated.
users['created_at'] = pd.to_datetime(users['created_at'])
top_earliest = users.sort_values(by='created_at').head()
print(','.join(top_earliest['login'].tolist()))

#Q3. What are the 3 most popular license among these users? Ignore missing licenses. List the license_name in order, comma-separated.
repos = pd.read_csv('repositories.csv')
repos.head()
repos['license_name'].value_counts().head(3)

#Q4. Which company do the majority of these developers work at?
users['company'].value_counts().head(1)

#Q5. Which programming language is most popular among these users?
repos['language'].value_counts().head(1)

#Q6. Which programming language is the second most popular among users who joined after 2020?
users_after_2020 = users[users['created_at'] > '2020-01-01']
users_after_2020.head()
repos_2020 = repos[repos['login'].isin(users_after_2020['login'].tolist())]
repos_2020['language'].value_counts().head()

#Q7. Which language has the highest average number of stars per repository?
avg_stars = repos.groupby('language')['stargazers_count'].mean()
top_lang = avg_stars.idxmax()
top_stars = avg_stars.max()
print(top_lang, top_stars)

#Q8. Let's define leader_strength as followers / (1 + following). Who are the top 5 in terms of leader_strength? List their login in order, comma-separated.
users['leader_strength'] = users['followers'] / (1 + users['following'])
top5_lead = users.sort_values(by='leader_strength', ascending=False).head()
print(','.join(top5_lead['login'].tolist()))

#Q9. What is the correlation between the number of followers and the number of public repositories among users in Melbourne?
correlation = users['followers'].corr(users['public_repos'])
correlation

#Q10. Does creating more repos help users get more followers? Using regression, estimate how many additional followers a user gets per additional public repository.
import csv
followers = []
public_repos = []
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        followers_count = int(row['followers'])
        public_repos_count = int(row['public_repos'])
        followers.append(followers_count)
        public_repos.append(public_repos_count)
if len(followers) > 1 and len(public_repos) > 1:
    slope, intercept = np.polyfit(public_repos, followers, 1)
    
    print(f"{slope:.3f}")
else:
    print("Error")

#Q11. Do people typically enable projects and wikis together? What is the correlation between a repo having projects enabled and having wiki enabled?
if repos['has_projects'].dtype == 'object':
    repos['has_projects'] = repos['has_projects'].map({'true': True, 'false': False})
if repos['has_wiki'].dtype == 'object':
    repos['has_wiki'] = repos['has_wiki'].map({'true': True, 'false': False})
    
correlation = repos['has_projects'].corr(repos['has_wiki'])
    
print(round(correlation, 3))

#Q12. Do hireable users follow more people than those who are not hireable?
hireable_avg_following = users[users['hireable'] == True]['following'].mean()
non_hireable_avg_following = users[users['hireable'] == False]['following'].mean()
difference = hireable_avg_following - non_hireable_avg_following
difference

#Q13. Some developers write long bios. Does that help them get more followers? What's the correlation of the length of their bio (in Unicode characters) with followers? (Ignore people without bios)
from sklearn.linear_model import LinearRegression
users_with_bio = users[(users['bio'].notna()) & (users['bio'] != '')].copy()
users_with_bio.loc[:, 'bio_len'] = users_with_bio['bio'].str.len()

X = users_with_bio['bio_len'].values.reshape(-1,1)
y = users_with_bio['followers']

lr2 = LinearRegression()
lr2.fit(X, y)
lr2.coef_[0]

#Q14. Who created the most repositories on weekends (UTC)? List the top 5 users' login in order, comma-separated
import csv
from collections import Counter
from datetime import datetime

weekend_repo_counts = Counter()

with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        created_at = row.get('created_at', '')
        if created_at:
            created_date = datetime.fromisoformat(created_at[:-1])  
            
            if created_date.weekday() in [5, 6]:
                user_login = row['login']
                weekend_repo_counts[user_login] += 1  

top_users = weekend_repo_counts.most_common(5)

top_logins = [user[0] for user in top_users]

print(','.join(top_logins))

#Q15. Do people who are hireable share their email addresses more often?
fraction_hierable = users[users['hireable'] == True]['email'].notna().mean()
fraction_non_hierable = users[users['hireable'] == False]['email'].notna().mean()
diff = fraction_hierable - fraction_non_hierable
diff

#Q16. Let's assume that the last word in a user's name is their surname (ignore missing names, trim and split by whitespace.) What's the most common surname? (If there's a tie, list them all, comma-separated, alphabetically)
new_users = users[users['name'].notna()].copy()
new_users['surname'] = new_users['name'].str.split().str[-1].str.strip()
surname_counts = new_users['surname'].value_counts()
max_count = surname_counts.max()
common_surnames = surname_counts[surname_counts == max_count].index.tolist()
common_surnames.sort()
print(','.join(common_surnames))

