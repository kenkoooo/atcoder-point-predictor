
# %%
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


# %%
problems = pd.read_json("data/problems.json")
contests = pd.read_json("data/contests.json")
submissions = pd.read_csv("data/atcoder_submissions.csv")


# %%
contests["rated"] = (contests["rate_change"] != '×') & (
    contests["start_epoch_second"] >= 1468670400)


# %%
table = pd.merge(submissions, contests, "left", left_on="contest_id", right_on="id")[
    ["problem_id", "user_id", "point", "result", "contest_id", "rated"]]


# %%
rated_point = table[(table["rated"]) & (table["result"] == "AC")][[
    "point", "problem_id"]].groupby("problem_id").max()
rated_point.rename(columns={"point": "rated_point"}, inplace=True)
table = pd.merge(table, rated_point, "left",
                 left_on="problem_id", right_index=True)


# %%
ac_count = table[table["result"] == "AC"][["user_id", "problem_id"]].groupby(
    "user_id")["problem_id"].nunique()
ac_count = pd.DataFrame(ac_count).rename(columns={"problem_id": "ac_count"})
table = pd.merge(table, ac_count, "left", left_on="user_id", right_index=True)


# %%
table["score"] = 0
table["score"].where(table["result"] == "AC", 1, inplace=True)
table["score"].where(table["result"] != "AC", -1, inplace=True)

# %%
scores = table[["user_id", "problem_id", "score"]].groupby(
    ["user_id", "problem_id"]).max().reset_index()

# %%
df = pd.DataFrame(index=problems["id"].values)
# %%
heavy_users = table[(table["rated"]) & (table["ac_count"] >= 300)]["user_id"]
heavy_users = set(heavy_users)


# %%
for user_id in heavy_users:
    user_score = scores[scores["user_id"] == user_id][["problem_id", "score"]].set_index(
        "problem_id").rename(columns={"score": user_id})
    df = pd.merge(df, user_score, "left", left_index=True, right_index=True)


# %%
df = pd.merge(df, rated_point, "left", left_index=True, right_index=True)


# %%
df.sort_index(inplace=True)

# %%
# Cross validation
train_data = df[df["rated_point"].notnull()]
train, test = train_test_split(train_data)

x_train = train.iloc[:, :-1].values
x_test = test.iloc[:, :-1].values
y_train = train.loc[:, "rated_point"].values
y_test = test.loc[:, "rated_point"].values

model = xgb.XGBRegressor()
model.fit(x_train, y_train)
model.score(x_test, y_test)

# %%
test["predicted_point"] = model.predict(x_test)
test[["rated_point", "predicted_point"]]
