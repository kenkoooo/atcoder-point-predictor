
# %%
import numpy as np
import pandas as pd
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold

RATED = "rated"
AC_COUNT = "ac_count"
P_ID = "problem_id"
U_ID = "user_id"
IS_AC = "is_ac"
SCORE = "score"
R_POINT = "rated_point"

BLACK_LIST = [
    re.compile(r"^luogu_bot\d+"),
    re.compile(r"^vjudge\d+"),
]


def add_rated_column(contests):
    rate_changable = contests["rate_change"] != '×'
    newer_contest = contests["start_epoch_second"] >= 1468670400
    contests[RATED] = rate_changable & newer_contest
    return contests


def add_is_ac_column(submissions):
    submissions[IS_AC] = submissions["result"] == "AC"
    return submissions


def merge_submissions_contests(submissions, contests):
    merged = pd.merge(submissions, contests,
                      "left", left_on="contest_id", right_index=True)
    merged = merged[[P_ID, U_ID, "point", IS_AC, "contest_id", RATED]]
    return merged


def add_rated_point_column(table):
    rated_point = table[(table[RATED]) & (table[IS_AC])][[
        "point", P_ID]].groupby(P_ID).max()
    rated_point.rename(columns={"point": R_POINT}, inplace=True)
    table = pd.merge(table, rated_point, "left",
                     left_on=P_ID, right_index=True)
    return table, rated_point


def add_ac_count_column(table):
    ac_table = table[table[IS_AC]][[U_ID, P_ID]]
    ac_count = ac_table.groupby(U_ID)[P_ID].nunique()
    ac_count = pd.DataFrame(ac_count).rename(
        columns={P_ID: AC_COUNT})
    table = pd.merge(table, ac_count, "left",
                     left_on=U_ID, right_index=True)
    return table


def add_score_column(table):
    table[SCORE] = 0
    table[SCORE].where(table[IS_AC], 1, inplace=True)
    table[SCORE].mask(table[IS_AC], -1, inplace=True)
    scores = table[[U_ID, P_ID, SCORE]].groupby(
        [U_ID, P_ID]).max().reset_index()
    return table, scores


def filter_user(users):
    filtered_users = []
    for user in users:
        matched = False
        for pattern in BLACK_LIST:
            if pattern.match(user):
                matched = True
        if not matched:
            filtered_users.append(user)
    return filtered_users


# %%
contests = pd.read_json("data/contests.json").set_index("id")
submissions = pd.read_csv("data/atcoder_submissions.csv")
problem_ids = pd.read_json("data/problems.json")["id"].values


# %%
contests = add_rated_column(contests)
submissions = add_is_ac_column(submissions)
table = merge_submissions_contests(submissions, contests)
table, rated_point = add_rated_point_column(table)
table = add_ac_count_column(table)
table, scores = add_score_column(table)


# %%
df = pd.DataFrame(index=problem_ids)
heavy_users = table[(table[RATED]) & (table[AC_COUNT] >= 300)][U_ID]
heavy_users = set(heavy_users)
heavy_users = filter_user(heavy_users)

# %%
for user_id in tqdm(heavy_users):
    user_score = scores[scores[U_ID] == user_id][[P_ID, SCORE]].set_index(
        P_ID).rename(columns={SCORE: user_id})
    df = pd.merge(df, user_score, "left", left_index=True, right_index=True)


# %%
df = pd.merge(df, rated_point, "left", left_index=True, right_index=True)
df.sort_index(inplace=True, axis=0)
df.sort_index(inplace=True, axis=1)

# %%
train_data = df[df[R_POINT].notnull()]
x = train_data.drop(R_POINT, axis=1).values
y = train_data[R_POINT]
kf = KFold(n_splits=5, random_state=71, shuffle=True)
for train_index, test_index in kf.split(train_data):
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]
    model = xgb.XGBRegressor(seed=71)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
