
# %%
import numpy as np
import pandas as pd
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from logzero import logger

RATED = "rated"
AC_COUNT = "ac_count"
P_ID = "problem_id"
U_ID = "user_id"
IS_AC = "is_ac"
SCORE = "score"
POINT = "point"

FIRST_AGC_START = 1468670400
HEAVY_USER_AC_COUNT_THRESHOLD = 200

NORMAL_USER_PATTERN = r"^(?!(luogu_bot|vjudge)\d+)"


def add_rated_column(contests):
    rate_changable = contests["rate_change"] != '×'
    newer_contest = contests["start_epoch_second"] >= 1468670400
    contests[RATED] = rate_changable & newer_contest
    return contests


def add_is_ac_column(submissions):
    submissions[IS_AC] = submissions["result"] == "AC"
    return submissions


def merge_submissions_contests(submissions, contests):
    merged = pd.merge(submissions, contests, "left",
                      left_on="contest_id", right_index=True, copy=False)
    merged = merged[[P_ID, U_ID, POINT, IS_AC, RATED]]
    return merged


def add_rated_point_column(table):
    rated_point = table[(table[RATED]) & (table[IS_AC])][[
        POINT, P_ID]].groupby(P_ID).max()
    table = pd.merge(table, rated_point, "left",
                     left_on=P_ID, right_index=True, copy=False)
    return table, rated_point


def add_ac_count_column(table):
    ac_table = table[table[IS_AC]][[U_ID, P_ID]]
    ac_count = ac_table.groupby(U_ID)[P_ID].nunique()
    ac_count = pd.DataFrame(ac_count).rename(
        columns={P_ID: AC_COUNT})
    table = pd.merge(table, ac_count, "left",
                     left_on=U_ID, right_index=True, copy=False)
    return table


def create_one_hot_table(t):
    p = pd.get_dummies(t[[P_ID, U_ID]], columns=[U_ID],
                       dtype='int8', prefix="", prefix_sep="")
    p = p.groupby(P_ID).max()
    p.reset_index(inplace=True)
    return p


def create_df():
    logger.info("loading data...")
    contests = pd.read_json("data/contests.json").set_index("id")
    submissions = pd.read_csv("data/atcoder_submissions.csv")

    logger.info("merging tables...")
    contests = add_rated_column(contests)
    submissions = add_is_ac_column(submissions)
    table = merge_submissions_contests(submissions, contests)
    table, rated_point = add_rated_point_column(table)
    table = add_ac_count_column(table)
    table = table[(table[AC_COUNT] >= HEAVY_USER_AC_COUNT_THRESHOLD)
                  & (table[U_ID].str.contains(NORMAL_USER_PATTERN))]

    logger.info("calculating scores...")
    t = table[[P_ID, U_ID, IS_AC]].drop_duplicates()
    t[SCORE] = 0
    t[SCORE].where(t[IS_AC], -1, inplace=True)
    t[SCORE].mask(t[IS_AC], 1, inplace=True)
    t[[P_ID, U_ID, SCORE]].groupby([P_ID, U_ID]).max().reset_index()

    logger.info("creating one hot table...")
    p = create_one_hot_table(t[t[SCORE] > 0])
    n = create_one_hot_table(t[t[SCORE] < 0]).replace(1, -1)

    df = pd.concat([p, n]).groupby(P_ID).max()
    df = pd.merge(df, rated_point, "left", left_index=True, right_index=True)

    return df


# %%
df = create_df()

# %%
logger.info("cross-validationg...")
train_data = df[df[POINT].notnull()]
x = train_data.drop(POINT, axis=1).values
y = train_data[POINT]
kf = KFold(n_splits=5, random_state=71, shuffle=True)
for train_index, test_index in kf.split(train_data):
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]
    model = xgb.XGBRegressor(seed=71)
    model.fit(x_train, y_train)
    logger.info(model.score(x_test, y_test))

# %%
importance = pd.DataFrame(model.feature_importances_,
                          index=df.drop(POINT, axis=1).columns,
                          columns=["importance"])
importance = importance[importance["importance"] > 0.0].sort_values(
    by="importance", ascending=False)
logger.info(importance)
