import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np
from lime.lime_tabular import LimeTabularExplainer
from config.db import get_collection, COLLECTION_FEATURE_STORE, load_model
from sklearn.metrics import r2_score

model_lgbm, scaler_lgbm, metadata = load_model('lgbm')
feat = metadata['features']
TARGET_COLS = ['AQI_t+1', 'AQI_t+2', 'AQI_t+3']

docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {'_id': 0}))
df = pd.DataFrame(docs)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
df_clean = df.dropna(subset=feat + TARGET_COLS).reset_index(drop=True)
split_date = df_clean['date'].max() - pd.Timedelta(days=30)
train_df = df_clean[df_clean['date'] <= split_date]
test_df  = df_clean[df_clean['date'] >  split_date].reset_index(drop=True)

X_train = scaler_lgbm.transform(train_df[feat].values)
X_test  = scaler_lgbm.transform(test_df[feat].values)
y_test  = test_df[TARGET_COLS].values

# Day1 predictions for instance selection
preds_d1 = model_lgbm.estimators_[0].predict(X_test)
errors_d1 = np.abs(preds_d1 - y_test[:,0])

# Select 5 instances
SEASON_MAP = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
test_df['season'] = test_df['date'].dt.month.map(SEASON_MAP)

best_idx  = errors_d1.argmin()
worst_idx = errors_d1.argmax()
summer_idx = test_df[test_df['season']=='Summer'].index[0] if len(test_df[test_df['season']=='Summer'])>0 else 0
winter_idx = test_df[test_df['season']=='Winter'].index[0] if len(test_df[test_df['season']=='Winter'])>0 else 1
high_aqi_idx = (test_df['AQI'] > 150).values.nonzero()[0][0] if (test_df['AQI']>150).any() else errors_d1.argsort()[-3]

selected = [
    (best_idx,  'best_prediction'),
    (worst_idx, 'worst_prediction'),
    (summer_idx,'summer_day'),
    (winter_idx,'winter_day'),
    (high_aqi_idx,'high_aqi_day'),
]

explainer_lime = LimeTabularExplainer(
    X_train, feature_names=feat, mode='regression',
    training_labels=train_df[TARGET_COLS[0]].values
)

lime_rows = []
print("\n=== LIME EXPLANATIONS (Day1) ===")
for inst_idx, label in selected:
    inst_idx = min(inst_idx, len(X_test)-1)
    exp = explainer_lime.explain_instance(X_test[inst_idx], model_lgbm.estimators_[0].predict,
                                           num_features=10, num_samples=1000)
    feat_weights = exp.as_list()
    top_feat_name = feat_weights[0][0] if feat_weights else 'unknown'
    top_feat_dir  = 'positive' if (feat_weights[0][1] > 0 if feat_weights else True) else 'negative'
    actual = y_test[inst_idx, 0]
    predicted = preds_d1[inst_idx]
    date_val = str(test_df.iloc[inst_idx]['date'])[:10]
    print(f"  [{label}] date={date_val}  actual={actual:.1f}  pred={predicted:.1f}  "
          f"error={abs(actual-predicted):.1f}  top_feature={top_feat_name} ({top_feat_dir})")
    lime_rows.append({'instance':label,'date':date_val,'actual':actual,'predicted':predicted,
                      'abs_error':abs(actual-predicted),'top_feature':top_feat_name,
                      'top_feature_direction':top_feat_dir})

pd.DataFrame(lime_rows).to_csv('analysis/tables/lime_summary.csv', index=False)
print("Saved analysis/tables/lime_summary.csv")
