import xgboost as xgb

def get_model(pos_scale, seed=42):
    model = xgb.XGBClassifier(max_depth=6, learning_rate=0.07,
        scale_pos_weight=pos_scale,
        n_estimators=200,
        subsample=1,
        colsample_bytree=1,
        eta=1, objective='binary:logistic',
        seed=seed,
        eval_metric='auc'
    )
    return model

def get_model2(pos_scale, seed=42):
    model = xgb.XGBClassifier(
        max_depth=7,
        scale_pos_weight=pos_scale,
        learning_rate=0.3,
        n_estimators=200,
        subsample=1,
        colsample_bytree=1,
        nthread=-1,
        seed=seed,
        eval_metric='logloss'
    )
    return model