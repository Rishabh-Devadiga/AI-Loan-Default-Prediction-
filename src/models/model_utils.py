def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)

def extract_feature_importance(model, feature_names):
    import pandas as pd
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = model.coef_[0]
    return pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)

def get_model_coefficients(model, feature_names):
    if hasattr(model, 'coef_'):
        return model.coef_[0]
    else:
        raise ValueError("Model does not have coefficients.")