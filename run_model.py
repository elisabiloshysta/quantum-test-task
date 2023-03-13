import pickle
import pandas as pd

if __name__ == "__main__":
    with open('lm3.pkl', 'rb') as f:
        lm3 = pickle.load(f)
        df_features = pd.read_csv("internship_hidden_test.csv")
        pred = lm3.predict(df_features[["6"]])
        pred_df = pd.DataFrame(pred)
        pred_df.columns = ["target"]
        df_final = pd.concat([df_features, pred_df], axis=1)
        df_final.to_csv("internship_hidden_test_preds.csv")
