from production import load_dfs, save_dfs, update_dfs, run_pred, run_pred_dfs, check
from config import watchlist

import pandas as pd
import os
import sys
import time
import schedule
from datetime import date, datetime
import warnings
warnings.filterwarnings("ignore")



globals().update({
    'watchlist': watchlist,
    'load_dfs': load_dfs,
    'save_dfs': save_dfs,
    'update_dfs': update_dfs,
    'run_pred': run_pred, 
    'run_pred_dfs':run_pred_dfs,
    'check': check,
})

os.makedirs("predictions", exist_ok=True)

# below: thats basically check()

# def daily_prediction():
#     """Run prediction for all tickers after market close"""
#     print("=== Running Daily Prediction ===")

#     dfs = load_dfs()

#     all_preds = []  
#     for ticker in watchlist:
#         print(f"\nPredicting {ticker}")
#         try:
#             df_pred = run_pred(ticker)
#             all_preds.append(df_pred)  # collect results
#         except Exception as e:
#             print(f"{ticker} prediction failed: {e}")

#     # Concatenate all predictions
#     if all_preds:
#         df_all = pd.concat(all_preds)
#         df_all.to_csv(f"predictions/{datetime.today().date().isoformat()}.csv")
#         print(f"\nAll predictions saved to predictions/{datetime.today().date().isoformat()}.csv")
#     else:
#         print("No predictions were made.")

# -----

def weekly_retraining():
    """Retrain all models (e.g. every Friday after close)"""
    print("=== Running Weekly Retraining ===")
    end_date = end_date or date.today().isoformat()

    dfs = update_dfs(watchlist, load_existing=True, force_retrain=True)
    print("Retraining complete. Models updated and saved.")


def setup_schedule():
    schedule.every().monday.at("18:00").do(check)
    schedule.every().tuesday.at("18:00").do(check)
    schedule.every().wednesday.at("18:00").do(check)
    schedule.every().thursday.at("18:00").do(check)
    schedule.every().friday.at("18:00").do(check)
    schedule.every().saturday.at("08:00").do(weekly_retraining)



def main():
    print("Starting scheduler. Press Ctrl+C to stop.\n")
    print("available vars/functions to call (if running python -i main.py in terminal):")
    print("watchlist, load_dfs(), save_dfs(), run_pred(), run_pred_dfs(), check()")

    setup_schedule()
    if hasattr(sys, 'ps1') or sys.flags.interactive:
        print("Interactive shell detected. Scheduler not started automatically.")
    else:
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == '__main__':
    main()
