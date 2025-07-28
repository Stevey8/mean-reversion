from production import build_dict, load_dfs, save_dfs, update_dfs, run_pred, run_pred_dfs, check
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
    'build_dict': build_dict,
    'load_dfs': load_dfs,
    'save_dfs': save_dfs,
    'update_dfs': update_dfs,
    'run_pred': run_pred, 
    'run_pred_dfs':run_pred_dfs,
    'check': check,
})

os.makedirs("predictions", exist_ok=True)

def weekly_retraining():
    """Retrain all models (between friday after market close to monday before market open)"""
    print("=== Running Weekly Retraining ===")
    end_date = end_date or date.today().isoformat()

    dfs = update_dfs(watchlist, load_existing=True, force_retrain=True)
    print("Retraining complete. Models updated and saved.")


def setup_schedule():
    schedule.every().friday.at("22:00").do(weekly_retraining)
    schedule.every().sunday.at("22:00").do(check)
    schedule.every().monday.at("22:00").do(check)
    schedule.every().tuesday.at("22:00").do(check)
    schedule.every().wednesday.at("22:00").do(check)
    schedule.every().thursday.at("22:00").do(check)
    



def main():
    print("Starting scheduler. Press Ctrl+C to stop.\n")
    print("available vars/functions to call (if running python -i main.py in terminal):")
    print("watchlist, build_dict(), load_dfs(), save_dfs(), update_dfs(), run_pred(), run_pred_dfs(), check()")

    setup_schedule()
    if hasattr(sys, 'ps1') or sys.flags.interactive:
        print("Interactive shell detected. Scheduler not started automatically.")
    else:
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == '__main__':
    main()
